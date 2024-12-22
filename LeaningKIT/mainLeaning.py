import os
import logging
import shutil
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import mobilenet_v3_large
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast
from PIL import Image
from ultralytics import YOLO
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, precision_score, recall_score

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        valid_images = self.validate_dataset(image_dir, label_dir)
        self.image_names = list(valid_images)

    def validate_dataset(self, image_dir, label_dir):
        image_files = {os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))}
        label_files = {os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.txt')}
        return image_files.intersection(label_files)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name + ".jpg")
        label_path = os.path.join(self.label_dir, image_name + ".txt")

        try:
            image = Image.open(image_path).convert("RGB")
            with open(label_path, 'r') as file:
                lines = file.readlines()
                class_id = int(lines[0].split()[0])

            if self.transform:
                image = self.transform(image)

            return image, class_id
        except Exception as e:
            logging.error(f"Error loading data for {image_name}: {e}")
            raise

class ComponentDetectionPipeline:
    def __init__(self, yolo_model_path='yolov8s.pt', classification_model_path='classification_model.pth', num_classes=5):
        self.yolo_model_path = yolo_model_path
        self.classification_model_path = classification_model_path
        self.num_classes = num_classes

        # Đường dẫn lưu mô hình YOLO sau khi huấn luyện
        self.detection_model_path = os.path.join("databaseKIT", "best.pt")

        # Load YOLO model
        if os.path.exists(self.detection_model_path):
            self.detection_model = YOLO(self.detection_model_path)
            logging.info(f"Loaded YOLO model from '{self.detection_model_path}'")
        else:
            self.detection_model = YOLO(self.yolo_model_path)
            logging.info(f"Loaded YOLO model from '{self.yolo_model_path}'")

        # Load Classification model (MobileNetV3)
        self.classification_model = mobilenet_v3_large(pretrained=True)
        self.classification_model.classifier[3] = torch.nn.Linear(self.classification_model.classifier[3].in_features, num_classes)

        if os.path.exists(self.classification_model_path):
            state_dict = torch.load(self.classification_model_path)
            self.classification_model.load_state_dict(state_dict, strict=False)
            logging.info(f"Loaded classification model from '{self.classification_model_path}'")
        else:
            logging.warning(f"No classification model found at '{self.classification_model_path}'. Training is required.")

        self.classification_model.eval()

    def train_classification_model(self, train_dir, val_dir, output_path, epochs=10, batch_size=32, lr=0.001, continue_training=False):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        train_dataset = CustomDataset(train_dir, './dataset/labels/train', transform)
        val_dataset = CustomDataset(val_dir, './dataset/labels/val', transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classification_model = self.classification_model.to(device)

        optimizer = Adam(self.classification_model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        criterion = CrossEntropyLoss()
        scaler = GradScaler()
        writer = SummaryWriter(log_dir="logs")

        if continue_training and os.path.exists(output_path):
            self.classification_model.load_state_dict(torch.load(output_path))
            logging.info(f"Continuing training classification model from '{output_path}'")

        best_loss = float('inf')
        for epoch in range(epochs):
            total_loss = 0
            self.classification_model.train()

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                with autocast():
                    outputs = self.classification_model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            writer.add_scalar("Loss/train", avg_loss, epoch)
            logging.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}")

            # Validation Step
            self.classification_model.eval()
            val_loss = 0
            all_preds, all_labels = [], []

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = self.classification_model(inputs)
                    val_loss += criterion(outputs, labels).item()

                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            val_loss /= len(val_loader)
            f1 = f1_score(all_labels, all_preds, average='weighted')
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("F1/val", f1, epoch)
            logging.info(f"Validation Loss: {val_loss}, F1-Score: {f1}")

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.classification_model.state_dict(), output_path)
                logging.info(f"Saved improved model to '{output_path}'")

            scheduler.step()

        writer.close()

    def train_yolo_model(self, data_yaml, epochs=50, continue_training=False):
        project = "output"
        name = "final_train"

        # Sao lưu file best.pt nếu tồn tại
        backup_best_path = "backup_best.pt"
        best_model_path = os.path.join(project, name, "weights", "best.pt")
        if os.path.exists(best_model_path):
            shutil.copy(best_model_path, backup_best_path)
            logging.info(f"Backup existing best model to '{backup_best_path}'")

        # Xóa thư mục "output" nếu tồn tại
        if os.path.exists(project):
            shutil.rmtree(project)
            logging.info(f"Removed existing project directory: {project}")

        # Khôi phục file backup_best.pt nếu tiếp tục huấn luyện
        if continue_training and os.path.exists(backup_best_path):
            os.makedirs(os.path.join(project, name, "weights"), exist_ok=True)
            shutil.move(backup_best_path, os.path.join(project, name, "weights", "best.pt"))
            logging.info("Restored best model for further training")

        model = YOLO(self.detection_model_path if continue_training else self.yolo_model_path)
        model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=960,
            save=True,
            project=project,
            name=name,
            augment=True
        )

        best_model_path = os.path.join(project, name, "weights", "best.pt")
        if os.path.exists(best_model_path):
            os.makedirs(os.path.dirname(self.detection_model_path), exist_ok=True)
            shutil.copy(best_model_path, self.detection_model_path)
            logging.info(f"Best YOLO model saved to '{self.detection_model_path}'")
        else:
            logging.error(f"Best model not found at '{best_model_path}'")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    pipeline = ComponentDetectionPipeline(
        yolo_model_path="yolov8s.pt",
        classification_model_path="classification_model.pth",
        num_classes=5
    )

    # Validate dataset
    train_images = "./dataset/images/train"
    train_labels = "./dataset/labels/train"
    val_images = "./dataset/images/val"
    val_labels = "./dataset/labels/val"

    valid_train = CustomDataset(train_images, train_labels).validate_dataset(train_images, train_labels)
    valid_val = CustomDataset(val_images, val_labels).validate_dataset(val_images, val_labels)

    if not valid_train or not valid_val:
        logging.error("Dataset validation failed. Ensure images and labels are correctly matched.")
        exit(1)

    # Train Classification Model
    pipeline.train_classification_model(
        train_dir=train_images,
        val_dir=val_images,
        output_path="classification_model.pth",
        epochs=10,
        batch_size=16,
        lr=0.001,
        continue_training=False
    )

    # Train YOLO Model
    data_yaml_path = "./dataset/data.yaml"
    pipeline.train_yolo_model(
        data_yaml=data_yaml_path,
        epochs=20,
        continue_training=False
    )

    logging.info("Pipeline execution completed.")
