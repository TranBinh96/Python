import os
import logging
import shutil
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast
from PIL import Image
import cv2
from ultralytics import YOLO
from torch.utils.tensorboard import SummaryWriter

# Logging configuration
logging.basicConfig(
    filename="training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_names = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        try:
            image_name = self.image_names[idx]
            image_path = os.path.join(self.image_dir, image_name)
            label_path = os.path.join(self.label_dir, os.path.splitext(image_name)[0] + '.txt')

            if not os.path.exists(label_path):
                logging.warning(f"Label file missing for {image_name}. Skipping.")
                return None

            image = Image.open(image_path).convert("RGB")
            with open(label_path, 'r') as file:
                lines = file.readlines()
                if not lines:
                    logging.warning(f"Empty label file: {label_path}. Skipping.")
                    return None
                class_id = int(lines[0].split()[0])  # Lấy class_id của đối tượng đầu tiên

            if self.transform:
                image = self.transform(image)

            return image, class_id
        except Exception as e:
            logging.error(f"Error loading data for index {idx}: {e}")
            return None


class ComponentDetectionPipeline:
    def __init__(self, yolo_model_path='yolov8s.pt', classification_model_path='classification_model.pth', num_classes=5):
        self.yolo_model_path = yolo_model_path
        self.classification_model_path = classification_model_path
        self.num_classes = num_classes

        # Tải YOLO model
        if os.path.exists('databaseKIT/best.pt'):
            self.detection_model = YOLO('databaseKIT/best.pt')
            logging.info("Loaded YOLO model from 'databaseKIT/best.pt'")
        else:
            self.detection_model = YOLO(self.yolo_model_path)
            logging.info(f"Loaded YOLO model from {self.yolo_model_path}")

        # Tải classification model
        self.classification_model = models.resnet50(pretrained=True)
        self.classification_model.fc = torch.nn.Linear(self.classification_model.fc.in_features, num_classes)

        if os.path.exists(self.classification_model_path):
            self.classification_model.load_state_dict(torch.load(self.classification_model_path))
            logging.info("Classification model loaded successfully.")
        else:
            logging.warning(f"Classification model not found at {self.classification_model_path}. Training is required.")

        self.classification_model.eval()

    def clean_train_directory(self, project="output", name="final_train"):
        """
        Xóa thư mục huấn luyện nếu đã tồn tại.
        """
        train_dir = os.path.join(project, name)
        if os.path.exists(train_dir):
            try:
                shutil.rmtree(train_dir)
                logging.info(f"Deleted existing training directory: {train_dir}")
            except Exception as e:
                logging.error(f"Error deleting directory {train_dir}: {e}")
        else:
            logging.info(f"No existing training directory to delete: {train_dir}")

    def train_classification_model(self, train_dir, val_dir, output_path, epochs=10, batch_size=32, lr=0.001):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
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
        criterion = CrossEntropyLoss()
        scaler = GradScaler()
        writer = SummaryWriter(log_dir="logs")

        for epoch in range(epochs):
            total_loss = 0
            self.classification_model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                with autocast():  # Mixed precision
                    outputs = self.classification_model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            writer.add_scalar("Loss/train", avg_loss, epoch)
            logging.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}")

        torch.save(self.classification_model.state_dict(), output_path)
        logging.info(f"Classification model saved to {output_path}")
        writer.close()

    def train_yolo_model(self, data_yaml, epochs=50):
        project = "output"
        name = "final_train"

        # Xóa thư mục huấn luyện cũ
        self.clean_train_directory(project=project, name=name)

        if os.path.exists('databaseKIT/best.pt'):
            model = YOLO('databaseKIT/best.pt')
            logging.info("Continuing YOLO training from 'databaseKIT/best.pt'")
        else:
            model = YOLO(self.yolo_model_path)
            logging.info(f"Starting YOLO training from {self.yolo_model_path}")

        # Huấn luyện YOLO
        model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=640,
            save=True,
            project=project,
            name=name,
            augment=True
        )

        # Sao lưu mô hình tốt nhất
        best_model_path = os.path.join(project, name, "weights", "best.pt")
        if os.path.exists(best_model_path):
            os.makedirs("databaseKIT", exist_ok=True)
            try:
                shutil.copy(best_model_path, "databaseKIT/best.pt")
                logging.info(f"Best YOLO model successfully copied to 'databaseKIT/best.pt'")
            except Exception as e:
                logging.error(f"Error copying best model: {e}")
        else:
            logging.error(f"Best model not found at {best_model_path}.")


def validate_dataset(image_dir, label_dir):
    image_files = {os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))}
    label_files = {os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.txt')}

    missing_labels = image_files - label_files
    missing_images = label_files - image_files

    if missing_labels:
        logging.warning(f"Images without labels: {missing_labels}")
    if missing_images:
        logging.warning(f"Labels without images: {missing_images}")

    return not (missing_labels or missing_images)


if __name__ == "__main__":
    logging.info("Starting pipeline...")

    # Khởi tạo pipeline
    pipeline = ComponentDetectionPipeline(
        yolo_model_path="yolov8s.pt",
        classification_model_path="classification_model.pth",
        num_classes=5
    )

    # Xác minh dữ liệu
    if not validate_dataset("./dataset/images/train", "./dataset/labels/train"):
        logging.error("Dataset validation failed.")
        exit(1)

    # Huấn luyện mô hình phân loại
    pipeline.train_classification_model(
        train_dir="./dataset/images/train",
        val_dir="./dataset/images/val",
        output_path="classification_model.pth",
        epochs=10,
        batch_size=16,
        lr=0.001
    )

    # Huấn luyện YOLO
    pipeline.train_yolo_model(data_yaml="./dataset/data.yaml", epochs=2)

    logging.info("Pipeline completed successfully.")
