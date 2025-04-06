import os
import random
import csv
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm 

# ========== Reproducibility ==========
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ========== Dataset Class ==========
class CancerDataset(Dataset):
    def __init__(self, dataframe, directory, transform=None):
        self.dataframe = dataframe
        self.directory = directory
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_id = self.dataframe.iloc[idx]['id']
        img_path = os.path.join(self.directory, img_id + ".tif")
        image = Image.open(img_path).convert("RGB")
        label = self.dataframe.iloc[idx]['label'] if 'label' in self.dataframe.columns else -1
        if self.transform:
            image = self.transform(image)
        return image, label

# ========== Model ==========
class CNNModel(nn.Module):
    def __init__(self, filters=(32, 64, 128), dropout=0.5):
        super(CNNModel, self).__init__()
        f1, f2, f3 = filters
        self.net = nn.Sequential(
            nn.Conv2d(3, f1, kernel_size=3, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(f1, f2, kernel_size=3, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(f2, f3, kernel_size=3, padding=1),
            nn.BatchNorm2d(f3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(f3 * (64 // 8) * (64 // 8), 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x)

# ========== Main ==========
def main():
    # Paths
    DATA_DIR = "."
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    TEST_DIR = os.path.join(DATA_DIR, "test")
    LABELS_PATH = os.path.join(DATA_DIR, "train_labels.csv")
    RESULTS_DIR = os.path.join(DATA_DIR, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Hyperparameters
    IMG_SIZE = 64
    BATCH_SIZE = 32
    EPOCHS = 2

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    # Load data
    df = pd.read_csv(LABELS_PATH)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

    train_dataset = CancerDataset(train_df, TRAIN_DIR, transform)
    val_dataset = CancerDataset(val_df, TRAIN_DIR, transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Experiment configs
    experiments = [
        {"filters": (32, 64, 128), "dropout": 0.5, "lr": 0.001},
        {"filters": (64, 128, 256), "dropout": 0.3, "lr": 0.0005},
    ]

    results_csv_path = os.path.join(RESULTS_DIR, "experiment_results.csv")
    with open(results_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filters", "dropout", "lr", "accuracy"])

    best_accuracy = 0
    best_model = None

    for i, exp in enumerate(experiments):
        print(f"\nRunning Experiment {i+1}: {exp}")
        model = CNNModel(filters=exp["filters"], dropout=exp["dropout"]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=exp["lr"])
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch [{epoch+1}/{EPOCHS}]")
            
            for images, labels in loop:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                loop.set_postfix(loss=loss.item())
            
            print(f"Epoch {epoch+1} completed. Avg Loss: {running_loss / len(train_loader):.4f}")

        # Evaluate
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                preds = torch.sigmoid(outputs).squeeze() > 0.5
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        print(f"Validation Accuracy: {accuracy:.4f}")

        # Save results
        with open(results_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([exp["filters"], exp["dropout"], exp["lr"], accuracy])

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model.state_dict()
            torch.save(best_model, os.path.join(RESULTS_DIR, "best_model.pth"))
            print("Best model updated and saved.")

    # Load best model and predict on test set
    print("\nGenerating predictions on test set...")
    model = CNNModel().to(device)
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "best_model.pth")))
    model.eval()

    test_ids = [f[:-4] for f in os.listdir(TEST_DIR) if f.endswith(".tif")]
    test_df = pd.DataFrame({"id": test_ids})
    test_dataset = CancerDataset(test_df, TEST_DIR, transform)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4, pin_memory=True)

    predictions = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
            predictions.extend(probs)

    submission_df = pd.DataFrame({"id": test_ids, "label": predictions})
    submission_df.to_csv(os.path.join(RESULTS_DIR, "submission.csv"), index=False)
    print("Saved predictions to results/submission.csv")

    # Visualize
    def plot_samples(dataset, count=6):
        fig, axs = plt.subplots(1, count, figsize=(15, 3))
        for i in range(count):
            img, label = dataset[i]
            axs[i].imshow(img.permute(1, 2, 0))
            axs[i].set_title(f"Label: {label}")
            axs[i].axis("off")
        plt.savefig(os.path.join(RESULTS_DIR, "sample_images.png"))
        plt.show()

    plot_samples(val_dataset)

# ========== Required for Windows multiprocessing ==========
if __name__ == '__main__':
    main()

