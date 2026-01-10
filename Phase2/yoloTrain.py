import os
import json
import cv2
import torch
import numpy as np
from glob import glob
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import resnet50
import torch.nn.functional as F

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
# Dataset
# ----------------------------
TARGET_SIZE = (1024, 1024)
TARGET_LABELS = ["Crack", "Rust"]
NUM_TARGET_CLASSES = len(TARGET_LABELS)

class CrackRustDataset(Dataset):
    def __init__(self, img_dir, json_dir, augment=False):
        self.img_paths = sorted(glob(os.path.join(img_dir, '*.jpg')))
        self.json_dir = json_dir
        self.augment = augment

    def __len__(self):
        return len(self.img_paths)

    def rasterize_polygon_to_mask(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)

        H, W = data['imageHeight'], data['imageWidth']
        mask = np.zeros((NUM_TARGET_CLASSES, H, W), dtype=np.uint8)
        label_to_channel = {label: i for i, label in enumerate(TARGET_LABELS)}

        try:
            for shape in data['shapes']:
                label = shape['label']
                if label in label_to_channel:
                    ch = label_to_channel[label]
                    points = np.array(shape['points'], np.int32)
                    temp_mask = np.zeros((H, W), np.uint8)
                    cv2.fillPoly(temp_mask, [points.reshape((-1,1,2))], 1)
                    mask[ch] = np.maximum(mask[ch], temp_mask)
        except Exception as e:
            print(f"CRITICAL RASTERIZATION ERROR in {json_path}: {e}")
            mask = np.zeros((NUM_TARGET_CLASSES, *TARGET_SIZE), dtype=np.uint8)

        # Resize to target
        resized_mask = np.zeros((NUM_TARGET_CLASSES, *TARGET_SIZE), dtype=np.uint8)
        for i in range(NUM_TARGET_CLASSES):
            resized_mask[i] = cv2.resize(mask[i], TARGET_SIZE[::-1], interpolation=cv2.INTER_AREA)
        return resized_mask.astype(np.float32)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        base_name = os.path.basename(img_path).split('.')[0]
        json_path = os.path.join(self.json_dir, base_name + '.json')

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, TARGET_SIZE)
        img = img.astype(np.float32) / 255.0  # Normalize 0-1

        mask = self.rasterize_polygon_to_mask(json_path)

        if self.augment:
            # Simple augment: horizontal flip
            if np.random.rand() > 0.5:
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()
            # Random rotation
            angle = np.random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((TARGET_SIZE[1]//2, TARGET_SIZE[0]//2), angle, 1.0)
            img = cv2.warpAffine(img, M, TARGET_SIZE)
            for i in range(NUM_TARGET_CLASSES):
                mask[i] = cv2.warpAffine(mask[i], M, TARGET_SIZE)

        img = torch.tensor(img.transpose(2,0,1), dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        return img, mask

# ----------------------------
# Model: UNet with ResNet50 encoder
# ----------------------------
class UNetResNet50(nn.Module):
    def __init__(self, num_classes=NUM_TARGET_CLASSES, pretrained=True):
        super().__init__()
        resnet = resnet50(weights='DEFAULT' if pretrained else None)

        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.encoder1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.center = nn.Sequential(
            nn.Conv2d(2048, 1024, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up4 = self.up_block(1024, 512)
        self.up3 = self.up_block(512, 256)
        self.up2 = self.up_block(256, 128)
        self.up1 = self.up_block(128, 64)
        self.up0 = self.up_block(64, 64)
        self.final = nn.Conv2d(64, num_classes, 1)

    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e0 = self.encoder0(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        center = self.center(e4)

        d4 = self.up4(center) + e4
        d3 = self.up3(d4) + e3
        d2 = self.up2(d3) + e2
        d1 = self.up1(d2) + e1
        d0 = self.up0(d1) + e0

        out = self.final(d0)
        out = torch.sigmoid(out)
        return out

# ----------------------------
# Losses
# ----------------------------
def dice_loss(y_pred, y_true, smooth=1e-6):
    y_pred = (y_pred > 0.5).float()
    intersection = (y_pred * y_true).sum(dim=(1,2,3))
    union = y_pred.sum(dim=(1,2,3)) + y_true.sum(dim=(1,2,3))
    return 1 - (intersection + smooth) / (union + smooth)

bce_loss = nn.BCELoss()

def focal_loss(y_pred, y_true, alpha=0.8, gamma=2.0):
    y_pred = torch.clamp(y_pred, 1e-6, 1-1e-6)
    pt = y_pred * y_true + (1 - y_pred)*(1 - y_true)
    loss = -alpha * (1 - pt)**gamma * torch.log(pt)
    return loss.mean()

def weighted_loss(y_pred, y_true):
    # per-channel weights
    crack_weight = 8.0
    rust_weight = 1.0

    yt_c = y_true[:,0:1]
    yp_c = y_pred[:,0:1]
    yt_r = y_true[:,1:2]
    yp_r = y_pred[:,1:2]

    crack_loss = 0.4*dice_loss(yp_c, yt_c) + 0.4*focal_loss(yp_c, yt_c) + 0.2*bce_loss(yp_c, yt_c)
    rust_loss = 0.5*dice_loss(yp_r, yt_r) + 0.5*bce_loss(yp_r, yt_r)

    return crack_weight*crack_loss + rust_weight*rust_loss

# ----------------------------
# DataLoaders
# ----------------------------
train_dataset = CrackRustDataset("../Data/images/train", "../Data/annotations/train", augment=True)
val_dataset   = CrackRustDataset("../Data/images/validation", "../Data/annotations/validation", augment=False)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

# ----------------------------
# Training Loop
# ----------------------------
model = UNetResNet50().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
EPOCHS = 100

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = weighted_loss(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = weighted_loss(outputs, masks)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

torch.save(model.state_dict(), "unet_crack_rust.pth")
print("Training finished and weights saved.")
