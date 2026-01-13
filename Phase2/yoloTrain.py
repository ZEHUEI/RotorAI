# PyTorch version of UNet corrosion training
import os
import cv2
import json
import torch
import random
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import resnet50
from pycocotools.coco import COCO
from tqdm import tqdm

# ---------------------------
# GPU setup
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------
# Hyperparameters
# ---------------------------
TARGET_SIZE = (512, 512)
BATCH_SIZE = 4
NUM_CLASSES = 1
EPOCHS = 100
LEARNING_RATE = 1e-4

# ---------------------------
# Rasterize COCO annotations
# ---------------------------
def rasterize_coco_annotations(anns, target_size, target_labels=["corrosion"]):
    H, W = target_size
    mask = np.zeros((H, W, len(target_labels)), dtype=np.uint8)
    label_to_channel = {label: i for i, label in enumerate(target_labels)}

    for ann in anns:
        category_name = ann['category_name']
        if category_name in label_to_channel:
            channel_idx = label_to_channel[category_name]
            for seg in ann['segmentation']:
                pts = np.array(seg).reshape(-1, 2).astype(np.int32)
                temp_mask = np.zeros((H, W), dtype=np.uint8)
                cv2.fillPoly(temp_mask, [pts], 1)
                mask[:, :, channel_idx] = np.maximum(mask[:, :, channel_idx], temp_mask)
    return mask

# ---------------------------
# Dataset
# ---------------------------
class CocoSegmentationDataset(Dataset):
    def __init__(self, img_dir, coco_json, target_size=(512,512), augment=False):
        self.coco = COCO(coco_json)
        self.img_dir = img_dir
        self.image_ids = self.coco.getImgIds()
        self.target_size = target_size
        self.augment = augment

        # Build image path -> annotations dict
        self.img_paths = []
        self.anns_dict = {}
        cat_id_to_name = {cat['id']: cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())}

        for img_id in self.image_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            img_filename = img_info['file_name']
            img_path = os.path.join(img_dir, img_filename)
            if os.path.exists(img_path):
                self.img_paths.append(img_path)
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                anns = self.coco.loadAnns(ann_ids)
                for ann in anns:
                    ann['category_name'] = cat_id_to_name[ann['category_id']]
                self.anns_dict[img_path] = anns
            else:
                print(f"Warning: {img_path} not found")

        print(f"Collected {len(self.img_paths)} images from {img_dir}")

        # Augmentations
        self.geometric_aug = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomRotation(degrees=3),
            T.RandomResizedCrop(target_size, scale=(0.9,1.0))
        ])
        self.photometric_aug = T.Compose([
            T.ColorJitter(brightness=0.1, contrast=0.2)
        ])
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        anns = self.anns_dict[img_path]

        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_size)

        # Load mask
        mask = rasterize_coco_annotations(anns, self.target_size)
        mask = mask[:, :, 0]  # single channel

        # Convert to PIL for torchvision transforms
        import PIL.Image as Image
        img = Image.fromarray(img)
        mask = Image.fromarray(mask*255)

        if self.augment:
            seed = np.random.randint(2147483647)
            random.seed(seed)
            torch.manual_seed(seed)
            img = self.geometric_aug(img)
            random.seed(seed)
            torch.manual_seed(seed)
            mask = self.geometric_aug(mask)

            img = self.photometric_aug(img)

        img = self.to_tensor(img)
        mask = self.to_tensor(mask)
        mask = (mask > 0.5).float()  # ensure binary mask

        return img, mask

# ---------------------------
# UNet Model (ResNet50 encoder)
# ---------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        base_model = resnet50(weights="IMAGENET1K_V1")
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu), #64
            nn.Sequential(base_model.maxpool, base_model.layer1), #256
            base_model.layer2, #512
            base_model.layer3, #1024
            base_model.layer4  #2048
        ])
        self.center = ConvBlock(2048, 512)
        self.dec4 = ConvBlock(512+1024, 512)
        self.dec3 = ConvBlock(512+512, 256)
        self.dec2 = ConvBlock(256+256, 128)
        self.dec1 = ConvBlock(128+64, 64)
        self.final = nn.Conv2d(64, num_classes, 1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        e0 = self.encoder_layers[0](x)
        e1 = self.encoder_layers[1](e0)
        e2 = self.encoder_layers[2](e1)
        e3 = self.encoder_layers[3](e2)
        e4 = self.encoder_layers[4](e3)

        c = self.center(e4)

        d4 = self.upsample(c)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.dec4(d4)

        d3 = self.upsample(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upsample(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upsample(d2)
        d1 = torch.cat([d1, e0], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        out = torch.sigmoid(out)
        return out

# ---------------------------
# Losses
# ---------------------------
class DiceBCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        smooth = 1e-6
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1 - (2*intersection + smooth)/(pred_flat.sum() + target_flat.sum() + smooth)
        bce_loss = self.bce(pred, target)
        return dice_loss + bce_loss

# ---------------------------
# Data loaders
# ---------------------------
train_dataset = CocoSegmentationDataset("../Data/train", "../Data/train/_annotations.coco.json", augment=True)
val_dataset = CocoSegmentationDataset("../Data/valid", "../Data/valid/_annotations.coco.json", augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ---------------------------
# Training loop
# ---------------------------
model = UNet(num_classes=NUM_CLASSES).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = DiceBCE()

best_val_iou = 0

def mean_iou(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for imgs, masks in tqdm(train_loader):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*imgs.size(0)
    train_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0
    val_iou = 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)
            val_loss += loss.item()*imgs.size(0)
            val_iou += mean_iou(preds, masks).item()*imgs.size(0)
    val_loss /= len(val_loader.dataset)
    val_iou /= len(val_loader.dataset)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IOU: {val_iou:.4f}")

    # Save best model
    if val_iou > best_val_iou:
        best_val_iou = val_iou
        torch.save(model.state_dict(), "best_unet_corrosion.pth")
        print("Saved best model!")

print("Training finished.")
