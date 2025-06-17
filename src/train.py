# src/train.py
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
import pandas as pd
import os
import numpy as np
from tqdm import tqdm

import config 
from dataset import LFCPairedAttentionDataset
from model import SaliencySingleStreamCNN
from utils import plot_loss_accuracy, save_checkpoint

def main():
    print(f"Cihaz kullanılıyor: {config.DEVICE}")
    print(f"Kullanılan renk uzayı: {config.COLOR_SPACE}")
    print(f"Kullanılan backbone (train.py içinde belirtilmeli): resnet18") # Manuel not

    normalize_mean = [0.485, 0.456, 0.406]
    normalize_std = [0.229, 0.224, 0.225]
    if config.COLOR_SPACE == "YCbCr":
        print("UYARI: YCbCr için normalizasyon ve ColorJitter ayarları gözden geçirilmeli.")
        # normalize_mean = [ YCbCr_Y_mean, YCbCr_Cb_mean, YCbCr_Cr_mean ] # Uygun değerler bulunmalı
        # normalize_std  = [ YCbCr_Y_std,  YCbCr_Cb_std,  YCbCr_Cr_std  ]

    train_pbb_img_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05) if config.COLOR_SPACE == "RGB" else nn.Identity(),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])
    val_pbb_img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])

    if not os.path.exists(config.TRAIN_CSV):
        print(f"HATA: {config.TRAIN_CSV} bulunamadı.")
        return
        
    full_dataset_train_transforms = LFCPairedAttentionDataset(
        csv_file=config.TRAIN_CSV, 
        project_root_path=config.PROJECT_ROOT, 
        transform_pbb_img=train_pbb_img_transforms,
        roi_size=config.ROI_SIZE,
        color_space=config.COLOR_SPACE
    )
    full_dataset_val_transforms = LFCPairedAttentionDataset(
        csv_file=config.TRAIN_CSV, 
        project_root_path=config.PROJECT_ROOT, 
        transform_pbb_img=val_pbb_img_transforms,
        roi_size=config.ROI_SIZE,
        color_space=config.COLOR_SPACE
    )

    if len(full_dataset_train_transforms) == 0:
        print("Eğitim veri seti yüklenemedi veya boş.")
        return

    val_split = 0.2
    dataset_size = len(full_dataset_train_transforms)
    indices = list(range(dataset_size))
    split_idx = int(np.floor(val_split * dataset_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split_idx:], indices[:split_idx]

    train_subset = Subset(full_dataset_train_transforms, train_indices)
    val_subset = Subset(full_dataset_val_transforms, val_indices)
    
    train_dataloader = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(val_subset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=False)

    print(f"Eğitim örnek sayısı: {len(train_subset)}")
    print(f"Doğrulama örnek sayısı: {len(val_subset)}")

    model = SaliencySingleStreamCNN(
        num_classes_classifier=config.NUM_CLASSES_CLASSIFIER,
        backbone_name='resnet18', # Burası önemli, config'e de yansıtılabilir
        num_attention_channels=config.NUM_ATTENTION_CHANNELS,
        num_reconstruction_channels=config.NUM_RECONSTRUCTION_CHANNELS,
        roi_size=config.ROI_SIZE,
        color_space_aware=(config.COLOR_SPACE=="YCbCr")
    ).to(config.DEVICE)
    
    criterion_classification = nn.BCEWithLogitsLoss()
    criterion_reconstruction = nn.MSELoss()

    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.3) # verbose kaldırıldı
    
    best_val_loss = float('inf')
    train_total_losses, val_total_losses = [], []
    train_class_accuracies, val_class_accuracies = [], []

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        
        model.train()
        running_total_loss = 0.0; running_class_loss = 0.0; running_recon_loss = 0.0
        running_class_corrects = 0
        
        progress_bar_train = tqdm(train_dataloader, desc=f"Epoch {epoch+1} Train", unit="batch")
        for pbb_images, human_masks_gt, firearm_masks_gt, labels in progress_bar_train:
            pbb_images = pbb_images.to(config.DEVICE)
            human_masks_input = human_masks_gt.to(config.DEVICE).float()
            firearm_masks_input = firearm_masks_gt.to(config.DEVICE).float()
            labels = labels.to(config.DEVICE).float().unsqueeze(1)

            optimizer.zero_grad()
            
            classification_outputs, reconstructed_masks = model(pbb_images, human_masks_input, firearm_masks_input)
            
            loss_c = criterion_classification(classification_outputs, labels)
            gt_recon_targets = torch.cat((human_masks_input, firearm_masks_input), dim=1)
            loss_p = criterion_reconstruction(reconstructed_masks, gt_recon_targets)
            total_loss = loss_c + config.LAMBDA_RECONSTRUCTION * loss_p
            
            total_loss.backward()
            optimizer.step()

            running_total_loss += total_loss.item() * pbb_images.size(0)
            running_class_loss += loss_c.item() * pbb_images.size(0)
            running_recon_loss += loss_p.item() * pbb_images.size(0)
            preds_class = (torch.sigmoid(classification_outputs) > 0.5).float()
            running_class_corrects += torch.sum(preds_class == labels.data)
            
            progress_bar_train.set_postfix(
                TotalL=f"{total_loss.item():.3f}",ClassL=f"{loss_c.item():.3f}",
                ReconL=f"{loss_p.item():.3f}",Acc=f"{torch.sum(preds_class == labels.data).item()/pbb_images.size(0):.3f}"
            )

        epoch_total_loss = running_total_loss / len(train_subset)
        epoch_class_loss = running_class_loss / len(train_subset)
        epoch_recon_loss = running_recon_loss / len(train_subset)
        epoch_class_acc = running_class_corrects.double() / len(train_subset)
        train_total_losses.append(epoch_total_loss); train_class_accuracies.append(epoch_class_acc.item())
        print(f"Eğitim - Total Loss: {epoch_total_loss:.4f}, Class Loss: {epoch_class_loss:.4f}, Recon Loss: {epoch_recon_loss:.4f}, Class Acc: {epoch_class_acc:.4f}")

        model.eval()
        val_running_total_loss = 0.0; val_running_class_corrects = 0
        progress_bar_val = tqdm(val_dataloader, desc=f"Epoch {epoch+1} Val  ", unit="batch")
        with torch.no_grad():
            for pbb_images, human_masks_gt, firearm_masks_gt, labels in progress_bar_val:
                pbb_images = pbb_images.to(config.DEVICE)
                human_masks_input = human_masks_gt.to(config.DEVICE).float()
                firearm_masks_input = firearm_masks_gt.to(config.DEVICE).float()
                labels = labels.to(config.DEVICE).float().unsqueeze(1)

                classification_outputs, reconstructed_masks = model(pbb_images, human_masks_input, firearm_masks_input)
                loss_c = criterion_classification(classification_outputs, labels)
                gt_recon_targets = torch.cat((human_masks_input, firearm_masks_input), dim=1)
                loss_p = criterion_reconstruction(reconstructed_masks, gt_recon_targets)
                total_loss = loss_c + config.LAMBDA_RECONSTRUCTION * loss_p
                
                val_running_total_loss += total_loss.item() * pbb_images.size(0)
                preds_class = (torch.sigmoid(classification_outputs) > 0.5).float()
                val_running_class_corrects += torch.sum(preds_class == labels.data)
                progress_bar_val.set_postfix(TotalL=f"{total_loss.item():.3f}", Acc=f"{torch.sum(preds_class == labels.data).item()/pbb_images.size(0):.3f}")

        val_epoch_total_loss = val_running_total_loss / len(val_subset)
        val_epoch_class_acc = val_running_class_corrects.double() / len(val_subset)
        val_total_losses.append(val_epoch_total_loss); val_class_accuracies.append(val_epoch_class_acc.item())
        print(f"Doğrulama - Total Loss: {val_epoch_total_loss:.4f}, Class Acc: {val_epoch_class_acc:.4f}")
        
        scheduler.step(val_epoch_total_loss)

        if val_epoch_total_loss < best_val_loss:
            best_val_loss = val_epoch_total_loss
            model_filename = f"saliency_model_{config.COLOR_SPACE.lower()}_ep{epoch+1}_loss{val_epoch_total_loss:.4f}_acc{val_epoch_class_acc:.4f}.pth"
            save_checkpoint({
                'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss, 'val_accuracy_at_best_loss': val_epoch_class_acc.item(),
                'backbone_name': 'resnet18', 'color_space': config.COLOR_SPACE
            }, filename=model_filename, save_dir=config.MODEL_SAVE_PATH)

    print(f"\nEğitim tamamlandı. En iyi doğrulama kaybı: {best_val_loss:.4f}")
    curves_save_path = os.path.join(config.RESULTS_DIR, f"training_curves_saliency_{config.COLOR_SPACE.lower()}.png")
    plot_loss_accuracy(train_total_losses, val_total_losses, train_class_accuracies, val_class_accuracies, save_path=curves_save_path)
    
if __name__ == '__main__':
    main()