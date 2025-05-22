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
from dataset import LFCHumanFirearmROIDataset
from model import DualStreamCNN # Değişti
from utils import plot_loss_accuracy, save_checkpoint # Değişti

def main():
    print(f"Cihaz kullanılıyor: {config.DEVICE}")

    # Veri transformları
    # Eğitim seti için daha fazla augmentation
    train_transforms = transforms.Compose([
        transforms.Resize(config.ROI_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(config.ROI_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if not os.path.exists(config.TRAIN_CSV):
        print(f"HATA: {config.TRAIN_CSV} bulunamadı. Lütfen önce dataAnalysis/data.py scriptini çalıştırın.")
        return
        
    # Tüm eğitim verisini yükle (transformları daha sonra subsetlere özel atayacağız)
    full_dataset = LFCHumanFirearmROIDataset(csv_file=config.TRAIN_CSV, project_root_path=config.PROJECT_ROOT) # NEW

    if len(full_dataset) == 0:
        print("Eğitim veri seti yüklenemedi veya boş.")
        return

    # Eğitim ve doğrulama setlerine ayır
    val_split = 0.2
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    np.random.seed(42) # Tekrarlanabilirlik için
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Her subset için ayrı transformlarla Dataset objeleri (aslında aynı CSV'yi kullanacaklar)
    # Bu, transformları Subset'e direk uygulayamadığımız için bir yöntem.
    # Daha verimli bir yol, Dataset sınıfının kendisinin train/val modunu desteklemesi olabilir.
    train_dataset_instance = LFCHumanFirearmROIDataset(csv_file=config.TRAIN_CSV, project_root_path=config.PROJECT_ROOT, transform_human=train_transforms, transform_weapon=train_transforms) # NEW
    val_dataset_instance = LFCHumanFirearmROIDataset(csv_file=config.TRAIN_CSV, project_root_path=config.PROJECT_ROOT, transform_human=val_transforms, transform_weapon=val_transforms) # NEW

    train_subset = Subset(train_dataset_instance, train_indices)
    val_subset = Subset(val_dataset_instance, val_indices)
    
    train_dataloader = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_dataloader = DataLoader(val_subset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)

    print(f"Eğitim örnek sayısı: {len(train_subset)}")
    print(f"Doğrulama örnek sayısı: {len(val_subset)}")

    model = DualStreamCNN(num_classes=1, backbone_name='resnet34').to(config.DEVICE) # num_classes=1 for BCEWithLogitsLoss
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.3)
    best_val_accuracy = 0.0
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        progress_bar_train = tqdm(train_dataloader, desc=f"Epoch {epoch+1} Train", unit="batch")
        for human_rois, weapon_rois, labels in progress_bar_train:
            human_rois = human_rois.to(config.DEVICE)
            weapon_rois = weapon_rois.to(config.DEVICE)
            labels = labels.to(config.DEVICE).float().unsqueeze(1)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(human_rois, weapon_rois)
                loss = criterion(outputs, labels)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * human_rois.size(0)
            running_corrects += torch.sum(preds == labels.data)
            progress_bar_train.set_postfix(loss=loss.item(), acc=torch.sum(preds == labels.data).item()/human_rois.size(0))

        epoch_loss = running_loss / len(train_subset)
        epoch_acc = running_corrects.double() / len(train_subset)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.item()) # .item() to get Python number
        print(f"Eğitim Kayıp: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        
        progress_bar_val = tqdm(val_dataloader, desc=f"Epoch {epoch+1} Val  ", unit="batch")
        with torch.no_grad():
            for human_rois, weapon_rois, labels in progress_bar_val:
                human_rois = human_rois.to(config.DEVICE)
                weapon_rois = weapon_rois.to(config.DEVICE)
                labels = labels.to(config.DEVICE).float().unsqueeze(1)

                outputs = model(human_rois, weapon_rois)
                loss = criterion(outputs, labels)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                
                val_running_loss += loss.item() * human_rois.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
                progress_bar_val.set_postfix(loss=loss.item(), acc=torch.sum(preds == labels.data).item()/human_rois.size(0))

        val_epoch_loss = val_running_loss / len(val_subset)
        val_epoch_acc = val_running_corrects.double() / len(val_subset)
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc.item())
        print(f"Doğrulama Kayıp: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")
        
        scheduler.step(val_epoch_loss)

        if val_epoch_acc > best_val_accuracy:
            best_val_accuracy = val_epoch_acc
            model_filename = f"dualstream_best_epoch_{epoch+1}_acc_{val_epoch_acc:.4f}.pth"
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_accuracy': best_val_accuracy,
            }, filename=model_filename, save_dir=config.MODEL_SAVE_PATH)

    print(f"\nEğitim tamamlandı. En iyi doğrulama doğruluğu: {best_val_accuracy:.4f}")
    curves_save_path = os.path.join(config.RESULTS_DIR, "training_curves.png")
    plot_loss_accuracy(train_losses, val_losses, train_accuracies, val_accuracies, save_path=curves_save_path)
    
if __name__ == '__main__':
    main()