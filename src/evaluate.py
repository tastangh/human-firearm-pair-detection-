# src/evaluate.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import os
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tqdm import tqdm
import numpy as np

import config
from dataset import LFCPairedAttentionDataset # Güncellendi
from model import SaliencySingleStreamCNN # Güncellendi
from utils import plot_confusion_matrix, load_checkpoint

def evaluate_model(model_path):
    print(f"Cihaz kullanılıyor: {config.DEVICE}")
    if not os.path.exists(config.TEST_CSV):
        print(f"HATA: {config.TEST_CSV} bulunamadı. Lütfen önce dataAnalysis/data.py scriptini çalıştırın.")
        return

    # Test veri transformları (eğitimdeki validation ile aynı olmalı)
    test_pbb_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = LFCPairedAttentionDataset(csv_file=config.TEST_CSV,
                                            project_root_path=config.PROJECT_ROOT,
                                            transform_pbb=test_pbb_transforms,
                                            roi_size=config.ROI_SIZE)
    
    if len(test_dataset) == 0:
        print("Test veri seti yüklenemedi veya boş.")
        return

    test_dataloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
    print(f"Test için {len(test_dataset)} örnek, {len(test_dataloader)} batch.")

    # Modeli yükle
    model = SaliencySingleStreamCNN(
        num_classes_classifier=config.NUM_CLASSES_CLASSIFIER,
        backbone_name='resnet18', # Kaydedilen modelle aynı olmalı
        pretrained=False, # Ağırlıkları checkpoint'ten yükleyeceğiz
        num_attention_channels=config.NUM_ATTENTION_CHANNELS,
        num_reconstruction_channels=config.NUM_RECONSTRUCTION_CHANNELS,
        roi_size=config.ROI_SIZE
    ).to(config.DEVICE)
    
    model, _, _ = load_checkpoint(model_path, model, optimizer=None, device=config.DEVICE)
    # Not: load_checkpoint dosya bulunamazsa uyarı verir ve orijinal modeli döndürür.
    # Eğer checkpoint'teki model yapısı mevcut model yapısıyla eşleşmiyorsa hata alırsınız.

    model.eval()
    all_labels = []
    all_preds_class = []

    progress_bar_test = tqdm(test_dataloader, desc="Test Ediliyor", unit="batch")

    with torch.no_grad():
        for pbb_images, human_masks_gt, firearm_masks_gt, labels in progress_bar_test:
            pbb_images = pbb_images.to(config.DEVICE)
            human_masks_input = human_masks_gt.to(config.DEVICE).float()
            firearm_masks_input = firearm_masks_gt.to(config.DEVICE).float()
            
            # Model artık iki çıktı veriyor, sadece sınıflandırma çıktısını kullanacağız
            classification_outputs, _ = model(pbb_images, human_masks_input, firearm_masks_input)
            
            preds_probs = torch.sigmoid(classification_outputs).cpu()
            predicted_classes = (preds_probs > 0.5).float().squeeze().numpy() # squeeze tek elemanlı batch'ler için

            all_labels.extend(labels.numpy())
            if predicted_classes.ndim == 0: # Tek bir örnekse
                 all_preds_class.append(predicted_classes.item())
            else: # Batch ise
                 all_preds_class.extend(predicted_classes.tolist())

    if not all_labels or not all_preds_class:
        print("Değerlendirme için etiket veya sınıflandırma tahmini üretilemedi.")
        return

    accuracy = accuracy_score(all_labels, all_preds_class)
    report = classification_report(all_labels, all_preds_class, target_names=config.CLASSES, digits=4, zero_division=0)
    
    print(f"\nTest Sınıflandırma Doğruluğu: {accuracy:.4f}")
    print("\nSınıflandırma Raporu:")
    print(report)

    # Sonuçları kaydet
    base_model_name = os.path.splitext(os.path.basename(model_path))[0]
    metrics_file_path = os.path.join(config.RESULTS_DIR, f"test_metrics_{base_model_name}.txt")
    confusion_matrix_save_path = os.path.join(config.RESULTS_DIR, f"confusion_matrix_{base_model_name}.png")
    
    with open(metrics_file_path, "w") as f:
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Test Classification Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Test metrikleri {metrics_file_path} dosyasına kaydedildi.")

    plot_confusion_matrix(all_labels, all_preds_class, class_names=config.CLASSES, save_path=confusion_matrix_save_path)
    print(f"Karışıklık matrisi {confusion_matrix_save_path} dosyasına kaydedildi.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Modeli Değerlendirme Script'i")
    parser.add_argument("model_path", type=str, help="Değerlendirilecek eğitilmiş modelin yolu (.pth dosyası)")
    args = parser.parse_args()
    
    if not os.path.isfile(args.model_path):
        print(f"HATA: Belirtilen model yolu bir dosya değil veya bulunamadı: {args.model_path}")
    else:
        evaluate_model(args.model_path)