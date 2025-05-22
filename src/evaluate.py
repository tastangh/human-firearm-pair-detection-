# src/evaluate.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import os
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tqdm import tqdm
import numpy as np

import config # config.py'yi import ediyoruz
from dataset import LFCHumanFirearmROIDataset
from model import DualStreamCNN
from utils import plot_confusion_matrix, load_checkpoint

def evaluate_model(model_path):
    print(f"Cihaz kullanılıyor: {config.DEVICE}")
    if not os.path.exists(config.TEST_CSV):
        print(f"HATA: {config.TEST_CSV} bulunamadı. Lütfen önce dataAnalysis/data.py scriptini çalıştırın.")
        return

    # Test veri transformları (eğitimdeki validation ile aynı olmalı)
    test_transforms = transforms.Compose([
        transforms.Resize(config.ROI_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = LFCHumanFirearmROIDataset(csv_file=config.TEST_CSV,
                                            project_root_path=config.PROJECT_ROOT, # Proje kök yolunu aktar
                                            transform_human=test_transforms,
                                            transform_weapon=test_transforms,
                                            roi_size=config.ROI_SIZE)
    
    if len(test_dataset) == 0:
        print("Test veri seti yüklenemedi veya boş.")
        return

    test_dataloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
    print(f"Test için {len(test_dataset)} örnek, {len(test_dataloader)} batch.")

    # Modeli yükle
    model = DualStreamCNN(num_classes=1, backbone_name='resnet34', pretrained=False).to(config.DEVICE)
    
    # Checkpoint yükleme
    # load_checkpoint fonksiyonu zaten utils.py içinde tanımlı ve os.path.exists kontrolünü yapıyor.
    model, _, _ = load_checkpoint(model_path, model, optimizer=None, device=config.DEVICE)
    # Eğer load_checkpoint sadece modeli döndürüyorsa ve dosya bulunamazsa bir uyarı basıp çıkabilir veya
    # ilk model örneğini (ağırlıksız) döndürebilir. Hata kontrolü load_checkpoint içinde yapılmalı.
    # Hata durumunda devam etmemek için load_checkpoint'un dönüşünü kontrol edebilirsiniz.
    # Örneğin, load_checkpoint dosya bulunamazsa None döndürürse:
    # if model is None:
    #     print(f"HATA: Model yüklenemedi: {model_path}")
    #     return
    # Ancak mevcut load_checkpoint yapınızda, bulunamazsa orijinal modeli döndürüyor.

    model.eval()
    all_labels = []
    all_preds = []

    progress_bar_test = tqdm(test_dataloader, desc="Test Ediliyor", unit="batch")

    with torch.no_grad():
        for human_rois, weapon_rois, labels in progress_bar_test:
            human_rois = human_rois.to(config.DEVICE)
            weapon_rois = weapon_rois.to(config.DEVICE)
            
            outputs = model(human_rois, weapon_rois)
            preds_probs = torch.sigmoid(outputs).cpu()
            predicted_classes = (preds_probs > 0.5).float().squeeze().numpy()

            all_labels.extend(labels.numpy())
            if predicted_classes.ndim == 0:
                 all_preds.append(predicted_classes.item())
            else:
                 all_preds.extend(predicted_classes.tolist())

    if not all_labels or not all_preds:
        print("Değerlendirme için etiket veya tahmin üretilemedi.")
        return

    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=config.CLASSES, digits=4, zero_division=0) # zero_division eklendi
    
    print(f"\nTest Doğruluğu: {accuracy:.4f}")
    print("\nSınıflandırma Raporu:")
    print(report)

    # Sonuçları kaydetmek için config.RESULTS_DIR kullan
    # config.py zaten RESULTS_DIR yoksa oluşturuyor.
    metrics_file_path = os.path.join(config.RESULTS_DIR, "test_metrics.txt")
    confusion_matrix_save_path = os.path.join(config.RESULTS_DIR, "confusion_matrix.png")
    
    with open(metrics_file_path, "w") as f:
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Test metrikleri {metrics_file_path} dosyasına kaydedildi.")

    plot_confusion_matrix(all_labels, all_preds, class_names=config.CLASSES, save_path=confusion_matrix_save_path)
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