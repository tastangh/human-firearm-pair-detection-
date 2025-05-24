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
from dataset import LFCPairedAttentionDataset
from model import SaliencySingleStreamCNN
from utils import plot_confusion_matrix, load_checkpoint

def evaluate_classification_accuracy(classifier_model_path):
    print(f"Cihaz kullanılıyor: {config.DEVICE}")
    
    # Checkpoint'ten bilgileri al (backbone, color_space)
    # Bu bilgiler checkpoint'e kaydedilmiş olmalı
    try:
        checkpoint = torch.load(classifier_model_path, map_location=torch.device('cpu')) # CPU'ya yükle sadece bilgi almak için
        ckpt_backbone_name = checkpoint.get('backbone_name', 'resnet18') # Varsayılan resnet18
        ckpt_color_space = checkpoint.get('color_space', 'RGB') # Varsayılan RGB
        print(f"Checkpoint bilgileri: Backbone='{ckpt_backbone_name}', Renk Uzayı='{ckpt_color_space}'")
    except Exception as e:
        print(f"UYARI: Checkpoint'ten ekstra bilgi okunamadı ({e}), varsayılanlar kullanılacak.")
        ckpt_backbone_name = 'resnet18' # Elle ayarla
        ckpt_color_space = config.COLOR_SPACE # Config'den al

    print(f"Değerlendirme için kullanılan renk uzayı: {ckpt_color_space}")


    if not os.path.exists(config.TEST_CSV):
        print(f"HATA: {config.TEST_CSV} bulunamadı.")
        return

    normalize_mean = [0.485, 0.456, 0.406]; normalize_std = [0.229, 0.224, 0.225]
    if ckpt_color_space == "YCbCr":
        pass # YCbCr için özel normalizasyon gerekebilir

    test_pbb_img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])

    test_dataset = LFCPairedAttentionDataset(
        csv_file=config.TEST_CSV,
        project_root_path=config.PROJECT_ROOT,
        transform_pbb_img=test_pbb_img_transforms,
        roi_size=config.ROI_SIZE,
        color_space=ckpt_color_space
    )
    
    if len(test_dataset) == 0:
        print("Test veri seti yüklenemedi veya boş.")
        return

    test_dataloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
    print(f"Test için {len(test_dataset)} örnek, {len(test_dataloader)} batch.")

    model = SaliencySingleStreamCNN(
        num_classes_classifier=config.NUM_CLASSES_CLASSIFIER,
        backbone_name=ckpt_backbone_name,
        pretrained=False,
        num_attention_channels=config.NUM_ATTENTION_CHANNELS,
        num_reconstruction_channels=config.NUM_RECONSTRUCTION_CHANNELS,
        roi_size=config.ROI_SIZE,
        color_space_aware=(ckpt_color_space=="YCbCr")
    ).to(config.DEVICE)
    
    model, _, epoch_loaded = load_checkpoint(classifier_model_path, model, optimizer=None, device=config.DEVICE)
    if epoch_loaded == 0 and 'state_dict' not in torch.load(classifier_model_path, map_location='cpu'): # Basit yükleme kontrolü
        print("UYARI: Model checkpoint'i düzgün yüklenememiş olabilir. Sonuçlar güvenilir olmayabilir.")


    model.eval()
    all_labels_gt = []
    all_preds_class = []

    progress_bar_test = tqdm(test_dataloader, desc="Test Ediliyor (Sınıflandırma)", unit="batch")

    with torch.no_grad():
        for pbb_images, human_masks_gt, firearm_masks_gt, labels_gt in progress_bar_test:
            pbb_images = pbb_images.to(config.DEVICE)
            human_masks_input = human_masks_gt.to(config.DEVICE).float()
            firearm_masks_input = firearm_masks_gt.to(config.DEVICE).float()
            
            classification_outputs, _ = model(pbb_images, human_masks_input, firearm_masks_input)
            preds_probs = torch.sigmoid(classification_outputs).cpu()
            predicted_classes = (preds_probs > 0.5).float().squeeze().numpy()

            all_labels_gt.extend(labels_gt.numpy())
            if predicted_classes.ndim == 0:
                 all_preds_class.append(predicted_classes.item())
            else:
                 all_preds_class.extend(predicted_classes.tolist())

    if not all_labels_gt or not all_preds_class:
        print("Değerlendirme için etiket veya sınıflandırma tahmini üretilemedi.")
        return

    accuracy = accuracy_score(all_labels_gt, all_preds_class)
    report = classification_report(all_labels_gt, all_preds_class, target_names=config.CLASSES, digits=4, zero_division=0)
    
    print(f"\nTest Sınıflandırma Doğruluğu: {accuracy:.4f}")
    print("\nSınıflandırma Raporu:")
    print(report)

    base_model_name = os.path.splitext(os.path.basename(classifier_model_path))[0]
    metrics_file_path = os.path.join(config.RESULTS_DIR, f"classification_metrics_{base_model_name}.txt")
    cm_save_path = os.path.join(config.RESULTS_DIR, f"confusion_matrix_classification_{base_model_name}.png")
    
    with open(metrics_file_path, "w") as f:
        f.write(f"Model Path: {classifier_model_path}\nLoaded Epoch: {epoch_loaded}\nBackbone: {ckpt_backbone_name}\nColor Space: {ckpt_color_space}\n")
        f.write(f"Test Classification Accuracy: {accuracy:.4f}\n\nClassification Report:\n{report}")
    print(f"Sınıflandırma metrikleri {metrics_file_path} dosyasına kaydedildi.")

    plot_confusion_matrix(all_labels_gt, all_preds_class, class_names=config.CLASSES, save_path=cm_save_path)
    print(f"Karışıklık matrisi {cm_save_path} dosyasına kaydedildi.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Sınıflandırma Modelini Değerlendirme Script'i")
    parser.add_argument("classifier_model_path", type=str, help="Değerlendirilecek eğitilmiş sınıflandırıcı modelin yolu (.pth)")
    args = parser.parse_args()
    
    if not os.path.isfile(args.classifier_model_path):
        print(f"HATA: Belirtilen model yolu bulunamadı: {args.classifier_model_path}")
    else:
        evaluate_classification_accuracy(args.classifier_model_path)