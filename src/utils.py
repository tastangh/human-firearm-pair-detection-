# src/utils.py
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import config
import os

def plot_loss_accuracy(train_losses, val_losses, train_accuracies, val_accuracies, save_path=None):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Eğitim Kaybı (Toplam)')
    plt.plot(epochs, val_losses, 'ro-', label='Doğrulama Kaybı (Toplam)')
    plt.title('Eğitim ve Doğrulama Toplam Kaybı')
    plt.xlabel('Epoch'); plt.ylabel('Kayıp'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'bo-', label='Eğitim Sınıf. Doğruluğu')
    plt.plot(epochs, val_accuracies, 'ro-', label='Doğrulama Sınıf. Doğruluğu')
    plt.title('Eğitim ve Doğrulama Sınıflandırma Doğruluğu')
    plt.xlabel('Epoch'); plt.ylabel('Doğruluk'); plt.legend()
    plt.tight_layout()
    if save_path: plt.savefig(save_path); print(f"Grafik kaydedildi: {save_path}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Karışıklık Matrisi'); plt.ylabel('Gerçek Etiket'); plt.xlabel('Tahmin Edilen Etiket')
    if save_path: plt.savefig(save_path); print(f"Karışıklık matrisi kaydedildi: {save_path}")
    plt.close()

def save_checkpoint(state, filename="my_checkpoint.pth.tar", save_dir=config.MODEL_SAVE_PATH):
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    print(f"Checkpoint kaydedildi: {filepath}")

def load_checkpoint(checkpoint_path, model, optimizer=None, device=config.DEVICE):
    if not os.path.exists(checkpoint_path):
        print(f"UYARI: Checkpoint dosyası bulunamadı: {checkpoint_path}.")
        return model, optimizer, 0
    print(f"=> Checkpoint yükleniyor: '{checkpoint_path}'")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        print(f"HATA: Checkpoint yüklenirken hata: {e}"); return model, optimizer, 0

    start_epoch = checkpoint.get('epoch', 0)
    # Model state_dict yükleme
    # Eğer checkpoint doğrudan model.state_dict() ise veya farklı bir yapıdaysa uyum sağlamaya çalış
    if 'state_dict' in checkpoint:
        model_state_dict = checkpoint['state_dict']
    elif all(k.startswith('feb_') or k.startswith('classifier.') or k.startswith('decoder.') or k.startswith('avgpool.') or k.startswith('modified_conv1.') for k in checkpoint.keys()):
         model_state_dict = checkpoint # Doğrudan state_dict gibi görünüyor
    else:
        print("UYARI: Checkpoint'te 'state_dict' veya tanınan bir model yapısı bulunamadı.")
        return model, optimizer, start_epoch # Model yüklenemedi

    try:
        model.load_state_dict(model_state_dict, strict=True)
    except RuntimeError as e:
        print(f"HATA: state_dict yüklenirken anahtar uyuşmazlığı: {e}")
        print("Model yapısı ve checkpoint eşleşmiyor. strict=False denenebilir ama önerilmez.")
        return model, optimizer, start_epoch # Hata durumunda orijinal modeli döndür
        
    if optimizer and 'optimizer' in checkpoint:
        try: optimizer.load_state_dict(checkpoint['optimizer'])
        except Exception as e: print(f"UYARI: Optimizer state yüklenemedi: {e}")
    elif optimizer: print("UYARI: Checkpoint'te optimizer state bulunamadı.")
        
    print(f"Checkpoint yüklendi. Epoch: {start_epoch if start_epoch > 0 else 'N/A'}")
    return model, optimizer, start_epoch