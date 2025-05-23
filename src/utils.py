# src/utils.py
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import config
import os

def plot_loss_accuracy(train_losses, val_losses, train_accuracies, val_accuracies, save_path=None):
    """
    Eğitim ve doğrulama kayıp ve doğruluk grafiklerini çizer.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Eğitim Kaybı (Toplam)')
    plt.plot(epochs, val_losses, 'ro-', label='Doğrulama Kaybı (Toplam)')
    plt.title('Eğitim ve Doğrulama Toplam Kaybı')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'bo-', label='Eğitim Sınıf. Doğruluğu')
    plt.plot(epochs, val_accuracies, 'ro-', label='Doğrulama Sınıf. Doğruluğu')
    plt.title('Eğitim ve Doğrulama Sınıflandırma Doğruluğu')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk')
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Grafik kaydedildi: {save_path}")
    plt.close() # Figürü kapat ki bellekte birikmesin

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Karışıklık matrisini çizer.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names))) # Sınıfların sıralamasını garantile
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Karışıklık Matrisi')
    plt.ylabel('Gerçek Etiket')
    plt.xlabel('Tahmin Edilen Etiket')
    if save_path:
        plt.savefig(save_path)
        print(f"Karışıklık matrisi kaydedildi: {save_path}")
    plt.close() # Figürü kapat

def save_checkpoint(state, filename="my_checkpoint.pth.tar", save_dir=config.MODEL_SAVE_PATH):
    """
    Model checkpoint'ini kaydeder.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    print(f"Checkpoint kaydedildi: {filepath}")


def load_checkpoint(checkpoint_path, model, optimizer=None, device=config.DEVICE):
    """
    Model checkpoint'ini yükler.
    """
    if not os.path.exists(checkpoint_path):
        print(f"UYARI: Checkpoint dosyası bulunamadı: {checkpoint_path}. Model rastgele ağırlıklarla devam edecek.")
        return model, optimizer, 0 # Veya None döndürerek hata yönetimi zorunlu kılınabilir

    print(f"=> Checkpoint yükleniyor: '{checkpoint_path}'")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        print(f"HATA: Checkpoint yüklenirken hata: {e}")
        return model, optimizer, 0


    # Doğrudan model.state_dict() olup olmadığını kontrol et
    if 'state_dict' not in checkpoint and all(k.startswith('feb_') or k.startswith('classifier.') or k.startswith('decoder.') or k.startswith('avgpool.') or k.startswith('modified_conv1.') for k in checkpoint.keys()):
        print("Checkpoint doğrudan model state_dict gibi görünüyor. Yükleniyor...")
        try:
            # strict=False, eğer checkpoint'te fazladan/eksik anahtar varsa esneklik sağlar
            # Ancak bu genellikle istenmeyen bir durumdur, model yapılarının eşleşmesi gerekir.
            model.load_state_dict(checkpoint, strict=True)
        except RuntimeError as e:
            print(f"HATA: Doğrudan state_dict yüklenirken anahtar uyuşmazlığı: {e}")
            print("Model yapısı ve checkpoint'teki anahtarların eşleştiğinden emin olun.")
            return model, optimizer, 0 # Hata durumunda orijinal modeli döndür
        start_epoch = checkpoint.get('epoch', 0) if isinstance(checkpoint, dict) else 0 # Eğer dict değilse epoch yok
        # Optimizer yüklenemez bu durumda
        if optimizer:
            print("UYARI: Checkpoint doğrudan state_dict olduğundan optimizer yüklenemedi.")
        return model, optimizer, start_epoch

    # Beklenen checkpoint yapısı
    if 'state_dict' in checkpoint:
        try:
            model.load_state_dict(checkpoint['state_dict'], strict=True)
        except RuntimeError as e:
            print(f"HATA: state_dict yüklenirken anahtar uyuşmazlığı: {e}")
            print("Model yapısı ve checkpoint'teki anahtarların eşleştiğinden emin olun.")
            return model, optimizer, 0 # Hata durumunda orijinal modeli döndür
    else:
        print("UYARI: Checkpoint'te 'state_dict' bulunamadı.")
        return model, optimizer, 0


    start_epoch = 0
    if optimizer and 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except ValueError as e: # Parametre grupları uyuşmazsa
            print(f"UYARI: Optimizer state yüklenirken hata (ValueError): {e}. Optimizer state yüklenmiyor.")
        except Exception as e: # Diğer olası hatalar
             print(f"UYARI: Optimizer state yüklenirken genel hata: {e}. Optimizer state yüklenmiyor.")

    elif optimizer:
        print("UYARI: Checkpoint'te 'optimizer' state bulunamadı veya optimizer sağlanmadı.")
        
    if 'epoch' in checkpoint:
        start_epoch = checkpoint['epoch']
        
    print(f"Checkpoint yüklendi. Epoch: {start_epoch if 'epoch' in checkpoint else 'N/A'}")
    return model, optimizer, start_epoch