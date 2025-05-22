# src/utils.py
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
import config
import os # <--- EKLENDİ

def plot_loss_accuracy(train_losses, val_losses, train_accuracies, val_accuracies, save_path=None):
    """
    Eğitim ve doğrulama kayıp ve doğruluk grafiklerini çizer.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Eğitim Kaybı')
    plt.plot(epochs, val_losses, 'ro-', label='Doğrulama Kaybı')
    plt.title('Eğitim ve Doğrulama Kaybı')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'bo-', label='Eğitim Doğruluğu')
    plt.plot(epochs, val_accuracies, 'ro-', label='Doğrulama Doğruluğu')
    plt.title('Eğitim ve Doğrulama Doğruluğu')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk')
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Grafik kaydedildi: {save_path}")
    # plt.show() # Eğitimi durdurmaması için bunu yorum satırı yapabilir veya en sona alabilirsiniz.
                 # Genellikle eğitim sonunda bir kez çağrılır.

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Karışıklık matrisini çizer.
    Args:
        y_true (list or np.array): Gerçek etiketler.
        y_pred (list or np.array): Tahmin edilen etiketler.
        class_names (list): Sınıf isimleri.
        save_path (str, optional): Grafiğin kaydedileceği dosya yolu.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Karışıklık Matrisi')
    plt.ylabel('Gerçek Etiket')
    plt.xlabel('Tahmin Edilen Etiket')
    if save_path:
        plt.savefig(save_path)
        print(f"Karışıklık matrisi kaydedildi: {save_path}")
    # plt.show() # Benzer şekilde, eğitim sonunda çağrılabilir.

def save_checkpoint(state, filename="my_checkpoint.pth.tar", save_dir=config.MODEL_SAVE_PATH):
    """
    Model checkpoint'ini kaydeder.
    Args:
        state (dict): Modelin state_dict'ini ve diğer bilgileri içeren dict.
        filename (str): Kaydedilecek dosya adı.
        save_dir (str): Kaydedileceği klasör.
    """
    if not os.path.exists(save_dir): # Artık 'os' tanımlı
        os.makedirs(save_dir)      # Artık 'os' tanımlı
    filepath = os.path.join(save_dir, filename) # Artık 'os' tanımlı
    torch.save(state, filepath)
    print(f"Checkpoint kaydedildi: {filepath}")


def load_checkpoint(checkpoint_path, model, optimizer=None, device=config.DEVICE):
    """
    Model checkpoint'ini yükler.
    Args:
        checkpoint_path (str): Yüklenecek checkpoint dosyasının yolu.
        model (torch.nn.Module): Ağırlıkların yükleneceği model.
        optimizer (torch.optim.Optimizer, optional): Optimizer state'inin yükleneceği optimizer.
        device (str): Modelin yükleneceği cihaz ('cuda' veya 'cpu').
    Returns:
        model, optimizer, start_epoch
    """
    if not os.path.exists(checkpoint_path): # Artık 'os' tanımlı
        print(f"UYARI: Checkpoint dosyası bulunamadı: {checkpoint_path}")
        return model, optimizer, 0

    print(f"=> Checkpoint yükleniyor: '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Eğer checkpoint doğrudan model.state_dict() ise
    if 'state_dict' not in checkpoint and 'optimizer' not in checkpoint:
        model.load_state_dict(checkpoint)
        print("Model state_dict doğrudan yüklendi.")
        return model, optimizer, 0 # Epoch bilgisi yoksa 0 döndür
    
    # Eğer checkpoint beklenen yapıda ise
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("UYARI: Checkpoint'te 'state_dict' bulunamadı.")

    start_epoch = 0
    if optimizer and 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except ValueError as e:
            print(f"UYARI: Optimizer state yüklenirken hata: {e}. Optimizer state yüklenmiyor.")
    elif optimizer:
        print("UYARI: Checkpoint'te 'optimizer' state bulunamadı.")
        
    if 'epoch' in checkpoint:
        start_epoch = checkpoint['epoch']
        
    print(f"Checkpoint yüklendi. Epoch: {start_epoch if 'epoch' in checkpoint else 'N/A'}")
    return model, optimizer, start_epoch