# src/config.py
import torch
import os

# Projenin kök dizinini bul
# Bu config.py dosyası src/ içinde olduğu için, ../ proje kökünü verir
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Veri Yolları (Proje köküne göre)
DATA_ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "dataAnalysis")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

TRAIN_CSV = os.path.join(DATA_ANALYSIS_DIR, "train_annotations_parsed.csv")
TEST_CSV = os.path.join(DATA_ANALYSIS_DIR, "test_annotations_parsed.csv")

# Görüntü ve ROI Ayarları
IMAGE_DIR_TRAIN = os.path.join(DATA_DIR, "Training_Dataset/images/")
IMAGE_DIR_TEST = os.path.join(DATA_DIR, "Test_Dataset/images/")
ROI_SIZE = (224, 224)

# Eğitim Hiperparametreleri
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 25
NUM_WORKERS = 2

# Model Kaydetme
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "saved_models")
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

# Sonuç Kaydetme
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Sınıflar
CLASSES = ['not_carrier', 'carrier']
NUM_CLASSES = len(CLASSES) # İkili sınıflandırma için bu 2 olacak, ama BCEWithLogitsLoss ile model çıktısı 1 olacak.
                           # Eğer CrossEntropyLoss ve model çıktısı 2 ise NUM_CLASSES = 2 kullanılır.
                           # Mevcut modelimiz (DualStreamCNN) num_classes=1 bekliyor.