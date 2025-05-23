# src/config.py
import torch
import os

# Projenin kök dizinini bul
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Veri Yolları (Proje köküne göre)
DATA_ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "dataAnalysis")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

TRAIN_CSV = os.path.join(DATA_ANALYSIS_DIR, "train_annotations_parsed.csv")
TEST_CSV = os.path.join(DATA_ANALYSIS_DIR, "test_annotations_parsed.csv")

# Görüntü ve ROI Ayarları
IMAGE_DIR_TRAIN_UNUSED = os.path.join(DATA_DIR, "Training_Dataset/images/") # Artık doğrudan kullanılmıyor
IMAGE_DIR_TEST_UNUSED = os.path.join(DATA_DIR, "Test_Dataset/images/")   # Artık doğrudan kullanılmıyor
ROI_SIZE = (224, 224) # PBB ve maskelerin hedef boyutu

# Eğitim Hiperparametreleri
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16 # GPU belleğine göre ayarlayın
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30 # Ayarlayın
NUM_WORKERS = 2 # Sisteminizin çekirdek sayısına göre ayarlayın
LAMBDA_RECONSTRUCTION = 0.5 # Rekonstrüksiyon kaybının ağırlığı (E-HFPL için)

# Model Kaydetme
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "saved_models")
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

# Sonuç Kaydetme
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Sınıflar
CLASSES = ['not_carrier', 'carrier'] # İkili sınıflandırma için
NUM_CLASSES_CLASSIFIER = 1 # BCEWithLogitsLoss için modelin sınıflandırma başlığı 1 çıktı verir
NUM_ATTENTION_CHANNELS = 2 # İnsan ve Silah için dikkat maskeleri
NUM_RECONSTRUCTION_CHANNELS = 2 # İnsan ve Silah için yeniden oluşturulacak maskeler