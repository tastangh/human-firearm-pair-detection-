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
ROI_SIZE = (224, 224) # PBB ve maskelerin hedef boyutu
COLOR_SPACE = "YCbCr" # "RGB" veya "YCbCr" olabilir

# Eğitim Hiperparametreleri
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
NUM_WORKERS = 2
LAMBDA_RECONSTRUCTION = 0.5

# Nesne Tespiti Ayarları (evaluate_with_detection.py için)
HUMAN_DETECTION_THRESHOLD = 0.7
WEAPON_DETECTION_THRESHOLD = 0.5
INTERACTION_THRESHOLD_FOR_AP = 0.5
IOU_THRESHOLD_FOR_AP_MATCHING = 0.5

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
NUM_CLASSES_CLASSIFIER = 1
NUM_ATTENTION_CHANNELS = 2
NUM_RECONSTRUCTION_CHANNELS = 2

COCO_PERSON_CLASS_ID = 1