# src/dataset.py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import numpy as np
# Ensure config is imported correctly if it's not already setup for your project structure
# from . import config # If dataset.py and config.py are in the same directory (src)
# Or, if config.py is one level up, this might not be needed if Python path is set up.
# For your structure, assuming train.py imports config and dataset:
import config # This should work if train.py is in src and adds src to path or runs from project root

class LFCHumanFirearmROIDataset(Dataset):
    def __init__(self, csv_file, project_root_path, transform_human=None, transform_weapon=None, roi_size=config.ROI_SIZE): # Added project_root_path
        """
        Args:
            csv_file (string): Annotasyonları içeren parse edilmiş CSV dosyasının yolu.
            project_root_path (string): Projenin kök dizininin mutlak yolu.
            transform_human (callable, optional): İnsan ROI'sine uygulanacak transform.
            transform_weapon (callable, optional): Silah ROI'sine uygulanacak transform.
            roi_size (tuple): ROI'lerin yeniden boyutlandırılacağı hedef boyut.
        """
        self.project_root = project_root_path
        try:
            self.annotations_df = pd.read_csv(csv_file)
        except FileNotFoundError:
            print(f"HATA: Annotasyon dosyası bulunamadı: {csv_file}")
            self.annotations_df = pd.DataFrame()
            return
        except pd.errors.EmptyDataError:
            print(f"HATA: Annotasyon dosyası boş: {csv_file}")
            self.annotations_df = pd.DataFrame()
            return


        self.transform_human = transform_human
        self.transform_weapon = transform_weapon
        self.roi_size = roi_size

    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx >= len(self.annotations_df):
            raise IndexError("Index out of bounds in dataset")

        row = self.annotations_df.iloc[idx]
        
        # Get relative path from CSV
        relative_img_path = row['image_path'] 
        
        # Construct absolute path
        # os.path.join will correctly handle if relative_img_path starts with '/' (making it absolute already)
        # os.path.abspath will resolve any '..' components and ensure it's a canonical absolute path.
        img_abs_path = os.path.abspath(os.path.join(self.project_root, relative_img_path))

        try:
            image = Image.open(img_abs_path).convert('RGB')
        except FileNotFoundError:
            print(f"HATA: Görüntü bulunamadı (Dataset). Aranan yol: {img_abs_path}")
            print(f"   (CSV'deki göreli yol: {relative_img_path}, Proje Kökü: {self.project_root})")
            dummy_tensor = torch.randn(3, self.roi_size[0], self.roi_size[1])
            return dummy_tensor, dummy_tensor, torch.tensor(-1, dtype=torch.long) # Return a dummy
        except Exception as e:
            print(f"HATA: Görüntü yüklenirken {img_abs_path} - Hata: {e}")
            dummy_tensor = torch.randn(3, self.roi_size[0], self.roi_size[1])
            return dummy_tensor, dummy_tensor, torch.tensor(-1, dtype=torch.long)


        h_bbox = [row['human_x1'], row['human_y1'], row['human_x2'], row['human_y2']]
        f_bbox = [row['weapon_x1'], row['weapon_y1'], row['weapon_x2'], row['weapon_y2']]
        label = int(row['label'])

        img_w, img_h = image.size
        h_bbox = [max(0, h_bbox[0]), max(0, h_bbox[1]), min(img_w, h_bbox[2]), min(img_h, h_bbox[3])]
        f_bbox = [max(0, f_bbox[0]), max(0, f_bbox[1]), min(img_w, f_bbox[2]), min(img_h, f_bbox[3])]

        try:
            if h_bbox[2] <= h_bbox[0] or h_bbox[3] <= h_bbox[1]:
                human_roi = Image.new('RGB', self.roi_size, (128,128,128)) # Placeholder
            else:
                human_roi = image.crop(h_bbox)

            if f_bbox[2] <= f_bbox[0] or f_bbox[3] <= f_bbox[1]:
                firearm_roi = Image.new('RGB', self.roi_size, (128,128,128)) # Placeholder
            else:
                firearm_roi = image.crop(f_bbox)
        except Exception as e:
            print(f"HATA: ROI Kırpma sırasında {img_abs_path} - H: {h_bbox}, F: {f_bbox} - Hata: {e}")
            dummy_tensor = torch.randn(3, self.roi_size[0], self.roi_size[1])
            return dummy_tensor, dummy_tensor, torch.tensor(-1, dtype=torch.long)


        if self.transform_human:
            human_roi_tensor = self.transform_human(human_roi)
        else: 
            preprocess = transforms.Compose([transforms.Resize(self.roi_size), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            human_roi_tensor = preprocess(human_roi)

        if self.transform_weapon:
            firearm_roi_tensor = self.transform_weapon(firearm_roi)
        else: 
            preprocess = transforms.Compose([transforms.Resize(self.roi_size), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            firearm_roi_tensor = preprocess(firearm_roi)
            
        return human_roi_tensor, firearm_roi_tensor, torch.tensor(label, dtype=torch.long)


if __name__ == '__main__':
    if not os.path.exists(config.TRAIN_CSV):
        print(f"Lütfen önce dataAnalysis/data.py scriptini çalıştırarak {config.TRAIN_CSV} dosyasını oluşturun.")
    else:
        dataset = LFCHumanFirearmROIDataset(csv_file=config.TRAIN_CSV, image_base_dir_unused="")
        if len(dataset) > 0:
            print(f"Dataset boyutu: {len(dataset)}")
            h_roi, f_roi, lbl = dataset[0]
            print("İnsan ROI tensor şekli:", h_roi.shape)
            print("Silah ROI tensor şekli:", f_roi.shape)
            print("Etiket:", lbl)

            dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
            h_batch, f_batch, lbl_batch = next(iter(dataloader))
            print("Batch İnsan ROI tensor şekli:", h_batch.shape)
            print("Batch Silah ROI tensor şekli:", f_batch.shape)
            print("Batch Etiketler:", lbl_batch)
        else:
             print("Dataset boş.")