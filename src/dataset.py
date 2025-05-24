# src/dataset.py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageDraw
import pandas as pd
import os
import numpy as np
import config

class LFCPairedAttentionDataset(Dataset):
    def __init__(self, csv_file, project_root_path, transform_pbb_img=None, roi_size=config.ROI_SIZE, color_space=config.COLOR_SPACE):
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

        self.transform_pbb_img = transform_pbb_img
        self.roi_size = roi_size
        self.color_space = color_space

        if self.transform_pbb_img is None:
            normalize_mean = [0.485, 0.456, 0.406]
            normalize_std = [0.229, 0.224, 0.225]
            if self.color_space == "YCbCr":
                pass 
            self.transform_pbb_img = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=normalize_mean, std=normalize_std)
            ])
        
        self.mask_to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.annotations_df)

    def _convert_color_space(self, pil_image):
        if self.color_space == "YCbCr":
            return pil_image.convert("YCbCr")
        return pil_image.convert("RGB")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx >= len(self.annotations_df):
            raise IndexError(f"Index {idx} out of bounds for dataset with size {len(self.annotations_df)}")

        row = self.annotations_df.iloc[idx]
        
        relative_img_path = row['image_path']
        img_abs_path = os.path.abspath(os.path.join(self.project_root, relative_img_path))

        try:
            original_pil_image = Image.open(img_abs_path)
            image = self._convert_color_space(original_pil_image)
        except FileNotFoundError:
            return self._get_dummy_data()
        except Exception as e:
            return self._get_dummy_data()

        h_bbox_orig = [row['human_x1'], row['human_y1'], row['human_x2'], row['human_y2']]
        f_bbox_orig = [row['weapon_x1'], row['weapon_y1'], row['weapon_x2'], row['weapon_y2']]
        label = int(row['label'])
        img_w, img_h = original_pil_image.size

        pbb_x1 = min(h_bbox_orig[0], f_bbox_orig[0])
        pbb_y1 = min(h_bbox_orig[1], f_bbox_orig[1])
        pbb_x2 = max(h_bbox_orig[2], f_bbox_orig[2])
        pbb_y2 = max(h_bbox_orig[3], f_bbox_orig[3])
        pbb_x1 = max(0, int(pbb_x1)); pbb_y1 = max(0, int(pbb_y1))
        pbb_x2 = min(img_w, int(pbb_x2)); pbb_y2 = min(img_h, int(pbb_y2))

        if pbb_x2 <= pbb_x1 or pbb_y2 <= pbb_y1:
            return self._get_dummy_data(label_val=label)

        try:
            pbb_image_pil = image.crop((pbb_x1, pbb_y1, pbb_x2, pbb_y2))
        except Exception as e:
            return self._get_dummy_data(label_val=label)
            
        h_bbox_in_pbb = [int(h_bbox_orig[0]-pbb_x1), int(h_bbox_orig[1]-pbb_y1), int(h_bbox_orig[2]-pbb_x1), int(h_bbox_orig[3]-pbb_y1)]
        f_bbox_in_pbb = [int(f_bbox_orig[0]-pbb_x1), int(f_bbox_orig[1]-pbb_y1), int(f_bbox_orig[2]-pbb_x1), int(f_bbox_orig[3]-pbb_y1)]
        pbb_w_orig, pbb_h_orig = pbb_image_pil.size
        human_mask_pil = Image.new('L', (pbb_w_orig, pbb_h_orig), 0)
        firearm_mask_pil = Image.new('L', (pbb_w_orig, pbb_h_orig), 0)
        h_clip = [max(0,h_bbox_in_pbb[0]), max(0,h_bbox_in_pbb[1]), min(pbb_w_orig,h_bbox_in_pbb[2]), min(pbb_h_orig,h_bbox_in_pbb[3])]
        f_clip = [max(0,f_bbox_in_pbb[0]), max(0,f_bbox_in_pbb[1]), min(pbb_w_orig,f_bbox_in_pbb[2]), min(pbb_h_orig,f_bbox_in_pbb[3])]
        if h_clip[2] > h_clip[0] and h_clip[3] > h_clip[1]: ImageDraw.Draw(human_mask_pil).rectangle(h_clip, fill=255)
        if f_clip[2] > f_clip[0] and f_clip[3] > f_clip[1]: ImageDraw.Draw(firearm_mask_pil).rectangle(f_clip, fill=255)

        pbb_image_resized_pil = pbb_image_pil.resize(self.roi_size, Image.BILINEAR)
        human_mask_resized_pil = human_mask_pil.resize(self.roi_size, Image.NEAREST)
        firearm_mask_resized_pil = firearm_mask_pil.resize(self.roi_size, Image.NEAREST)

        pbb_image_tensor = self.transform_pbb_img(pbb_image_resized_pil)
        human_mask_tensor = self.mask_to_tensor(human_mask_resized_pil)
        firearm_mask_tensor = self.mask_to_tensor(firearm_mask_resized_pil)
            
        return pbb_image_tensor, human_mask_tensor, firearm_mask_tensor, torch.tensor(label, dtype=torch.long)

    def _get_dummy_data(self, label_val=-1):
        dummy_pbb = torch.randn(3, self.roi_size[0], self.roi_size[1])
        dummy_mask = torch.zeros(1, self.roi_size[0], self.roi_size[1], dtype=torch.float32)
        return dummy_pbb, dummy_mask, dummy_mask, torch.tensor(label_val, dtype=torch.long)


if __name__ == '__main__':
    if not os.path.exists(config.TRAIN_CSV):
        print(f"Lütfen önce dataAnalysis/data.py scriptini çalıştırarak {config.TRAIN_CSV} dosyasını oluşturun.")
    else:
        dataset = LFCPairedAttentionDataset(csv_file=config.TRAIN_CSV,
                                            project_root_path=config.PROJECT_ROOT,
                                            roi_size=config.ROI_SIZE,
                                            color_space=config.COLOR_SPACE)
        if len(dataset) > 0:
            print(f"Dataset boyutu: {len(dataset)}")
            pbb_tensor, h_mask_tensor, f_mask_tensor, lbl = dataset[0]
            print("PBB tensor şekli:", pbb_tensor.shape)
            print("İnsan Maskesi tensor şekli:", h_mask_tensor.shape)
            print("Silah Maskesi tensor şekli:", f_mask_tensor.shape)
            print("Etiket:", lbl)
            print("İnsan Maskesi min/max:", h_mask_tensor.min(), h_mask_tensor.max())
            print("Silah Maskesi min/max:", f_mask_tensor.min(), f_mask_tensor.max())

            from torch.utils.data import DataLoader
            dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
            try:
                pbb_batch, h_mask_batch, f_mask_batch, lbl_batch = next(iter(dataloader))
                print("\nBatch PBB tensor şekli:", pbb_batch.shape)
                print("Batch İnsan Maskesi tensor şekli:", h_mask_batch.shape)
                print("Batch Silah Maskesi tensor şekli:", f_mask_batch.shape)
                print("Batch Etiketler:", lbl_batch)
            except Exception as e:
                print(f"DataLoader testi sırasında hata: {e}")
        else:
             print("Dataset boş veya yüklenemedi.")