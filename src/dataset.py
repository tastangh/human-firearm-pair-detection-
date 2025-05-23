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
    def __init__(self, csv_file, project_root_path, transform_pbb=None, roi_size=config.ROI_SIZE):
        """
        Args:
            csv_file (string): Annotasyonları içeren parse edilmiş CSV dosyasının yolu.
            project_root_path (string): Projenin kök dizininin mutlak yolu.
            transform_pbb (callable, optional): Kırpılmış ve yeniden boyutlandırılmış PBB görüntüsüne uygulanacak transform.
            roi_size (tuple): PBB'nin ve maskelerin yeniden boyutlandırılacağı hedef boyut.
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

        self.transform_pbb = transform_pbb
        self.roi_size = roi_size

        if self.transform_pbb is None:
            # Eğer özel bir transform verilmediyse, varsayılanı kullan
            self.transform_pbb = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # Maskeler için tensora dönüştürme (normalize edilmeyecekler)
        self.mask_to_tensor = transforms.ToTensor()


    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx >= len(self.annotations_df):
            # Bu durum normalde DataLoader tarafından ele alınır, ama bir güvenlik önlemi
            raise IndexError(f"Index {idx} out of bounds for dataset with size {len(self.annotations_df)}")

        row = self.annotations_df.iloc[idx]
        
        relative_img_path = row['image_path']
        img_abs_path = os.path.abspath(os.path.join(self.project_root, relative_img_path))

        try:
            image = Image.open(img_abs_path).convert('RGB')
        except FileNotFoundError:
            print(f"HATA: Görüntü bulunamadı (Dataset): {img_abs_path}")
            return self._get_dummy_data()
        except Exception as e:
            print(f"HATA: Görüntü yüklenirken {img_abs_path} - Hata: {e}")
            return self._get_dummy_data()

        h_bbox_orig = [row['human_x1'], row['human_y1'], row['human_x2'], row['human_y2']]
        f_bbox_orig = [row['weapon_x1'], row['weapon_y1'], row['weapon_x2'], row['weapon_y2']]
        label = int(row['label'])
        img_w, img_h = image.size

        # PBB'yi (Paired Bounding Box) hesapla
        pbb_x1 = min(h_bbox_orig[0], f_bbox_orig[0])
        pbb_y1 = min(h_bbox_orig[1], f_bbox_orig[1])
        pbb_x2 = max(h_bbox_orig[2], f_bbox_orig[2])
        pbb_y2 = max(h_bbox_orig[3], f_bbox_orig[3])
        
        # Görüntü sınırları içinde kalmasını sağla
        pbb_x1 = max(0, int(pbb_x1))
        pbb_y1 = max(0, int(pbb_y1))
        pbb_x2 = min(img_w, int(pbb_x2))
        pbb_y2 = min(img_h, int(pbb_y2))

        if pbb_x2 <= pbb_x1 or pbb_y2 <= pbb_y1: # Geçersiz PBB
            # print(f"UYARI: Geçersiz PBB {img_abs_path} - H:{h_bbox_orig}, F:{f_bbox_orig} -> PBB:[{pbb_x1},{pbb_y1},{pbb_x2},{pbb_y2}]")
            return self._get_dummy_data(label_val=label) # Etiketi koruyabiliriz ya da -1 yapabiliriz

        try:
            pbb_image_pil = image.crop((pbb_x1, pbb_y1, pbb_x2, pbb_y2))
        except Exception as e:
            print(f"HATA: PBB Kırpma sırasında {img_abs_path} - PBB:[{pbb_x1},{pbb_y1},{pbb_x2},{pbb_y2}] - Hata: {e}")
            return self._get_dummy_data(label_val=label)
            
        # PBB içindeki insan ve silah bbox'larını PBB'nin kendi koordinat sistemine göre ayarla
        h_bbox_in_pbb = [
            int(h_bbox_orig[0] - pbb_x1), int(h_bbox_orig[1] - pbb_y1),
            int(h_bbox_orig[2] - pbb_x1), int(h_bbox_orig[3] - pbb_y1)
        ]
        f_bbox_in_pbb = [
            int(f_bbox_orig[0] - pbb_x1), int(f_bbox_orig[1] - pbb_y1),
            int(f_bbox_orig[2] - pbb_x1), int(f_bbox_orig[3] - pbb_y1)
        ]

        pbb_w_orig, pbb_h_orig = pbb_image_pil.size

        human_mask_pil = Image.new('L', (pbb_w_orig, pbb_h_orig), 0) # 'L' tek kanal 8-bit (0-255)
        firearm_mask_pil = Image.new('L', (pbb_w_orig, pbb_h_orig), 0)
        
        h_bbox_in_pbb_clipped = [max(0,h_bbox_in_pbb[0]), max(0,h_bbox_in_pbb[1]), min(pbb_w_orig,h_bbox_in_pbb[2]), min(pbb_h_orig,h_bbox_in_pbb[3])]
        f_bbox_in_pbb_clipped = [max(0,f_bbox_in_pbb[0]), max(0,f_bbox_in_pbb[1]), min(pbb_w_orig,f_bbox_in_pbb[2]), min(pbb_h_orig,f_bbox_in_pbb[3])]

        if h_bbox_in_pbb_clipped[2] > h_bbox_in_pbb_clipped[0] and h_bbox_in_pbb_clipped[3] > h_bbox_in_pbb_clipped[1]:
             ImageDraw.Draw(human_mask_pil).rectangle(h_bbox_in_pbb_clipped, fill=255) # Binary mask için 255 (ToTensor 1'e çevirecek)
        if f_bbox_in_pbb_clipped[2] > f_bbox_in_pbb_clipped[0] and f_bbox_in_pbb_clipped[3] > f_bbox_in_pbb_clipped[1]:
             ImageDraw.Draw(firearm_mask_pil).rectangle(f_bbox_in_pbb_clipped, fill=255)

        # PBB görüntüsünü ve maskeleri hedef `roi_size`'a yeniden boyutlandır
        pbb_image_resized_pil = pbb_image_pil.resize(self.roi_size, Image.BILINEAR)
        human_mask_resized_pil = human_mask_pil.resize(self.roi_size, Image.NEAREST) # Binary için NEAREST
        firearm_mask_resized_pil = firearm_mask_pil.resize(self.roi_size, Image.NEAREST)

        # PBB görüntüsünü tensora çevir ve normalize et
        pbb_image_tensor = self.transform_pbb(pbb_image_resized_pil)
        
        # Maskeleri tensora çevir (0-1 aralığında olacaklar, [1, H, W] şeklinde)
        human_mask_tensor = self.mask_to_tensor(human_mask_resized_pil) # [1, H, W]
        firearm_mask_tensor = self.mask_to_tensor(firearm_mask_resized_pil) # [1, H, W]
            
        return pbb_image_tensor, human_mask_tensor, firearm_mask_tensor, torch.tensor(label, dtype=torch.long)

    def _get_dummy_data(self, label_val=-1):
        """Hata durumlarında boş veri döndürür."""
        dummy_pbb = torch.randn(3, self.roi_size[0], self.roi_size[1])
        # Maskeler tek kanallı ve unsqueeze edilmiş olmalı (modelin beklediği gibi)
        dummy_mask = torch.zeros(1, self.roi_size[0], self.roi_size[1], dtype=torch.float32)
        return dummy_pbb, dummy_mask, dummy_mask, torch.tensor(label_val, dtype=torch.long)


if __name__ == '__main__':
    if not os.path.exists(config.TRAIN_CSV):
        print(f"Lütfen önce dataAnalysis/data.py scriptini çalıştırarak {config.TRAIN_CSV} dosyasını oluşturun.")
    else:
        # Özel transformları eğitim sırasında tanımlayacağız, burada varsayılanı test edelim
        dataset = LFCPairedAttentionDataset(csv_file=config.TRAIN_CSV,
                                            project_root_path=config.PROJECT_ROOT,
                                            roi_size=config.ROI_SIZE)
        if len(dataset) > 0:
            print(f"Dataset boyutu: {len(dataset)}")
            pbb_tensor, h_mask_tensor, f_mask_tensor, lbl = dataset[0]
            print("PBB tensor şekli:", pbb_tensor.shape) # Beklenen: [3, H, W]
            print("İnsan Maskesi tensor şekli:", h_mask_tensor.shape) # Beklenen: [1, H, W]
            print("Silah Maskesi tensor şekli:", f_mask_tensor.shape) # Beklenen: [1, H, W]
            print("Etiket:", lbl)
            print("İnsan Maskesi min/max:", h_mask_tensor.min(), h_mask_tensor.max())
            print("Silah Maskesi min/max:", f_mask_tensor.min(), f_mask_tensor.max())


            # DataLoader testi
            from torch.utils.data import DataLoader
            dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0) # num_workers=0 debug için
            try:
                pbb_batch, h_mask_batch, f_mask_batch, lbl_batch = next(iter(dataloader))
                print("\nBatch PBB tensor şekli:", pbb_batch.shape)
                print("Batch İnsan Maskesi tensor şekli:", h_mask_batch.shape)
                print("Batch Silah Maskesi tensor şekli:", f_mask_batch.shape)
                print("Batch Etiketler:", lbl_batch)
            except Exception as e:
                print(f"DataLoader testi sırasında hata: {e}")
                # Eğer dummy data içinde -1 etiket döndürülüyorsa ve loss bunu kaldıramıyorsa,
                # DataLoader'da sorunlu veriyi atlayan bir collate_fn gerekebilir veya dataset filtrelenmeli.
        else:
             print("Dataset boş veya yüklenemedi.")