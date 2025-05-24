# src/evaluate_with_detection.py
import torch
import torchvision
from torchvision.transforms import functional as TF
from torchvision import transforms 
from PIL import Image, ImageDraw
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
# from torch.utils.data import Dataset, DataLoader # Gerekirse

import config
from model import SaliencySingleStreamCNN
from utils import load_checkpoint
from torchmetrics.detection import MeanAveragePrecision
# from torchvision.ops import box_iou # Gerekirse manuel IoU için

def create_pbb_and_masks_from_detections(pil_image_rgb, h_box_coords, w_box_coords, roi_size, color_space_cfg, device):
    img_w, img_h = pil_image_rgb.size

    if color_space_cfg == "YCbCr":
        pil_image_processed = pil_image_rgb.convert("YCbCr")
    else:
        pil_image_processed = pil_image_rgb # Zaten RGB ise tekrar convert etmeye gerek yok

    pbb_x1 = min(h_box_coords[0], w_box_coords[0])
    pbb_y1 = min(h_box_coords[1], w_box_coords[1])
    pbb_x2 = max(h_box_coords[2], w_box_coords[2])
    pbb_y2 = max(h_box_coords[3], w_box_coords[3])

    pbb_x1 = max(0, int(pbb_x1)); pbb_y1 = max(0, int(pbb_y1))
    pbb_x2 = min(img_w, int(pbb_x2)); pbb_y2 = min(img_h, int(pbb_y2))

    if pbb_x2 <= pbb_x1 or pbb_y2 <= pbb_y1: return None, None, None

    pbb_pil = pil_image_processed.crop((pbb_x1, pbb_y1, pbb_x2, pbb_y2))
    pbb_w_orig, pbb_h_orig = pbb_pil.size

    h_box_in_pbb = [int(h_box_coords[0]-pbb_x1), int(h_box_coords[1]-pbb_y1), int(h_box_coords[2]-pbb_x1), int(h_box_coords[3]-pbb_y1)]
    w_box_in_pbb = [int(w_box_coords[0]-pbb_x1), int(w_box_coords[1]-pbb_y1), int(w_box_coords[2]-pbb_x1), int(w_box_coords[3]-pbb_y1)]

    human_mask_pil = Image.new('L', (pbb_w_orig, pbb_h_orig), 0)
    firearm_mask_pil = Image.new('L', (pbb_w_orig, pbb_h_orig), 0)
    h_clip = [max(0,h_box_in_pbb[0]),max(0,h_box_in_pbb[1]),min(pbb_w_orig,h_box_in_pbb[2]),min(pbb_h_orig,h_box_in_pbb[3])]
    f_clip = [max(0,w_box_in_pbb[0]),max(0,w_box_in_pbb[1]),min(pbb_w_orig,w_box_in_pbb[2]),min(pbb_h_orig,w_box_in_pbb[3])]
    if h_clip[2]>h_clip[0] and h_clip[3]>h_clip[1]: ImageDraw.Draw(human_mask_pil).rectangle(h_clip, fill=255)
    if f_clip[2]>f_clip[0] and f_clip[3]>f_clip[1]: ImageDraw.Draw(firearm_mask_pil).rectangle(f_clip, fill=255)

    pbb_resized_pil = pbb_pil.resize(roi_size, Image.BILINEAR)
    human_mask_resized_pil = human_mask_pil.resize(roi_size, Image.NEAREST)
    firearm_mask_resized_pil = firearm_mask_pil.resize(roi_size, Image.NEAREST)

    norm_mean = [0.485,0.456,0.406]; norm_std=[0.229,0.224,0.225]
    if color_space_cfg == "YCbCr": pass # Özel YCbCr normalizasyonu
    
    # 'transforms' artık global olarak tanımlı olmalı
    transform_img = transforms.Compose([ 
        transforms.ToTensor(), 
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])
    
    pbb_tensor = transform_img(pbb_resized_pil).to(device)
    human_mask_tensor = TF.to_tensor(human_mask_resized_pil).to(device)
    firearm_mask_tensor = TF.to_tensor(firearm_mask_resized_pil).to(device)
    return pbb_tensor, human_mask_tensor, firearm_mask_tensor

def apply_max_out_detections(image_level_predictions):
    if not image_level_predictions: return []
    # image_level_predictions: [{"h_box":tensor, "w_box":tensor, "h_score":float, "w_score":float, "interaction_score":float, "pbb_for_map":tensor}, ...]
    
    weapon_to_interactions = {}
    for pred_dict in image_level_predictions:
        w_box_tuple = tuple(pred_dict["w_box"].int().tolist())
        if w_box_tuple not in weapon_to_interactions:
            weapon_to_interactions[w_box_tuple] = []
        weapon_to_interactions[w_box_tuple].append(pred_dict)

    final_preds_after_max_out = []
    for w_box_tuple, interactions_for_weapon in weapon_to_interactions.items():
        if not interactions_for_weapon: continue
        best_interaction_for_weapon = max(interactions_for_weapon, key=lambda x: x["interaction_score"])
        final_preds_after_max_out.append(best_interaction_for_weapon)
    return final_preds_after_max_out

def evaluate_full_system(classifier_model_path, use_gt_weapons_for_debug=True, apply_max_out=True):
    device = config.DEVICE
    print(f"Cihaz kullanılıyor: {device}")
    
    try:
        checkpoint = torch.load(classifier_model_path, map_location=torch.device('cpu'))
        ckpt_backbone_name = checkpoint.get('backbone_name', 'resnet18')
        ckpt_color_space = checkpoint.get('color_space', 'RGB')
        loaded_epoch = checkpoint.get('epoch', 'N/A')
        print(f"Checkpoint: Backbone='{ckpt_backbone_name}', Renk='{ckpt_color_space}', Epoch='{loaded_epoch}'")
    except Exception as e:
        print(f"UYARI: Checkpoint bilgisi okunamadı ({e}). Varsayılanlar kullanılacak.")
        ckpt_backbone_name = 'resnet18'; ckpt_color_space = config.COLOR_SPACE

    print(f"Kullanılan renk uzayı (çıkarım): {ckpt_color_space}")
    print(f"Max-Out Detections uygulanacak mı: {apply_max_out}")

    print("Nesne tespit modelleri yükleniyor...");
    human_detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    ).eval().to(device)

    # !!! SİLAH TESPİT MODELİ YÜKLEME YERİ (Şimdilik placeholer veya GT) !!!
    weapon_detector = None # Eğer gerçek bir modeliniz yoksa
    if not use_gt_weapons_for_debug and weapon_detector is None:
        print("UYARI: Gerçek silah tespit modeli yüklenmedi ve GT kullanılmıyor. Silah tespiti yapılmayacak.")

    print("Etkileşim sınıflandırıcı yükleniyor...");
    classifier_model = SaliencySingleStreamCNN(
        backbone_name=ckpt_backbone_name, pretrained=False,
        color_space_aware=(ckpt_color_space=="YCbCr")
    ).to(device)
    classifier_model, _, _ = load_checkpoint(classifier_model_path, classifier_model, device=device)
    classifier_model.eval()

    test_df = pd.read_csv(config.TEST_CSV)
    image_paths_relative = test_df['image_path'].unique()
    
    map_metric = MeanAveragePrecision(box_format='xyxy', class_metrics=True, iou_thresholds=[config.IOU_THRESHOLD_FOR_AP_MATCHING])

    for img_idx, relative_img_path in enumerate(tqdm(image_paths_relative, desc="Görüntüler İşleniyor")):
        img_abs_path = os.path.abspath(os.path.join(config.PROJECT_ROOT, relative_img_path))
        try:
            pil_image_rgb = Image.open(img_abs_path).convert("RGB")
            img_tensor_for_detector = TF.to_tensor(pil_image_rgb).to(device)
        except Exception: continue

        gt_annotations_for_image = test_df[test_df['image_path'] == relative_img_path]
        target_pbb_for_map_list = []
        for _, row in gt_annotations_for_image.iterrows():
            if int(row['label']) == 1: # Sadece "carrier" (hold) ground truth'ları
                h_gt_b = [row['human_x1'],row['human_y1'],row['human_x2'],row['human_y2']]
                w_gt_b = [row['weapon_x1'],row['weapon_y1'],row['weapon_x2'],row['weapon_y2']]
                pbb_x1=min(h_gt_b[0],w_gt_b[0]); pbb_y1=min(h_gt_b[1],w_gt_b[1])
                pbb_x2=max(h_gt_b[2],w_gt_b[2]); pbb_y2=max(h_gt_b[3],w_gt_b[3])
                target_pbb_for_map_list.append(torch.tensor([pbb_x1,pbb_y1,pbb_x2,pbb_y2]))
        
        current_img_targets = dict(
            boxes=torch.stack(target_pbb_for_map_list).float().to(device) if target_pbb_for_map_list else torch.empty((0,4), device=device),
            labels=torch.zeros(len(target_pbb_for_map_list), dtype=torch.long, device=device) # "hold" sınıfı için etiket 0
        )

        with torch.no_grad():
            human_dets = human_detector([img_tensor_for_detector])[0]
            h_boxes = human_dets['boxes'][(human_dets['labels']==config.COCO_PERSON_CLASS_ID) & (human_dets['scores']>config.HUMAN_DETECTION_THRESHOLD)]
            h_scores = human_dets['scores'][(human_dets['labels']==config.COCO_PERSON_CLASS_ID) & (human_dets['scores']>config.HUMAN_DETECTION_THRESHOLD)]

            if use_gt_weapons_for_debug:
                w_boxes_list = []; w_scores_list = []
                for _, row in gt_annotations_for_image.iterrows(): # Tüm silahları alalım
                    w_boxes_list.append(torch.tensor([row['weapon_x1'],row['weapon_y1'],row['weapon_x2'],row['weapon_y2']]))
                    w_scores_list.append(torch.tensor(0.99)) # Yüksek dummy skor
                w_boxes = torch.stack(w_boxes_list).float().to(device) if w_boxes_list else torch.empty((0,4),device=device)
                w_scores = torch.stack(w_scores_list).float().to(device) if w_scores_list else torch.empty(0,device=device)
            elif weapon_detector is not None:
                # weapon_dets = weapon_detector([img_tensor_for_detector])[0]
                # w_boxes = weapon_dets['boxes'][weapon_dets['scores'] > config.WEAPON_DETECTION_THRESHOLD]
                # w_scores = weapon_dets['scores'][weapon_dets['scores'] > config.WEAPON_DETECTION_THRESHOLD]
                pass # Placeholder
            else:
                w_boxes = torch.empty((0,4),device=device); w_scores = torch.empty(0,device=device)

        image_level_predictions_for_maxout = []
        if len(h_boxes) > 0 and len(w_boxes) > 0:
            for i in range(len(h_boxes)):
                for j in range(len(w_boxes)):
                    h_b, h_s = h_boxes[i], h_scores[i]
                    w_b, w_s = w_boxes[j], w_scores[j]
                    pbb_t, h_m_t, f_m_t = create_pbb_and_masks_from_detections(pil_image_rgb, h_b, w_b, config.ROI_SIZE, ckpt_color_space, device)
                    if pbb_t is None: continue
                    
                    with torch.no_grad():
                        class_out, _ = classifier_model(pbb_t.unsqueeze(0), h_m_t.unsqueeze(0), f_m_t.unsqueeze(0))
                    inter_prob = torch.sigmoid(class_out).squeeze().item()
                    
                    pbb_map_x1=torch.min(h_b[0],w_b[0]); pbb_map_y1=torch.min(h_b[1],w_b[1])
                    pbb_map_x2=torch.max(h_b[2],w_b[2]); pbb_map_y2=torch.max(h_b[3],w_b[3])
                    pbb_for_map_tensor = torch.stack([pbb_map_x1,pbb_map_y1,pbb_map_x2,pbb_map_y2])

                    image_level_predictions_for_maxout.append({
                        "h_box": h_b, "w_box": w_b, "h_score":h_s.item(), "w_score":w_s.item(),
                        "interaction_score":inter_prob, "pbb_for_map": pbb_for_map_tensor
                    })
        
        final_predictions_this_image = image_level_predictions_for_maxout
        if apply_max_out:
            final_predictions_this_image = apply_max_out_detections(image_level_predictions_for_maxout)

        pred_boxes_for_map = torch.stack([p["pbb_for_map"] for p in final_predictions_this_image]).float().to(device) if final_predictions_this_image else torch.empty((0,4),device=device)
        # AP için skor: h_score * w_score * interaction_score (makaledeki gibi) veya sadece interaction_score
        pred_scores_for_map = torch.tensor([p["h_score"]*p["w_score"]*p["interaction_score"] for p in final_predictions_this_image]).float().to(device) if final_predictions_this_image else torch.empty(0,device=device)
        pred_labels_for_map = torch.zeros(len(final_predictions_this_image), dtype=torch.long, device=device) # Hepsi "hold" adayı

        current_img_preds = dict(boxes=pred_boxes_for_map, scores=pred_scores_for_map, labels=pred_labels_for_map)
        map_metric.update([current_img_preds], [current_img_targets])

    print("\nOrtalama Hassasiyet (AP) hesaplanıyor...")
    try:
        map_results = map_metric.compute()
        print(f"Torchmetrics mAP sonuçları: {map_results}")
        ap_hold_value = map_results['map_50'].item() # IoU=0.50 için
        print(f"  AP_hold (mAP@0.50 IOU_threshold={config.IOU_THRESHOLD_FOR_AP_MATCHING}): {ap_hold_value:.4f}")
        
        results_str = f"Classifier Model: {classifier_model_path}\nLoaded Epoch: {loaded_epoch}\nBackbone: {ckpt_backbone_name}\nColor Space: {ckpt_color_space}\n"
        results_str += f"Human Det Thr: {config.HUMAN_DETECTION_THRESHOLD}\nWeapon Det Thr: {config.WEAPON_DETECTION_THRESHOLD} (GT Weapons Used: {use_gt_weapons_for_debug})\n"
        results_str += f"Max-Out Applied: {apply_max_out}\n"
        results_str += f"AP_hold (mAP@.50 with IOU_thr={config.IOU_THRESHOLD_FOR_AP_MATCHING}): {ap_hold_value:.4f}\n"
        results_str += f"Full torchmetrics results: {map_results}\n"

        fname = f"ap_metrics_fullsys_{os.path.splitext(os.path.basename(classifier_model_path))[0]}.txt"
        with open(os.path.join(config.RESULTS_DIR, fname), "w") as f: f.write(results_str)
        print(f"AP metrikleri {fname} dosyasına kaydedildi.")

    except Exception as e:
        print(f"AP hesaplanırken hata oluştu: {e}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Uçtan Uca Etkileşim Tespit Sistemini Değerlendirme")
    parser.add_argument("classifier_model_path", type=str, help="Eğitilmiş sınıflandırıcı modeli (.pth)")
    parser.add_argument("--no_gt_weapons", action="store_false", dest="use_gt_weapons", help="GT silah kutuları yerine tespit kullan (gerçek silah tespit modeli GEREKLİ)")
    parser.add_argument("--no_max_out", action="store_false", dest="apply_max_out", help="Max-Out Detections uygulama")
    parser.set_defaults(use_gt_weapons=True, apply_max_out=True)
    args = parser.parse_args()
    
    if not os.path.isfile(args.classifier_model_path):
        print(f"HATA: Sınıflandırıcı model yolu bulunamadı: {args.classifier_model_path}")
    else:
        evaluate_full_system(args.classifier_model_path, 
                             use_gt_weapons_for_debug=args.use_gt_weapons,
                             apply_max_out=args.apply_max_out)