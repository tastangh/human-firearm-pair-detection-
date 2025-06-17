# src/evaluate_with_detection.py
import torch
import torchvision
from torchvision.transforms import functional as TF
from torchvision import transforms 
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import random

import config
from model import SaliencySingleStreamCNN
from utils import load_checkpoint
from torchmetrics.detection import MeanAveragePrecision

def draw_detections_on_image(pil_image_rgb_original_copy, human_box_coords, weapon_box_coords, interaction_score, output_path):
    draw = ImageDraw.Draw(pil_image_rgb_original_copy)
    human_color = "green"
    weapon_color = "red"

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    
    h_coords_list = [tuple(human_box_coords[:2].int().tolist()), tuple(human_box_coords[2:].int().tolist())]
    draw.rectangle(h_coords_list, outline=human_color, width=4)
    
    w_coords_list = [tuple(weapon_box_coords[:2].int().tolist()), tuple(weapon_box_coords[2:].int().tolist())]
    try:
        draw.rectangle(w_coords_list, outline=weapon_color, width=3, dash=(6, 4))
    except TypeError:
        # print("UYARI: Pillow versiyonunuz 'dash' parametresini desteklemiyor. Silah düz kırmızı çizgiyle çizilecek.")
        draw.rectangle(w_coords_list, outline=weapon_color, width=3)
    
    text_y_offset_val = 5
    score_text_val = f"Score: {interaction_score:.2f}"
    text_x_val = min(human_box_coords[0].item(), weapon_box_coords[0].item())
    text_y_val = min(human_box_coords[1].item(), weapon_box_coords[1].item()) - 25 - text_y_offset_val
    if text_y_val < 5 : text_y_val = 5
    draw.text((text_x_val, text_y_val), score_text_val, fill="blue", font=font)

    pil_image_rgb_original_copy.save(output_path)

def create_pbb_and_masks_from_detections(pil_image_rgb_input, h_box_coords_tensor_input, w_box_coords_tensor_input, roi_size_input, color_space_cfg_input, device_input):
    img_w_val, img_h_val = pil_image_rgb_input.size

    h_box_coords_list_input = h_box_coords_tensor_input.tolist()
    w_box_coords_list_input = w_box_coords_tensor_input.tolist()

    if color_space_cfg_input == "YCbCr":
        pil_image_processed_input = pil_image_rgb_input.convert("YCbCr")
    else:
        pil_image_processed_input = pil_image_rgb_input 

    pbb_x1_val = min(h_box_coords_list_input[0], w_box_coords_list_input[0])
    pbb_y1_val = min(h_box_coords_list_input[1], w_box_coords_list_input[1])
    pbb_x2_val = max(h_box_coords_list_input[2], w_box_coords_list_input[2])
    pbb_y2_val = max(h_box_coords_list_input[3], w_box_coords_list_input[3])

    pbb_x1_val = max(0, int(pbb_x1_val)); pbb_y1_val = max(0, int(pbb_y1_val))
    pbb_x2_val = min(img_w_val, int(pbb_x2_val)); pbb_y2_val = min(img_h_val, int(pbb_y2_val))

    if pbb_x2_val <= pbb_x1_val or pbb_y2_val <= pbb_y1_val: return None, None, None

    pbb_pil_input = pil_image_processed_input.crop((pbb_x1_val, pbb_y1_val, pbb_x2_val, pbb_y2_val))
    pbb_w_orig_val, pbb_h_orig_val = pbb_pil_input.size

    h_box_in_pbb_list = [int(h_box_coords_list_input[0]-pbb_x1_val), int(h_box_coords_list_input[1]-pbb_y1_val), int(h_box_coords_list_input[2]-pbb_x1_val), int(h_box_coords_list_input[3]-pbb_y1_val)]
    w_box_in_pbb_list = [int(w_box_coords_list_input[0]-pbb_x1_val), int(w_box_coords_list_input[1]-pbb_y1_val), int(w_box_coords_list_input[2]-pbb_x1_val), int(w_box_coords_list_input[3]-pbb_y1_val)]

    human_mask_pil_input = Image.new('L', (pbb_w_orig_val, pbb_h_orig_val), 0)
    firearm_mask_pil_input = Image.new('L', (pbb_w_orig_val, pbb_h_orig_val), 0)
    h_clip_list = [max(0,h_box_in_pbb_list[0]),max(0,h_box_in_pbb_list[1]),min(pbb_w_orig_val,h_box_in_pbb_list[2]),min(pbb_h_orig_val,h_box_in_pbb_list[3])]
    f_clip_list = [max(0,w_box_in_pbb_list[0]),max(0,w_box_in_pbb_list[1]),min(pbb_w_orig_val,w_box_in_pbb_list[2]),min(pbb_h_orig_val,w_box_in_pbb_list[3])]
    if h_clip_list[2]>h_clip_list[0] and h_clip_list[3]>h_clip_list[1]: ImageDraw.Draw(human_mask_pil_input).rectangle(h_clip_list, fill=255)
    if f_clip_list[2]>f_clip_list[0] and f_clip_list[3]>f_clip_list[1]: ImageDraw.Draw(firearm_mask_pil_input).rectangle(f_clip_list, fill=255)

    pbb_resized_pil_input = pbb_pil_input.resize(roi_size_input, Image.BILINEAR)
    human_mask_resized_pil_input = human_mask_pil_input.resize(roi_size_input, Image.NEAREST)
    firearm_mask_resized_pil_input = firearm_mask_pil_input.resize(roi_size_input, Image.NEAREST)

    norm_mean_vals = [0.485,0.456,0.406]; norm_std_vals=[0.229,0.224,0.225]
    if color_space_cfg_input == "YCbCr": pass 
    
    transform_img_fn = transforms.Compose([ 
        transforms.ToTensor(), 
        transforms.Normalize(mean=norm_mean_vals, std=norm_std_vals)
    ])
    
    pbb_tensor_output = transform_img_fn(pbb_resized_pil_input).to(device_input)
    human_mask_tensor_output = TF.to_tensor(human_mask_resized_pil_input).to(device_input)
    firearm_mask_tensor_output = TF.to_tensor(firearm_mask_resized_pil_input).to(device_input)
    return pbb_tensor_output, human_mask_tensor_output, firearm_mask_tensor_output

def apply_max_out_detections(image_level_preds_list):
    if not image_level_preds_list: return []
    
    weapon_to_interactions_dict = {}
    for pred_item_dict in image_level_preds_list:
        w_box_tuple_item = tuple(pred_item_dict["w_box"].int().tolist())
        if w_box_tuple_item not in weapon_to_interactions_dict:
            weapon_to_interactions_dict[w_box_tuple_item] = []
        weapon_to_interactions_dict[w_box_tuple_item].append(pred_item_dict)

    final_preds_after_max_out_result_list = []
    for w_box_tuple_key_item, interactions_for_weapon_item_list in weapon_to_interactions_dict.items():
        if not interactions_for_weapon_item_list: continue
        best_interaction_item = max(interactions_for_weapon_item_list, key=lambda x_item_lambda: x_item_lambda["interaction_score"])
        final_preds_after_max_out_result_list.append(best_interaction_item)
    return final_preds_after_max_out_result_list

def evaluate_full_system(classifier_model_file_path, use_gt_weapons_for_debug_flag=True, apply_max_out_flag=True, num_visual_outputs_per_img=1, visual_score_thr=0.7):
    current_device = config.DEVICE
    print(f"Cihaz kullanılıyor: {current_device}")
    
    try:
        checkpoint_data = torch.load(classifier_model_file_path, map_location=torch.device('cpu'))
        ckpt_backbone_name_val = checkpoint_data.get('backbone_name', 'resnet18')
        ckpt_color_space_val = checkpoint_data.get('color_space', 'RGB')
        loaded_epoch_val = checkpoint_data.get('epoch', 'N/A')
        print(f"Checkpoint: Backbone='{ckpt_backbone_name_val}', Renk='{ckpt_color_space_val}', Epoch='{loaded_epoch_val}'")
    except Exception as e_ckpt:
        print(f"UYARI: Checkpoint bilgisi okunamadı ({e_ckpt}). Varsayılanlar kullanılacak.")
        ckpt_backbone_name_val = 'resnet18'; ckpt_color_space_val = config.COLOR_SPACE

    print(f"Kullanılan renk uzayı (çıkarım): {ckpt_color_space_val}")
    print(f"Max-Out Detections uygulanacak mı: {apply_max_out_flag}")

    print("Nesne tespit modelleri yükleniyor...");
    human_detector_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    ).eval().to(current_device)

    weapon_detector_model = None 
    if not use_gt_weapons_for_debug_flag and weapon_detector_model is None:
        print("UYARI: Gerçek silah tespit modeli yüklenmedi ve GT kullanılmıyor. Silah tespiti yapılmayacak.")

    print("Etkileşim sınıflandırıcı yükleniyor...");
    interaction_classifier_model = SaliencySingleStreamCNN(
        backbone_name=ckpt_backbone_name_val, pretrained=False,
        color_space_aware=(ckpt_color_space_val=="YCbCr")
    ).to(current_device)
    interaction_classifier_model, _, _ = load_checkpoint(classifier_model_file_path, interaction_classifier_model, device=current_device)
    interaction_classifier_model.eval()

    test_dataframe = pd.read_csv(config.TEST_CSV)
    unique_image_paths_relative = test_dataframe['image_path'].unique()
    
    map_metric_calculator = MeanAveragePrecision(box_format='xyxy', class_metrics=True, iou_thresholds=[config.IOU_THRESHOLD_FOR_AP_MATCHING])
    
    visual_output_main_dir = os.path.join(config.RESULTS_DIR, "visual_outputs", os.path.splitext(os.path.basename(classifier_model_file_path))[0])
    os.makedirs(visual_output_main_dir, exist_ok=True)
    
    all_predictions_for_map_metric = []
    all_targets_for_map_metric = []
    
    for loop_img_idx, loop_relative_img_path in enumerate(tqdm(unique_image_paths_relative, desc="Görüntüler İşleniyor")):
        loop_img_abs_path = os.path.abspath(os.path.join(config.PROJECT_ROOT, loop_relative_img_path))
        try:
            loop_pil_image_rgb_original = Image.open(loop_img_abs_path).convert("RGB")
            loop_img_tensor_for_detector = TF.to_tensor(loop_pil_image_rgb_original).to(current_device)
        except Exception: continue
        
        loop_gt_annotations_for_image = test_dataframe[test_dataframe['image_path'] == loop_relative_img_path]
        loop_target_pbb_for_map_list = []
        for _, loop_row_gt in loop_gt_annotations_for_image.iterrows():
            if int(loop_row_gt['label']) == 1:
                h_gt_coords = [loop_row_gt['human_x1'],loop_row_gt['human_y1'],loop_row_gt['human_x2'],loop_row_gt['human_y2']]
                w_gt_coords = [loop_row_gt['weapon_x1'],loop_row_gt['weapon_y1'],loop_row_gt['weapon_x2'],loop_row_gt['weapon_y2']]
                pbb_x1_gt_val=min(h_gt_coords[0],w_gt_coords[0]); pbb_y1_gt_val=min(h_gt_coords[1],w_gt_coords[1])
                pbb_x2_gt_val=max(h_gt_coords[2],w_gt_coords[2]); pbb_y2_gt_val=max(h_gt_coords[3],w_gt_coords[3])
                loop_target_pbb_for_map_list.append(torch.tensor([pbb_x1_gt_val,pbb_y1_gt_val,pbb_x2_gt_val,pbb_y2_gt_val]))
        
        loop_current_img_targets_dict = dict(
            boxes=torch.stack(loop_target_pbb_for_map_list).float().to(current_device) if loop_target_pbb_for_map_list else torch.empty((0,4), device=current_device),
            labels=torch.zeros(len(loop_target_pbb_for_map_list), dtype=torch.long, device=current_device) 
        )
        all_targets_for_map_metric.append(loop_current_img_targets_dict)

        with torch.no_grad():
            human_detections_output = human_detector_model([loop_img_tensor_for_detector])[0]
            h_boxes_detected_tensor = human_detections_output['boxes'][(human_detections_output['labels']==config.COCO_PERSON_CLASS_ID) & (human_detections_output['scores']>config.HUMAN_DETECTION_THRESHOLD)]
            h_scores_detected_tensor = human_detections_output['scores'][(human_detections_output['labels']==config.COCO_PERSON_CLASS_ID) & (human_detections_output['scores']>config.HUMAN_DETECTION_THRESHOLD)]

            if use_gt_weapons_for_debug_flag:
                w_boxes_list_detected_gt = []; w_scores_list_detected_gt = []
                for _, loop_row_gt_w in loop_gt_annotations_for_image.iterrows():
                    w_boxes_list_detected_gt.append(torch.tensor([loop_row_gt_w['weapon_x1'],loop_row_gt_w['weapon_y1'],loop_row_gt_w['weapon_x2'],loop_row_gt_w['weapon_y2']]))
                    w_scores_list_detected_gt.append(torch.tensor(0.99))
                w_boxes_detected_tensor = torch.stack(w_boxes_list_detected_gt).float().to(current_device) if w_boxes_list_detected_gt else torch.empty((0,4),device=current_device)
                w_scores_detected_tensor = torch.stack(w_scores_list_detected_gt).float().to(current_device) if w_scores_list_detected_gt else torch.empty(0,device=current_device)
            elif weapon_detector_model is not None:
                 pass # Placeholder for actual weapon detector
            else:
                w_boxes_detected_tensor = torch.empty((0,4),device=current_device); w_scores_detected_tensor = torch.empty(0,device=current_device)

        loop_image_level_predictions_for_maxout_list = []
        if len(h_boxes_detected_tensor) > 0 and len(w_boxes_detected_tensor) > 0:
            for loop_h_idx in range(len(h_boxes_detected_tensor)):
                for loop_w_idx in range(len(w_boxes_detected_tensor)):
                    h_b_current_det = h_boxes_detected_tensor[loop_h_idx]
                    w_b_current_det = w_boxes_detected_tensor[loop_w_idx]
                    h_s_current_det = h_scores_detected_tensor[loop_h_idx]
                    w_s_current_det = w_scores_detected_tensor[loop_w_idx]
                    
                    pbb_tensor_for_model, h_mask_tensor_for_model, f_mask_tensor_for_model = create_pbb_and_masks_from_detections(
                        loop_pil_image_rgb_original, h_b_current_det, w_b_current_det, config.ROI_SIZE, ckpt_color_space_val, current_device
                    )
                    if pbb_tensor_for_model is None: continue
                    
                    with torch.no_grad():
                        class_out_logit_val, _ = interaction_classifier_model(pbb_tensor_for_model.unsqueeze(0), h_mask_tensor_for_model.unsqueeze(0), f_mask_tensor_for_model.unsqueeze(0))
                    interaction_prob_score = torch.sigmoid(class_out_logit_val).squeeze().item()
                    
                    pbb_map_x1_calc=torch.min(h_b_current_det[0],w_b_current_det[0]); pbb_map_y1_calc=torch.min(h_b_current_det[1],w_b_current_det[1])
                    pbb_map_x2_calc=torch.max(h_b_current_det[2],w_b_current_det[2]); pbb_map_y2_calc=torch.max(h_b_current_det[3],w_b_current_det[3])
                    pbb_for_map_tensor_output = torch.stack([pbb_map_x1_calc,pbb_map_y1_calc,pbb_map_x2_calc,pbb_map_y2_calc])

                    loop_image_level_predictions_for_maxout_list.append({
                        "h_box": h_b_current_det, "w_box": w_b_current_det, "h_score":h_s_current_det.item(), "w_score":w_s_current_det.item(),
                        "interaction_score":interaction_prob_score, "pbb_for_map": pbb_for_map_tensor_output
                    })
        
        final_predictions_this_image_output_list = loop_image_level_predictions_for_maxout_list
        if apply_max_out_flag:
            final_predictions_this_image_output_list = apply_max_out_detections(loop_image_level_predictions_for_maxout_list)

        pred_boxes_for_map_metric_tensor = torch.stack([p_item_dict["pbb_for_map"] for p_item_dict in final_predictions_this_image_output_list]).float().to(current_device) if final_predictions_this_image_output_list else torch.empty((0,4),device=current_device)
        pred_scores_for_map_metric_tensor = torch.tensor([p_item_dict["h_score"]*p_item_dict["w_score"]*p_item_dict["interaction_score"] for p_item_dict in final_predictions_this_image_output_list]).float().to(current_device) if final_predictions_this_image_output_list else torch.empty(0,device=current_device)
        pred_labels_for_map_metric_tensor = torch.zeros(len(final_predictions_this_image_output_list), dtype=torch.long, device=current_device)

        loop_current_img_preds_dict = dict(boxes=pred_boxes_for_map_metric_tensor, scores=pred_scores_for_map_metric_tensor, labels=pred_labels_for_map_metric_tensor)
        all_predictions_for_map_metric.append(loop_current_img_preds_dict)
        
        drawn_for_this_image_counter = 0
        if num_visual_outputs_per_img > 0 :
            sorted_preds_for_visualization = sorted(final_predictions_this_image_output_list, key=lambda x_sort_item: x_sort_item["interaction_score"], reverse=True)
            for pred_to_draw_item in sorted_preds_for_visualization:
                if drawn_for_this_image_counter >= num_visual_outputs_per_img: break
                if pred_to_draw_item["interaction_score"] > visual_score_thr:
                    img_copy_for_drawing = loop_pil_image_rgb_original.copy()
                    output_filename_viz = f"img_{loop_img_idx:04d}_pred_{drawn_for_this_image_counter}_score_{pred_to_draw_item['interaction_score']:.2f}.png"
                    output_filepath_viz = os.path.join(visual_output_main_dir, output_filename_viz)
                    draw_detections_on_image(
                        img_copy_for_drawing,
                        pred_to_draw_item["h_box"],
                        pred_to_draw_item["w_box"],
                        pred_to_draw_item["interaction_score"],
                        output_filepath_viz
                    )
                    drawn_for_this_image_counter += 1

    print("\nOrtalama Hassasiyet (AP) hesaplanıyor...")
    try:
        preds_for_metric_calc = [{k_metric: v_metric.to(current_device) for k_metric,v_metric in p_metric.items()} for p_metric in all_predictions_for_map_metric]
        targets_for_metric_calc = [{k_metric: v_metric.to(current_device) for k_metric,v_metric in t_metric.items()} for t_metric in all_targets_for_map_metric]
        map_metric_calculator.update(preds_for_metric_calc, targets_for_metric_calc)
        
        map_results_final_dict = map_metric_calculator.compute()
        print(f"Torchmetrics mAP sonuçları: {map_results_final_dict}")
        ap_hold_value_final = map_results_final_dict['map_50'].item()
        print(f"  AP_hold (mAP@0.50 IOU_threshold={config.IOU_THRESHOLD_FOR_AP_MATCHING}): {ap_hold_value_final:.4f}")
        
        results_output_final_str = f"Classifier Model: {classifier_model_file_path}\nLoaded Epoch: {loaded_epoch_val}\nBackbone: {ckpt_backbone_name_val}\nColor Space: {ckpt_color_space_val}\n"
        results_output_final_str += f"Human Det Thr: {config.HUMAN_DETECTION_THRESHOLD}\nWeapon Det Thr: {config.WEAPON_DETECTION_THRESHOLD} (GT Weapons Used: {use_gt_weapons_for_debug_flag})\n"
        results_output_final_str += f"Max-Out Applied: {apply_max_out_flag}\n"
        results_output_final_str += f"Visuals Saved (per image if enabled): {num_visual_outputs_per_img} with score > {visual_score_thr}\n"
        results_output_final_str += f"AP_hold (mAP@.50 with IOU_thr={config.IOU_THRESHOLD_FOR_AP_MATCHING}): {ap_hold_value_final:.4f}\n"
        results_output_final_str += f"Full torchmetrics results: {map_results_final_dict}\n"

        output_metrics_filename_final = f"ap_metrics_fullsys_{os.path.splitext(os.path.basename(classifier_model_file_path))[0]}.txt"
        with open(os.path.join(config.RESULTS_DIR, output_metrics_filename_final), "w") as f_out_final: f_out_final.write(results_output_final_str)
        print(f"AP metrikleri {output_metrics_filename_final} dosyasına kaydedildi.")

    except Exception as e_ap_final:
        print(f"AP hesaplanırken hata oluştu: {e_ap_final}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Uçtan Uca Etkileşim Tespit Sistemini Değerlendirme ve Görselleştirme")
    parser.add_argument("classifier_model_path", type=str, help="Eğitilmiş sınıflandırıcı modeli (.pth)")
    parser.add_argument("--no_gt_weapons", action="store_false", dest="use_gt_weapons", help="GT silah kutuları yerine tespit kullan")
    parser.add_argument("--no_max_out", action="store_false", dest="apply_max_out", help="Max-Out Detections uygulama")
    parser.add_argument("--num_visuals", type=int, default=1, help="Her görüntü için kaydedilecek maks görsel çıktı (yüksek skorlu)")
    parser.add_argument("--visual_thr", type=float, default=0.7, help="Görsel çıktı için min etkileşim skoru eşiği")
    parser.set_defaults(use_gt_weapons=True, apply_max_out=True)
    args = parser.parse_args()
    
    if not os.path.isfile(args.classifier_model_path):
        print(f"HATA: Sınıflandırıcı model yolu bulunamadı: {args.classifier_model_path}")
    else:
        evaluate_full_system(args.classifier_model_path, 
                             use_gt_weapons_for_debug_flag=args.use_gt_weapons,
                             apply_max_out_flag=args.apply_max_out,
                             num_visual_outputs_per_img=args.num_visuals,
                             visual_score_thr=args.visual_thr)