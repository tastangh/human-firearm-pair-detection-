# dataAnalysis/data.py
import pandas as pd
import os

# This should be the absolute path to your project's root directory
# human-firearm-pair-detection-
PROJECT_ROOT_FOR_DATAPY = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def parse_lfc_annotations_from_tagged_file(annotation_file_path,
                                         images_subdir_name, # Örneğin "images/"
                                         default_image_ext=".jpg"):
    data = []
    # ... (skipped lines counters) ...
    found_images_count = 0
    missing_images_info = {}

    common_extensions = [default_image_ext.lower()] + [ext for ext in [".jpg", ".jpeg", ".png", ".bmp", ".gif"] if ext != default_image_ext.lower()]

    # annotation_file_path is ALREADY absolute when passed from __main__
    annot_dir = os.path.dirname(annotation_file_path)
    # This is the absolute path to the images directory
    # e.g., /media/.../human-firearm-pair-detection-/data/Training_Dataset/images/
    full_image_base_dir_abs = os.path.join(annot_dir, images_subdir_name)

    print(f"Annotasyon dosyası okunuyor: {annotation_file_path}")
    print(f"Görüntülerin mutlak temel dizini: {full_image_base_dir_abs}")
    print(f"Denenecek varsayılan ve yaygın görüntü uzantıları: {common_extensions}")
    print(f"CSV yolları için kullanılacak Proje Kökü (PROJECT_ROOT_FOR_DATAPY): {PROJECT_ROOT_FOR_DATAPY}")


    with open(annotation_file_path, 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if not parts:
                continue

            # ... (expected_parts logic) ...

            if len(parts) < 10 : # Basic check, might need adjustment based on expected_parts
                # skipped_lines_field_count += 1 # Already defined
                continue

            try:
                filename_base = parts[0]
                path_for_csv = None
                actual_image_filename_with_ext = None

                for ext in common_extensions:
                    temp_filename = filename_base + ext
                    # This is the absolute path to the specific image file being checked
                    abs_img_path_check = os.path.join(full_image_base_dir_abs, temp_filename)

                    if os.path.exists(abs_img_path_check):
                        # Create path relative to PROJECT_ROOT_FOR_DATAPY
                        # This should result in "data/Training_Dataset/images/file.png"
                        path_for_csv = os.path.relpath(abs_img_path_check, PROJECT_ROOT_FOR_DATAPY)
                        path_for_csv = path_for_csv.replace(os.sep, '/') # Platform independence
                        actual_image_filename_with_ext = temp_filename
                        found_images_count +=1
                        break

                if not path_for_csv:
                    # ... (missing image logic) ...
                    continue

                # ... (bbox and label parsing) ...
                human_bbox_str = parts[1:5]
                weapon_bbox_str = parts[5:9]
                label_original_str = parts[9]

                human_bbox = [int(p) for p in human_bbox_str]
                weapon_bbox = [int(p) for p in weapon_bbox_str]
                label_original = int(label_original_str)

                if label_original == 1:
                    label = 1
                elif label_original == 2 or label_original == 0:
                    label = 0
                else:
                    # skipped_lines_unknown_label += 1 # Already defined
                    continue
                
                # DEBUGGING: Print the path_for_csv
                if i < 5 : # Print for the first few entries
                    print(f"  -> Oluşturulan path_for_csv: {path_for_csv} (from abs: {abs_img_path_check})")

                data.append({
                    'image_path': path_for_csv,
                    'filename_base': actual_image_filename_with_ext,
                    'human_x1': human_bbox[0],'human_y1': human_bbox[1],'human_x2': human_bbox[2],'human_y2': human_bbox[3],
                    'weapon_x1': weapon_bbox[0],'weapon_y1': weapon_bbox[1],'weapon_x2': weapon_bbox[2],'weapon_y2': weapon_bbox[3],
                    'label_original': label_original, 'label': label
                })
            except ValueError:
                # skipped_lines_conversion_error += 1 # Already defined
                pass # Add proper handling or logging
            except IndexError:
                # skipped_lines_field_count += 1 # Already defined
                pass # Add proper handling or logging
    # ... (rest of the function) ...
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    IMAGE_EXTENSION = ".png"
    
    # Construct absolute paths for annotation files from the perspective of data.py
    train_annotations_file_abs = os.path.abspath(os.path.join(PROJECT_ROOT_FOR_DATAPY, "data", "Training_Dataset", "tagged_file.txt"))
    train_images_subdir = "images/" 
    
    print("--- Eğitim Verisi İşleniyor ---")
    if not os.path.exists(train_annotations_file_abs):
        print(f"Hata: Eğitim annotasyon dosyası bulunamadı: {train_annotations_file_abs}")
    else:
        print(f"'{train_annotations_file_abs}' annotasyon dosyası işleniyor...")
        train_df = parse_lfc_annotations_from_tagged_file(train_annotations_file_abs,
                                                         train_images_subdir,
                                                         default_image_ext=IMAGE_EXTENSION)
        if not train_df.empty:
            print(train_df.head())
            # ... (save to CSV in dataAnalysis) ...
            train_df.to_csv(os.path.join(os.path.dirname(__file__), "train_annotations_parsed.csv"), index=False)
            print(f"\nEğitim annotasyonları '{os.path.abspath(os.path.join(os.path.dirname(__file__), 'train_annotations_parsed.csv'))}' dosyasına kaydedildi.")

    test_annotations_file_abs = os.path.abspath(os.path.join(PROJECT_ROOT_FOR_DATAPY, "data", "Test_Dataset", "tagged_file.txt"))
    test_images_subdir = "images/"
    print("\n\n--- Test Verisi İşleniyor ---")
    if not os.path.exists(test_annotations_file_abs):
        print(f"Hata: Test annotasyon dosyası bulunamadı: {test_annotations_file_abs}")
    else:
        print(f"'{test_annotations_file_abs}' annotasyon dosyası işleniyor...")
        test_df = parse_lfc_annotations_from_tagged_file(test_annotations_file_abs,
                                                        test_images_subdir,
                                                        default_image_ext=IMAGE_EXTENSION)
        if not test_df.empty:
            print(test_df.head())
            # ... (save to CSV in dataAnalysis) ...
            test_df.to_csv(os.path.join(os.path.dirname(__file__), "test_annotations_parsed.csv"), index=False)
            print(f"\nTest annotasyonları '{os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_annotations_parsed.csv'))}' dosyasına kaydedildi.")