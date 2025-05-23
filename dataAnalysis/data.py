# dataAnalysis/data.py
import pandas as pd
import os

# Bu, data.py'nin çalıştırıldığı yerden projenin kök dizinine olan göreli yoldur.
# data.py, dataAnalysis/ içinde olduğu için ../ proje kökünü verir.
PROJECT_ROOT_FOR_DATAPY = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def parse_lfc_annotations_from_tagged_file(annotation_file_path,
                                         images_base_rel_to_project_root, # Örneğin "data/Training_Dataset/images/"
                                         default_image_ext=".jpg"):
    data = []
    skipped_lines_field_count = 0
    skipped_lines_conversion_error = 0
    skipped_lines_unknown_label = 0
    found_images_count = 0
    missing_images_count = 0
    missing_images_info = {} # Hangi dosyanın hangi satırda eksik olduğunu kaydetmek için

    # Görüntülerin mutlak temel dizini
    # images_base_rel_to_project_root, PROJECT_ROOT_FOR_DATAPY'ye göre göreceli olmalı
    # örn: data/Training_Dataset/images/
    # full_image_base_dir_abs = os.path.join(PROJECT_ROOT_FOR_DATAPY, images_base_rel_to_project_root)
    # Yukarıdaki satır yerine, annotation_file_path'a göre daha güvenli bir yol izleyelim:
    # annotation_file_path zaten mutlak. Bunun bir üst dizinindeki images/ klasörünü hedefliyoruz.
    # Bu, data/Training_Dataset/tagged_file.txt için data/Training_Dataset/images/ olur.
    annot_parent_dir = os.path.dirname(annotation_file_path)
    # 'images' klasörünün adı images_base_rel_to_project_root'un son kısmı olmalı
    images_folder_name = os.path.basename(images_base_rel_to_project_root.strip('/\\'))
    full_image_base_dir_abs = os.path.join(annot_parent_dir, images_folder_name)


    common_extensions = [default_image_ext.lower()] + [ext for ext in [".jpg", ".jpeg", ".png", ".bmp", ".gif"] if ext != default_image_ext.lower()]

    print(f"Annotasyon dosyası okunuyor: {annotation_file_path}")
    print(f"Görüntülerin aranacağı mutlak temel dizin: {full_image_base_dir_abs}")
    print(f"CSV yolları için kullanılacak Proje Kökü (relpath için): {PROJECT_ROOT_FOR_DATAPY}")


    with open(annotation_file_path, 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if not parts:
                continue

            # Dosya adı (uzantısız), 4 insan bbox, 4 silah bbox, 1 etiket = 10 parça
            if len(parts) < 10 :
                skipped_lines_field_count += 1
                continue

            try:
                filename_base = parts[0]
                path_for_csv = None # CSV'ye yazılacak göreli yol
                actual_image_filename_with_ext = None # Bulunan dosyanın tam adı

                image_found_in_loop = False
                for ext in common_extensions:
                    temp_filename_with_ext = filename_base + ext
                    abs_img_path_check = os.path.join(full_image_base_dir_abs, temp_filename_with_ext)

                    if os.path.exists(abs_img_path_check):
                        # Bulunan mutlak yoldan, PROJECT_ROOT_FOR_DATAPY'ye göre göreli yol oluştur
                        path_for_csv = os.path.relpath(abs_img_path_check, PROJECT_ROOT_FOR_DATAPY)
                        path_for_csv = path_for_csv.replace(os.sep, '/') # Platform bağımsızlığı için
                        actual_image_filename_with_ext = temp_filename_with_ext
                        found_images_count +=1
                        image_found_in_loop = True
                        break
                
                if not image_found_in_loop:
                    missing_images_count += 1
                    if filename_base not in missing_images_info:
                        missing_images_info[filename_base] = []
                    missing_images_info[filename_base].append(f"Satır {i+1} (Aranan Dizin: {full_image_base_dir_abs})")
                    continue # Bu satırı atla

                human_bbox_str = parts[1:5]
                weapon_bbox_str = parts[5:9]
                label_original_str = parts[9]

                human_bbox = [int(p) for p in human_bbox_str]
                weapon_bbox = [int(p) for p in weapon_bbox_str]
                label_original = int(label_original_str) # 0: non-interaction, 1: gun-human, 2: rifle-human

                # Etiketi binary'ye çevir: 0 (not_carrier), 1 (carrier)
                # Orijinal: 0=no_interaction, 1=gun, 2=rifle. Biz 1 ve 2'yi "carrier" (1) yapıyoruz.
                if label_original == 1 or label_original == 2:
                    binary_label = 1 # carrier
                elif label_original == 0:
                    binary_label = 0 # not_carrier
                else:
                    skipped_lines_unknown_label += 1
                    continue
                
                data.append({
                    'image_path': path_for_csv, # Proje köküne göre göreli yol
                    'filename_original': actual_image_filename_with_ext, # Sadece dosya adı ve uzantısı
                    'human_x1': human_bbox[0],'human_y1': human_bbox[1],'human_x2': human_bbox[2],'human_y2': human_bbox[3],
                    'weapon_x1': weapon_bbox[0],'weapon_y1': weapon_bbox[1],'weapon_x2': weapon_bbox[2],'weapon_y2': weapon_bbox[3],
                    'label_original': label_original,
                    'label': binary_label # Bizim kullanacağımız binary etiket
                })
            except ValueError:
                skipped_lines_conversion_error += 1
            except IndexError: # Bu zaten yukarıda handle ediliyor ama bir daha
                skipped_lines_field_count += 1
    
    print(f"\nİşlem Tamamlandı: {annotation_file_path}")
    print(f"Toplam okunan satır: {i+1}")
    print(f"İşlenen geçerli satır (veriye eklenen): {len(data)}")
    print(f"Bulunan görüntü sayısı: {found_images_count} (benzersiz olmayabilir, her satır için kontrol edilir)")
    print(f"Eksik görüntü nedeniyle atlanan satır sayısı: {missing_images_count}")
    print(f"Alan sayısı yetersiz olduğu için atlanan satır: {skipped_lines_field_count}")
    print(f"Değer dönüştürme hatası nedeniyle atlanan satır: {skipped_lines_conversion_error}")
    print(f"Bilinmeyen etiket nedeniyle atlanan satır: {skipped_lines_unknown_label}")

    if missing_images_count > 0:
        print("\nEksik Görüntü Detayları (ilk birkaç örnek):")
        count = 0
        for fname, lines in missing_images_info.items():
            print(f"  Dosya tabanı '{fname}': {lines}")
            count += 1
            if count >= 5: break # Çok fazla yazdırmamak için

    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    # Kullanıcının tercih ettiği varsayılan uzantı (örn. .png veya .jpg)
    # Eğer tagged_file.txt'deki isimler uzantısızsa bu önemli.
    # Eğer tagged_file.txt'deki isimler zaten uzantılıysa, common_extensions listesi yine de diğerlerini dener.
    IMAGE_EXTENSION = ".png" # VEYA ".jpg"
    
    # Eğitim verisi için yollar
    # Bu yollar, PROJECT_ROOT_FOR_DATAPY'ye göre GÖRECELİ olmalı ki CSV'ye doğru yazılsın
    # Ama parse_lfc_annotations_from_tagged_file fonksiyonu MUTLAK yol bekliyor annotation_file_path için.
    train_annotations_file_abs = os.path.join(PROJECT_ROOT_FOR_DATAPY, "data", "Training_Dataset", "tagged_file.txt")
    # images_base_rel_to_project_root: CSV'ye yazılacak path'in oluşumunda relpath() için referans olacak
    # ve ayrıca parse_lfc_... fonksiyonu içinde tagged_file.txt'nin yanındaki 'images' klasörünü bulmak için kullanılacak.
    train_images_base_rel_to_project_root = "data/Training_Dataset/images/"
    
    print("--- Eğitim Verisi İşleniyor ---")
    if not os.path.exists(train_annotations_file_abs):
        print(f"Hata: Eğitim annotasyon dosyası bulunamadı: {train_annotations_file_abs}")
    else:
        train_df = parse_lfc_annotations_from_tagged_file(
            train_annotations_file_abs,
            train_images_base_rel_to_project_root,
            default_image_ext=IMAGE_EXTENSION
        )
        if not train_df.empty:
            print("\nEğitim Veri Çerçevesi Başlığı:")
            print(train_df.head())
            save_path_train = os.path.join(os.path.dirname(__file__), "train_annotations_parsed.csv")
            train_df.to_csv(save_path_train, index=False)
            print(f"\nEğitim annotasyonları '{os.path.abspath(save_path_train)}' dosyasına kaydedildi.")
        else:
            print("Eğitim verisi için DataFrame oluşturulamadı.")

    # Test verisi için yollar
    test_annotations_file_abs = os.path.join(PROJECT_ROOT_FOR_DATAPY, "data", "Test_Dataset", "tagged_file.txt")
    test_images_base_rel_to_project_root = "data/Test_Dataset/images/"
    
    print("\n\n--- Test Verisi İşleniyor ---")
    if not os.path.exists(test_annotations_file_abs):
        print(f"Hata: Test annotasyon dosyası bulunamadı: {test_annotations_file_abs}")
    else:
        test_df = parse_lfc_annotations_from_tagged_file(
            test_annotations_file_abs,
            test_images_base_rel_to_project_root,
            default_image_ext=IMAGE_EXTENSION
        )
        if not test_df.empty:
            print("\nTest Veri Çerçevesi Başlığı:")
            print(test_df.head())
            save_path_test = os.path.join(os.path.dirname(__file__), "test_annotations_parsed.csv")
            test_df.to_csv(save_path_test, index=False)
            print(f"\nTest annotasyonları '{os.path.abspath(save_path_test)}' dosyasına kaydedildi.")
        else:
            print("Test verisi için DataFrame oluşturulamadı.")