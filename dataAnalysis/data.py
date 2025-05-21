import pandas as pd
import os

def parse_lfc_annotations_from_tagged_file(annotation_file_path, image_base_dir, default_image_ext=".jpg"):
    """
    LFC veri setinin 'tagged_file.txt' formatındaki annotasyon dosyasını okur
    ve bir Pandas DataFrame'e dönüştürür.

    Args:
        annotation_file_path (str): Annotasyon dosyasının yolu.
        image_base_dir (str): Görüntülerin bulunduğu ana klasör.
        default_image_ext (str): Görüntülerin varsayılan uzantısı (örn: ".jpg", ".png").

    Returns:
        pandas.DataFrame: Parsed edilmiş annotasyonları içeren DataFrame.
    """
    data = []
    skipped_lines_field_count = 0
    skipped_lines_conversion_error = 0
    skipped_lines_unknown_label = 0
    found_images_count = 0
    missing_images_info = {} # {filename_base: count_of_references}
    
    # Yaygın uzantılar listesi
    common_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
    if default_image_ext.lower() in common_extensions:
        common_extensions.insert(0, common_extensions.pop(common_extensions.index(default_image_ext.lower()))) # Varsayılanı başa al

    print(f"Annotasyon dosyası okunuyor: {annotation_file_path}")
    print(f"Görüntülerin temel dizini: {image_base_dir}")
    print(f"Denenecek varsayılan ve yaygın görüntü uzantıları: {common_extensions}")

    with open(annotation_file_path, 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split() 
            if not parts:
                continue

            expected_parts = 10
            if annotation_file_path.endswith("tagged_file_with_object_label.txt"):
                expected_parts = 11

            if len(parts) != expected_parts:
                skipped_lines_field_count += 1
                continue

            try:
                filename_base = parts[0]
                
                # Görüntü dosyasını bulmayı dene
                image_full_path = None
                actual_image_filename = ""

                for ext in common_extensions:
                    temp_filename = filename_base + ext
                    temp_path = os.path.join(image_base_dir, temp_filename)
                    temp_path = os.path.normpath(temp_path)
                    if os.path.exists(temp_path):
                        image_full_path = temp_path
                        actual_image_filename = temp_filename
                        found_images_count +=1
                        break
                
                if not image_full_path:
                    missing_images_info[filename_base] = missing_images_info.get(filename_base, 0) + 1
                    if missing_images_info[filename_base] <= 5 and sum(missing_images_info.values()) <=20 :
                         # İlk birkaç hatayı göster
                         print(f"Satır {i+1}: Uyarı: Görüntü dosyası '{filename_base}' uzantılarla ({', '.join(common_extensions)}) bulunamadı. Temel dizin: {image_base_dir}")
                    continue # Görüntü bulunamazsa bu satırı atla


                human_bbox_str = parts[1:5]
                weapon_bbox_str = parts[5:9]
                label_original_str = parts[9]

                human_bbox = [int(p) for p in human_bbox_str]
                weapon_bbox = [int(p) for p in weapon_bbox_str]
                label_original = int(label_original_str)

                # Etiketleri map'leyelim:
                # Varsayım: 1 -> taşıyor (hedefimiz 1)
                #           2 -> taşımıyor (hedefimiz 0)
                # TODO: Kullanıcıya sor: Orijinal etiket 0 ne anlama geliyor?
                # Eğer 0 da "taşımıyor" ise, aşağıdaki koşulu `elif label_original == 2 or label_original == 0:` yap.
                if label_original == 1:
                    label = 1 # Taşıyor
                elif label_original == 2:
                    label = 0 # Taşımıyor
                # Orijinal etiket 0 ise şimdilik atlıyoruz.
                # elif label_original == 0: 
                #     label = 0 # Eğer 0 da taşımıyor demekse
                #     pass
                else:
                    # print(f"Satır {i+1}: Uyarı: Beklenmeyen orijinal etiket değeri ({label_original}). Satır atlanıyor: '{line.strip()}'")
                    skipped_lines_unknown_label += 1
                    continue 

                data.append({
                    'image_path': image_full_path, # Bulunan tam dosya yolu
                    'filename_base': actual_image_filename, # Bulunan dosya adı (uzantılı)
                    'human_x1': human_bbox[0],
                    'human_y1': human_bbox[1],
                    'human_x2': human_bbox[2],
                    'human_y2': human_bbox[3],
                    'weapon_x1': weapon_bbox[0],
                    'weapon_y1': weapon_bbox[1],
                    'weapon_x2': weapon_bbox[2],
                    'weapon_y2': weapon_bbox[3],
                    'label_original': label_original,
                    'label': label
                })
            except ValueError:
                skipped_lines_conversion_error += 1
            except IndexError:
                skipped_lines_field_count += 1 # Bu genellikle ilk alan sayısı kontrolünde yakalanır

    if skipped_lines_field_count > 0:
        print(f"Uyarı: {skipped_lines_field_count} satır yetersiz/fazla alan sayısı nedeniyle atlandı.")
    if skipped_lines_conversion_error > 0:
        print(f"Uyarı: {skipped_lines_conversion_error} satır bounding box veya etiket format hatası nedeniyle atlandı (sayısal olmayan değer).")
    if skipped_lines_unknown_label > 0:
        print(f"Uyarı: {skipped_lines_unknown_label} satır bilinmeyen orijinal etiket değeri nedeniyle atlandı.")

    total_missing_image_references = sum(missing_images_info.values())
    unique_missing_images = len(missing_images_info)
    if total_missing_image_references > 0:
        print(f"Uyarı: Toplam {total_missing_image_references} referansta {unique_missing_images} farklı görüntü dosyası belirtilen yolda bulunamadı (denenen uzantılar: {', '.join(common_extensions)}).")
        print(f"Lütfen `image_base_dir` ('{os.path.abspath(image_base_dir)}') yolunu ve görüntülerin varlığını/adlarını kontrol edin.")
        if unique_missing_images <= 20: # Çok fazla değilse listele
            print("Bulunamayan bazı görüntü dosyası temelleri (ilk 20):")
            for k, (fname_base, count) in enumerate(missing_images_info.items()):
                if k >= 20: break
                print(f"  - {fname_base} ({count} kez referans verilmiş)")
    
    print(f"Toplam {found_images_count} görüntü referansı için dosya bulundu.")

    df = pd.DataFrame(data)

    if not df.empty:
        df['apbb_x1'] = df[['human_x1', 'weapon_x1']].min(axis=1)
        df['apbb_y1'] = df[['human_y1', 'weapon_y1']].min(axis=1)
        df['apbb_x2'] = df[['human_x2', 'weapon_x2']].max(axis=1)
        df['apbb_y2'] = df[['human_y2', 'weapon_y2']].max(axis=1)
    else:
        print(f"Uyarı: {annotation_file_path} dosyasından hiçbir geçerli veri okunamadı veya bulunan görüntülerle eşleşen annotasyon yok.")

    return df

# --- Script'i Kullanma Ana Bloğu ---
if __name__ == "__main__":
    # Görüntülerin varsayılan uzantısı (eğer çoğunlukla bu ise)
    # Script diğer yaygın uzantıları da deneyecektir (.jpeg, .png, .bmp)
    IMAGE_EXTENSION = ".jpg" 

    # Eğitim verileri
    train_annotations_file = "../data/Training_Dataset/tagged_file.txt"
    train_images_base_dir = "../data/Training_Dataset/images/"

    print("--- Eğitim Verisi İşleniyor ---")
    if not os.path.exists(train_annotations_file):
        print(f"Hata: Eğitim annotasyon dosyası bulunamadı: {os.path.abspath(train_annotations_file)}")
    elif not os.path.isdir(train_images_base_dir):
        print(f"Hata: Eğitim görüntü klasörü bulunamadı: {os.path.abspath(train_images_base_dir)}")
    else:
        print(f"'{os.path.abspath(train_annotations_file)}' annotasyon dosyası ve ")
        print(f"'{os.path.abspath(train_images_base_dir)}' görüntü klasörü işleniyor...")
        train_df = parse_lfc_annotations_from_tagged_file(train_annotations_file, train_images_base_dir, default_image_ext=IMAGE_EXTENSION)

        if not train_df.empty:
            print("\n--- train_df İlk 5 Satır ---")
            print(train_df.head())
            print(f"\nToplam {len(train_df)} eğitim örneği (insan-silah çifti) bulundu ve görüntü dosyalarıyla eşleştirildi.")

            print("\n--- Eğitim Verisi Etiket Dağılımı (0: taşımıyor, 1: taşıyor) ---")
            print(train_df['label'].value_counts(dropna=False))

            print("\n--- Eğitim Verisi Orijinal Etiket Dağılımı ---")
            print(train_df['label_original'].value_counts(dropna=False))

            if not train_df.empty and 'image_path' in train_df.columns:
                example_path = train_df['image_path'].iloc[0]
                print(f"\nDataFrame'deki örnek resim yolu: {example_path}")
                if not os.path.exists(example_path): # Bu kontrol artık parse fonksiyonu içinde yapılıyor
                    print(f"Uyarı: Örnek resim yolu {example_path} diskte bulunamadı (Bu bir sorun olmamalı, parse fonksiyonu zaten kontrol etti).")
            
            train_df.to_csv("train_annotations_parsed.csv", index=False)
            print(f"\nEğitim annotasyonları '{os.path.abspath('train_annotations_parsed.csv')}' dosyasına kaydedildi.")
        else:
            print("Eğitim DataFrame'i boş. Lütfen annotasyon dosyası içeriğini, formatını veya yollarını kontrol edin.")

    # Test verileri
    test_annotations_file = "../data/Test_Dataset/tagged_file.txt"
    test_images_base_dir = "../data/Test_Dataset/images/"

    print("\n\n--- Test Verisi İşleniyor ---")
    if not os.path.exists(test_annotations_file):
        print(f"Hata: Test annotasyon dosyası bulunamadı: {os.path.abspath(test_annotations_file)}")
    elif not os.path.isdir(test_images_base_dir):
        print(f"Hata: Test görüntü klasörü bulunamadı: {os.path.abspath(test_images_base_dir)}")
    else:
        print(f"'{os.path.abspath(test_annotations_file)}' annotasyon dosyası ve ")
        print(f"'{os.path.abspath(test_images_base_dir)}' görüntü klasörü işleniyor...")
        test_df = parse_lfc_annotations_from_tagged_file(test_annotations_file, test_images_base_dir, default_image_ext=IMAGE_EXTENSION)

        if not test_df.empty:
            print("\n--- test_df İlk 5 Satır ---")
            print(test_df.head())
            print(f"\nToplam {len(test_df)} test örneği (insan-silah çifti) bulundu ve görüntü dosyalarıyla eşleştirildi.")

            print("\n--- Test Verisi Etiket Dağılımı (0: taşımıyor, 1: taşıyor) ---")
            print(test_df['label'].value_counts(dropna=False))

            print("\n--- Test Verisi Orijinal Etiket Dağılımı ---")
            print(test_df['label_original'].value_counts(dropna=False))
            
            test_df.to_csv("test_annotations_parsed.csv", index=False)
            print(f"\nTest annotasyonları '{os.path.abspath('test_annotations_parsed.csv')}' dosyasına kaydedildi.")
        else:
            print("Test DataFrame'i boş. Lütfen annotasyon dosyası içeriğini, formatını veya yollarını kontrol edin.")