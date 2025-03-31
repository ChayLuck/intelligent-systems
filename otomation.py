import os
import shutil
import re

# Klasör yolları
image_folder = "extracted_letters"
destination_folder = "big_extracted_letters"

# Eğer hedef klasör yoksa oluştur
os.makedirs(destination_folder, exist_ok=True)

i = 1  # Sayacı başlat

# Klasördeki tüm dosyaları al
for image_name in sorted(os.listdir(image_folder)):  # Dosyaları sıralı al
    old_path = os.path.join(image_folder, image_name)

    # Dosya adını ve uzantısını al
    name, ext = os.path.splitext(image_name)

    # Sondaki sayıyı temizle (örneğin: "char_52" → "char")
    temp_name = re.sub(r"_\d+$", "", name)

    # Yeni dosya adını oluştur: "big_char_1.png" formatında
    new_name = f"big_{temp_name}_{i}_{ext}"
    new_path = os.path.join(destination_folder, new_name)

    # Dosyayı yeni klasöre KOPYALA
    shutil.copy(old_path, new_path)

    print(f"{image_name} → {new_name}")  # İşlem çıktısını göster

    i += 1  # Sayaç artır

print("Tüm dosyalar başarıyla güncellendi ve kopyalandı!")