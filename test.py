import cv2
import numpy as np
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract.exe'

image_path = "img/Upper.jpg"
output_folder = "extracted_letters"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Görsel yüklenemedi. Dosya yolu yanlış veya dosya bozuk: {image_path}")

# Görüntüyü ters çevir (Beyaz karakterler - Siyah arka plan)
_, thresh = cv2.threshold(image, 115, 255, cv2.THRESH_BINARY_INV)

contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

cv2.drawContours(image_color, contours, -1, (0, 255, 0), 2)

scale = 0.22  # Görüntüyü %50 küçültme
thresh_resized = cv2.resize(thresh, (0, 0), fx=scale, fy=scale)
image_resized = cv2.resize(image, (0, 0), fx=scale, fy=scale)
image_color_resized = cv2.resize(image_color, (0, 0), fx=scale, fy=scale)

cv2.imshow("Thresh", thresh_resized)
cv2.imshow("Characters Only", image_resized)
cv2.imshow("Contours", image_color_resized)

# Bounding box'ları sırala (Önce satır bazlı, sonra sütun bazlı)
boxes = [cv2.boundingRect(c) for c in contours]
boxes = sorted(boxes, key=lambda x: (x[1], x[0]))  # Önce satır bazlı, sonra sütun bazlı sıralama

char_index = 1


for x, y, w, h in boxes:
    roi = image[y:y + h, x:x + w]

    # OCR kullanarak karakteri oku
    char = pytesseract.image_to_string(roi, config='--psm 10').strip()

    # Eğer karakter boşsa veya alfanumerik değilse kaydetme
    if not char or not char.isalnum():
        print(f"Atlandı: ({x}, {y}, {w}, {h}) -> '{char}'")
        continue

    char_filename = f"{output_folder}/char_{char_index}.png"
    cv2.imwrite(char_filename, roi)

    print(f"Karakter '{char}' kaydedildi: {char_filename}")

    char_index += 1 

cv2.waitKey(0)
cv2.destroyAllWindows()