import cv2
import pytesseract
import os
import numpy as np

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract.exe'

# Resmi yükleme
image_path = "img/Upper.jpg"  # Resmin yolunu buraya yaz
output_folder = "extracted_letters"

# Klasör kontrolü
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# OpenCV ile resmi oku
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gürültü azaltma ve eşikleme
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blurred, 110, 255, cv2.THRESH_BINARY_INV)

# Harf konturlarını bulma
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Harfleri tek tek kesip kaydetme
#for i, contour in enumerate(contours):
    #x, y, w, h = cv2.boundingRect(contour)
    
    # Boyut filtresi (çok küçük nesneleri hariç tut)
    #if w > 10 and h > 10:
        #letter = image[y:y+h, x:x+w]
        #letter_path = os.path.join(output_folder, f"letter_{i}.png")
        #cv2.imwrite(letter_path, letter)

#print(f"Extracted {len(contours)} letters and saved them in {output_folder}")

#hImg, wImg,_ = image.shape
#boxes = pytesseract.image_to_boxes(image)
#for b in boxes.splitlines():
    #print(b)
    #b = b.split(' ')
    #x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    #cv2.rectangle(image, (x, hImg - y), (w, hImg - h), (0, 0, 255), 1)
    #cv2.putText(image, b[0], (x, hImg - y + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)

#cv2.imshow('Result', image)
#cv2.waitKey(0)

cv2.drawContours(gray, contours, -1, (0, 255, 0), 2)

cv2.imshow("Image", image)
cv2.imshow("Thresh", thresh)
cv2.imshow("Contours", gray)

boxes = [cv2.boundingRect(c) for c in contours]
boxes = sorted(boxes, key=lambda x: (x[1], x[0]))  # Satır ve sütun bazlı sıralama

#for (x, y, w, h) in boxes:
#    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Image", image)

cell_width = int(np.mean([w for x, y, w, h in boxes]))
cell_height = int(np.mean([h for x, y, w, h in boxes]))
print(cell_width, cell_height)

char_index = 1

for x, y, w, h in sorted(boxes, key=lambda b: (b[0], b[1])):  # Sütun bazlı sıralama
    roi = image[y:y + h, x:x + w]
    roi = cv2.resize(roi, (cell_width, cell_height))

    char = pytesseract.image_to_boxes(roi, config='--psm 10').strip()

    #TODO
    #if not char or not char.isalnum():
    #    continue

    char_filename = f"{output_folder}/char_{char_index}.png"

    cv2.imwrite(char_filename, roi)

    print(f"Karakter '{char}' kaydedildi: {char_filename}")

    char_index += 1

cv2.waitKey(0)
cv2.destroyAllWindows()
