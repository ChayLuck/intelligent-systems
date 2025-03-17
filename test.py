import cv2
import numpy as np
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract.exe'

image_path = "img/Upper.jpg"
output_folder = "extracted_letters"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) #Black and white image (gray)

if image is None:
    raise FileNotFoundError(f"Image couldnt uploaded. Wrong path or corrupted file: {image_path}")

# Reverse colors (White characters - Black background)
_, thresh = cv2.threshold(image, 115, 255, cv2.THRESH_BINARY_INV) #threshold makes the image binary and 115 is the value determines the breakpoint

contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #find the edges of image, tree is hierarchy, chain is memory upgrade 

image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) #Converts the image to color back again

cv2.drawContours(image_color, contours, -1, (0, 255, 0), 2) #-1 mean full, middle is color green, 2 is thickness

scale = 0.22  # Scaling the image for better view
thresh_resized = cv2.resize(thresh, (0, 0), fx=scale, fy=scale) #fx width, fy height
image_resized = cv2.resize(image, (0, 0), fx=scale, fy=scale)
image_color_resized = cv2.resize(image_color, (0, 0), fx=scale, fy=scale)

cv2.imshow("Thresh", thresh_resized)
cv2.imshow("Base", image_resized)
cv2.imshow("Contours", image_color_resized)

# Draw rectangle for each contour
boxes = [cv2.boundingRect(c) for c in contours]
boxes = sorted(boxes, key=lambda x: (x[1], x[0]))  #First rows, then collumns

char_index = 1

#start to extract the characters for each characters
for x, y, w, h in boxes:
    roi = image[y:y + h, x:x + w] #Region of interest when detect a character it cuts it from the image

    # Reads the characters using OCR (Optic Character Recognition)
    char = pytesseract.image_to_string(roi, config='--psm 10').strip() #psm 10 is for single character

    # If character is empty or non-alphanumeric, skip
    if not char or not char.isalnum():
        print(f"Atlandı: ({x}, {y}, {w}, {h}) -> '{char}'")
        continue

    #save the characters image to the folder
    char_filename = f"{output_folder}/char_{char_index}.png"
    cv2.imwrite(char_filename, roi)

    #Let the user know which character is saved
    print(f"Karakter '{char}' kaydedildi: {char_filename}")

    char_index += 1 

cv2.waitKey(0)
cv2.destroyAllWindows()