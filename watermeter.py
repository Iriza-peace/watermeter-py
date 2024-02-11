import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


img = cv2.imread("water_meter.png")


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)


contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


largest_contour = max(contours, key=cv2.contourArea)


x, y, w, h = cv2.boundingRect(largest_contour)
roi = img[y:y+h, x:x+w]


text = pytesseract.image_to_string(roi, config='--psm 10')


print(text)