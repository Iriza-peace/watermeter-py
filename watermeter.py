import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the image
img = cv2.imread("water_meter.png")

# Preprocess the image to improve text extraction
# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding to improve contrast
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# Find contours in the image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the contour with the largest area
largest_contour = max(contours, key=cv2.contourArea)

# Crop the image to the region of interest
x, y, w, h = cv2.boundingRect(largest_contour)
roi = img[y:y+h, x:x+w]

# Use Tesseract to extract text from the ROI
text = pytesseract.image_to_string(roi, config='--psm 10')

# Print the extracted text
print(text)