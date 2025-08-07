import cv2
import pytesseract
import os

# Set image path
img_path = "images/log1.png"  # Replace with your image
if not os.path.exists(img_path):
    print("Image not found.")
    exit()

# Load image
img = cv2.imread(img_path)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Optional: thresholding for better results
thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

# Extract text using Tesseract
text = pytesseract.image_to_string(thresh)

print("=== Extracted Log Content ===")
print(text)

keywords = ["ERROR", "FAILED", "WARNING", "CRITICAL"]

print("\n=== Alerts Found ===")
for line in text.split('\n'):
    for keyword in keywords:
        if keyword in line:
            print(f"[{keyword}] {line.strip()}")