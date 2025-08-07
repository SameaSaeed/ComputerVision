Install required tools:
sudo apt update
sudo apt install tesseract-ocr -y
sudo apt install libtesseract-dev -y
pip install pytesseract opencv-python

Place log screenshot images (e.g., log1.png, error_log.jpg) in a folder:
~/ocr-lab/images/

Step: Create OCR Log Extractor and analyzer
python3 ocr_log_reader.py

Bonus: Preprocessing Tips for Better Accuracy

Noise Reduction:
blur = cv2.medianBlur(gray, 3)

Invert text for white-on-black logs:
inverted = cv2.bitwise_not(gray)