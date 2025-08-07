import cv2
import os
import numpy as np

def compute_histogram(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

image_dir = "minio-images"
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

base_hist = compute_histogram(image_files[0])  # Use first image as baseline
print("Baseline image:", image_files[0])

for img_path in image_files[1:]:
    hist = compute_histogram(img_path)
    score = cv2.compareHist(base_hist, hist, cv2.HISTCMP_CORREL)
    print(f"{img_path} - Similarity: {score:.3f}")
    if score < 0.7:
        print(">> Potential Anomaly Detected!")