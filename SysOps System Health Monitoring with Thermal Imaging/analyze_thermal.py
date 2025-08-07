import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "images/thermal1.jpg"  # Replace with your actual image
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Image not found.")
    exit()

plt.imshow(img, cmap='inferno')
plt.title("Thermal Image")
plt.colorbar(label='Temperature Intensity')
plt.show()

# Threshold to find hot zones
threshold = 200  # Adjust based on your image scale (0-255)
_, hot_mask = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

# Find contours (hot regions)
contours, _ = cv2.findContours(hot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img_color, (x, y), (x+w, y+h), (0, 0, 255), 2)

cv2.imshow("Detected Hotspots", img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(contours) >= 2:
    print("â ï¸  Warning: Multiple hotspots detected. Possible overheating risk.")
elif len(contours) == 1:
    print("â ï¸  One hotspot detected. Monitor the component.")
else:
    print("â  No critical heat anomalies detected.")