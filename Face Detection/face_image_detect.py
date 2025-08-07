# Haar cascade files are included with OpenCV. Path:
# /usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml

import cv2

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load image
image = cv2.imread("face.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Show image
cv2.imshow("Face Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()