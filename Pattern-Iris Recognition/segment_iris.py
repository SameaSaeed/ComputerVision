import cv2
import numpy as np

# Load eye image
image = cv2.imread("eye.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# Use Hough Circles to detect the iris (approximated as a dark circle)
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                           param1=100, param2=30, minRadius=20, maxRadius=60)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for (x, y, r) in circles[0, :1]:  # Use first detected circle
        cv2.circle(image, (x, y), r, (0, 255, 0), 2)
        iris = gray[y - r:y + r, x - r:x + r]
        cv2.imwrite("isolated_iris.jpg", iris)
        print("Iris segmented and saved as isolated_iris.jpg")
else:
    print("No iris detected.")

cv2.imshow("Iris Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()