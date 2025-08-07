import cv2

# Load an image
image = cv2.imread("sample.jpg")  # Make sure sample.jpg is in the same folder
if image is None:
    print("Image not found.")
    exit()

# Display the image
cv2.imshow("Original Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Extend your script to save a copy:
cv2.imwrite("output.jpg", image)
print("Image saved as output.jpg")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale Image", gray)
cv2.imwrite("grayscale.jpg", gray)

# Resize the image
resized = cv2.resize(image, (200, 200))
cv2.imshow("Resized Image", resized)
cv2.imwrite("resized.jpg", resized)

cv2.waitKey(0)
cv2.destroyAllWindows()