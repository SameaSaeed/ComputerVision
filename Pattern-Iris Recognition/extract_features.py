import cv2
import numpy as np
import numpy as np
from scipy.spatial.distance import cosine

iris = cv2.imread("isolated_iris.jpg", 0)

if iris is None:
    print("Iris image not found.")
    exit()

# Resize to a fixed size
iris = cv2.resize(iris, (64, 64))

# Compute histogram as feature vector
hist = cv2.calcHist([iris], [0], None, [256], [0, 256])
hist = cv2.normalize(hist, hist).flatten()

# Save feature vector
np.save("iris_features.npy", hist)
print("Feature vector saved as iris_features.npy")


f1 = np.load("iris_features.npy")
f2 = np.load("another_iris_features.npy")  # Use a second sample

similarity = 1 - cosine(f1, f2)
print("Similarity Score:", similarity)