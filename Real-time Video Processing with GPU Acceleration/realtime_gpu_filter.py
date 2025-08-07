import cv2
import time
import numpy as np
import tensorflow as tf

use_gpu = len(tf.config.list_physical_devices('GPU')) > 0
print("Using GPU:", use_gpu)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for consistent processing
    frame = cv2.resize(frame, (640, 480))

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Start timing
    start = time.time()

    # Simulate GPU task using TensorFlow (dummy blur operation)
    img_tensor = tf.convert_to_tensor(gray, dtype=tf.float32)
    img_tensor = tf.reshape(img_tensor, [1, 480, 640, 1])
    kernel = tf.constant([[1/9.]*3]*3, dtype=tf.float32, shape=[3, 3, 1, 1])
    blurred = tf.nn.conv2d(img_tensor, kernel, strides=[1,1,1,1], padding='SAME')
    blurred_np = blurred.numpy().squeeze()

    # End timing
    end = time.time()
    elapsed = (end - start) * 1000  # in ms

    # Display
    cv2.imshow("Real-time Blurred (Simulated GPU)", blurred_np.astype(np.uint8))
    cv2.putText(frame, f"Processing Time: {elapsed:.2f} ms", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Original", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()