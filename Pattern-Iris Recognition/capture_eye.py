import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("Press 's' to save an image of your eye.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Eye Capture", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite("eye.jpg", frame)
        print("Image saved as eye.jpg")
        break
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()