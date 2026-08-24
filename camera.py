import cv2

CAMERA_URL = request.form["camera_url"]

cap = cv2.VideoCapture(CAMERA_URL)

if not cap.isOpened():
    print("Cannot connect to phone camera")
    exit()

print("Phone camera connected")

while True:
    success, frame = cap.read()

    if not success:
        print("Failed to read frame")
        break

    cv2.imshow("Phone Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
