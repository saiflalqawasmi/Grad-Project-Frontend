import cv2

# Replace with the IP and port shown in your DroidCam app
DROIDCAM_URL = "http://192.168.31.66:4747/video"

cap = cv2.VideoCapture(DROIDCAM_URL)

if not cap.isOpened():
    print("Failed to connect to DroidCam. Check the IP/port and that both devices are on the same network.")
    exit()

print("Connected. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Lost connection to stream.")
        break

    cv2.imshow("DroidCam Live Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()