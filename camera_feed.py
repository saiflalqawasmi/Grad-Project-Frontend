import cv2

camera = None
camera_url = None


def connect_camera(url):
    """Open a stream from a phone IP-camera app (e.g. IP Webcam's /video URL)."""
    global camera, camera_url

    if camera is not None:
        camera.release()

    camera = cv2.VideoCapture(url)
    camera_url = url if camera.isOpened() else None

    return camera.isOpened()


def disconnect_camera():
    global camera, camera_url

    if camera is not None:
        camera.release()

    camera = None
    camera_url = None


def is_connected():
    return camera is not None and camera.isOpened()


def get_frame_jpeg():
    """Grab a single frame as JPEG bytes — used to feed the /predict pipeline."""
    global camera

    if camera is None:
        return None

    success, frame = camera.read()
    if not success:
        return None

    ret, buffer = cv2.imencode(".jpg", frame)
    if not ret:
        return None

    return buffer.tobytes()


def generate_frames():
    global camera

    while camera is not None:
        success, frame = camera.read()

        if not success:
            break

        ret, buffer = cv2.imencode(".jpg", frame)

        if not ret:
            continue

        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame
            + b"\r\n"
        )
