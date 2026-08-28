import threading
import time

import cv2
import requests

# --------------------------------------------------------------------------
# Phone MJPEG streams (DroidCam, IP Webcam, various "IP Camera" apps) turn
# out to be unreliable when opened through cv2.VideoCapture()/FFmpeg -- some
# of these apps' multipart framing doesn't match what FFmpeg expects, even
# though the exact same URL displays fine in a browser (browsers just render
# the raw multipart stream directly).
#
# So instead of asking OpenCV to open the network stream, a background
# thread pulls the raw bytes itself via `requests` and scans for JPEG
# start/end markers (0xFFD8 ... 0xFFD9) to pull out each frame. This is
# what browsers effectively do under the hood, and it works with any of
# these apps regardless of exact header quirks.
# --------------------------------------------------------------------------

_lock = threading.Lock()
_thread = None
_stop_event = None

camera_url = None
_latest_jpeg = None  # raw JPEG bytes, straight from the phone's stream


_HEADERS = {
    # Some phone camera apps silently reject requests whose User-Agent
    # doesn't look like a browser -- this is likely why the stream loads
    # fine when you paste the URL into a browser tab but fails here.
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def _reader_loop(url, stop_event):
    global _latest_jpeg

    while not stop_event.is_set():
        try:
            resp = requests.get(url, headers=_HEADERS, stream=True, timeout=5)
            resp.raise_for_status()
            buffer = b""

            for chunk in resp.iter_content(chunk_size=4096):
                if stop_event.is_set():
                    resp.close()
                    return
                if not chunk:
                    continue

                buffer += chunk
                # Guard against unbounded growth if no JPEG markers are found
                # (e.g. the URL returned an HTML page, not an image stream).
                if len(buffer) > 5_000_000:
                    buffer = buffer[-500_000:]

                start = buffer.find(b"\xff\xd8")
                if start == -1:
                    continue
                end = buffer.find(b"\xff\xd9", start + 2)
                if end == -1:
                    continue

                jpg = buffer[start:end + 2]
                buffer = buffer[end + 2:]

                with _lock:
                    _latest_jpeg = jpg

            resp.close()
        except requests.RequestException as exc:
            # Surface the real reason in the console instead of failing
            # silently -- if this still doesn't connect, whatever prints
            # here is what we need to see.
            print(f"[camera_feed] connection to {url} failed: {exc!r}")

        if not stop_event.is_set():
            time.sleep(1)  # brief pause before retrying a dropped connection


def connect_camera(url):
    """Start pulling frames from a phone camera stream URL. Blocks briefly
    to confirm at least one real frame arrives before reporting success."""
    global _thread, _stop_event, camera_url, _latest_jpeg

    disconnect_camera()

    _latest_jpeg = None
    _stop_event = threading.Event()
    _thread = threading.Thread(
        target=_reader_loop, args=(url, _stop_event), daemon=True
    )
    _thread.start()

    # Wait up to ~6s for the first real frame before declaring success/failure.
    deadline = time.time() + 6
    while time.time() < deadline:
        with _lock:
            if _latest_jpeg is not None:
                camera_url = url
                return True
        time.sleep(0.2)

    print(f"[camera_feed] no frame received from {url} within 6s")
    disconnect_camera()
    return False


def disconnect_camera():
    global _thread, _stop_event, camera_url, _latest_jpeg

    if _stop_event is not None:
        _stop_event.set()
    if _thread is not None and _thread.is_alive():
        _thread.join(timeout=2)

    _thread = None
    _stop_event = None
    camera_url = None
    _latest_jpeg = None


def is_connected():
    return _thread is not None and _thread.is_alive() and _latest_jpeg is not None


def get_frame_jpeg():
    """Grab the most recent JPEG frame -- used to feed the /predict pipeline."""
    with _lock:
        return _latest_jpeg


def generate_frames():
    """Yield an MJPEG multipart stream built from the latest pulled frames."""
    last_sent = None

    while is_connected():
        with _lock:
            jpg = _latest_jpeg

        if jpg is not None and jpg is not last_sent:
            last_sent = jpg
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpg
                + b"\r\n"
            )
        else:
            time.sleep(0.03)