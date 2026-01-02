"""
Simple RTSP test viewer.

Usage:
  python testconnection.py

The script opens a hard-coded RTSP URL (from your request) but you can pass a different
URL as the first command-line argument.

Controls:
  q or ESC - quit

This script uses OpenCV (cv2). On Windows, install requirements with:
  pip install -r requirements.txt
"""

import sys
import time
import cv2

# Default RTSP URL (from user)
DEFAULT_RTSP = ""


def open_stream(url, timeout=10):
	"""Try to open a cv2.VideoCapture for the given URL. Returns capture or None."""
	start = time.time()
	cap = cv2.VideoCapture(url)
	# Some backends return immediately but report not opened until a bit later.
	while time.time() - start < timeout:
		if cap.isOpened():
			return cap
		time.sleep(0.3)
	# final check
	if cap.isOpened():
		return cap
	cap.release()
	return None


def main():
	url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RTSP

	print(f"Opening RTSP stream: {url}")

	cap = open_stream(url)
	if cap is None:
		print("Failed to open stream. Exiting.")
		return

	window_name = "RTSP Test (small)"
	cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
	# Set an initial small window size
	cv2.resizeWindow(window_name, 640, 360)

	reconnect_delay = 2.0

	try:
		while True:
			ret, frame = cap.read()
			if not ret or frame is None:
				print("Frame read failed, attempting reconnect...")
				cap.release()
				time.sleep(reconnect_delay)
				cap = open_stream(url)
				if cap is None:
					print("Reconnect failed, will retry...")
					time.sleep(reconnect_delay)
					continue
				else:
					print("Reconnected to stream.")
					continue

			# Resize frame to a smaller preview for a compact window while keeping aspect
			h, w = frame.shape[:2]
			target_w = 640
			target_h = int(target_w * h / w)
			small = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

			cv2.imshow(window_name, small)

			key = cv2.waitKey(1) & 0xFF
			if key == ord('q') or key == 27:
				print("Exit requested by user.")
				break

	finally:
		try:
			cap.release()
		except Exception:
			pass
		cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
