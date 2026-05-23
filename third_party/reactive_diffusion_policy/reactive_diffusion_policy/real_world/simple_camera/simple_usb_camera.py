import cv2
from loguru import logger
import pyudev


class SimpleUSBCamera:
    def __init__(self, camera_index=0):
        self.context = pyudev.Context()
        self.camera_index = camera_index
        self.cap = None

    def start(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                logger.error("Could not open video device")
                raise Exception("Could not open video device")
            logger.info("Camera started")
        else:
            logger.warning("Camera is already running")

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Camera stopped")
        else:
            logger.warning("Camera is not running")

    def get_rgb_frame(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to capture image")
                raise Exception("Failed to capture image")
            return frame
        else:
            logger.error("Camera is not running")
            raise Exception("Camera is not running")


# Example usage:
if __name__ == "__main__":
    camera = SimpleUSBCamera(camera_index=8)

    try:
        camera.start()
        while True:
            frame = camera.get_rgb_frame()
            # Display the frame
            cv2.imshow('RGB Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        logger.exception(e)
    finally:
        camera.stop()
        cv2.destroyAllWindows()