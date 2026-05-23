import os
import cv2
import numpy as np

def compress_image(img, quality=80):
    """
    img: np.ndarray, shape = (240, 320, 3), BGR format
    """
    # JPEG compression
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

    # Return a one-dimensional bytes object.
    success, encoded_image = cv2.imencode('.jpg', img, encode_param)
    if not success:
        raise RuntimeError("Image encoding failed")

    return encoded_image.tobytes()

def compress_video_frames(array, quality=80):
    """
    Input:
        array: np.ndarray, shape = [H, W, C] or [T, H, W, C]
    Output:
        For a single frame, return bytes.
        For multiple frames, return a list of bytes with length T.
    """
    if array.ndim == 3:   # Single frame
        return compress_image(array, quality)

    elif array.ndim == 4: # Multiple frames
        T = array.shape[0]
        return [compress_image(array[i], quality) for i in range(T)]

    else:
        raise ValueError(f"Unsupported ndim {array.ndim}, expected 3 or 4.")


def decompress_image(encoded_bytes):
    """
    encoded_bytes: compressed JPEG bytes
    return: decoded BGR image with shape (height, width, 3)
    """
    nparr = np.frombuffer(encoded_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Image decoding failed")
    return img

def decompress_video_frames(data):
    """
    Input:
        data: bytes or [bytes, bytes, ...]
    Output:
        Single frame: return [H, W, C].
        Multiple frames: return an np.ndarray with shape [T, H, W, C].
    """
    if isinstance(data, bytes):
        return decompress_image(data)

    elif isinstance(data, list):
        frames = [decompress_image(b) for b in data]
        return np.stack(frames, axis=0)

    else:
        raise ValueError("Unsupported data type for decompression.")