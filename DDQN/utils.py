import numpy as np
import cv2

def preprocess(frame, crop=(0,210,0,160), frame_size=84):
    """
    Preprocess
    crop=(h,dh,w,dw)
    crop=(0,210,0,160) : default screen
    """
    frame = np.reshape(cv2.resize(frame[crop[0]:crop[1], crop[2]:crop[3], :], dsize=(frame_size, frame_size))[..., 0],
                        (1, frame_size, frame_size, 1))
    return np.array(frame, dtype=np.uint8)

def normalize(sequence):
    return sequence / 255