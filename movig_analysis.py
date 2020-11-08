"""Module with video-based analysis tools

Authors:
    Sapir Avrahami
"""

import numpy as np
import cv2

def calculate_delta(frame_1: np.array, frame_2: np.array, threshold: int = 50):
    """This function calculates moving rate between given frames.
    Moving rate is the difference between frame_1 and frame_2

    Parameters
    ----------
    frame_1 : np.array
        The first frame
    frame_2 : np.array
        The second frame
    threshold : int
        Parameter for moving detection

    Returns
    -------
    double
        moving rate
    """
    # Convert to B&W image
    curr_frame = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
    delta_frame = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)

    # Blur cropped B&W image
    kernel_size = np.max(curr_frame.shape)//200  * 2 + 1
    print(kernel_size)
    curr_frame = cv2.GaussianBlur(curr_frame, (kernel_size, kernel_size), 0)
    delta_frame = cv2.GaussianBlur(delta_frame, (kernel_size, kernel_size), 0)


    frame_delta = cv2.absdiff(curr_frame, delta_frame)
    thresh = cv2.threshold(frame_delta, threshold, 255, cv2.THRESH_BINARY)[1]

    frame_area = curr_frame.shape[0] * curr_frame.shape[1]
    moving_rate_f = np.sum(thresh)/255 / frame_area
    return moving_rate_f/2

def analyse_moving(video:Video, coordinates:np.array, delta:int) -> np.array:
    """
    This function calculates the moving rate in the given Video object.
    Moving rate at frame X is the difference between BB area at frame X
    and BB area at frame X-delta.

    Parameters
    ----------
    video: A Video object -- blyzer/common/video.py
    coordinates: Nx4 array with BB coordinates [[xmin, ymin, xmax, ymax], [...]],
    Coordinates can be None(np.nan) if object is not existed on frame
    delta: Distance in frames for moving calculation

    Returns
    -------
    numpy.array with the moving rate of the entire video
    """

    moving_rate = np.full(video.get_frames_count(), np.nan)

    for i in range(video.get_frames_count()):

        if i - delta < 0:
            continue

        # if the object does not exist on either frames, continue.
        if np.isnan(coordinates[i]).any():
            continue
        else:
            # Get frames
            curr_frame = video.get_frame(i)
            delta_frame = video.get_frame(i - delta)

            # Get coordinates and make them int
            ar_curr = np.int64(coordinates[i])

            # Crop frames with given coordinates
            curr_frame = curr_frame[ar_curr[1]: ar_curr[3], ar_curr[0]: ar_curr[2]]
            delta_frame = delta_frame[ar_curr[1]: ar_curr[3], ar_curr[0]: ar_curr[2]]

            moving_rate_f = calculate_delta(curr_frame, delta_frame)

            moving_rate[i] = moving_rate_f

    return moving_rate

if __name__ == '__main__':
    #cap = cv2.VideoCapture('example.mp4')
    print("hi")

