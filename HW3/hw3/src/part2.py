import numpy as np
import cv2
from cv2 import aruco
from tqdm import tqdm
from utils import solve_homography, warping


def planarAR(REF_IMAGE_PATH, VIDEO_PATH):
    """
    Reuse the previously written function "solve_homography" and "warping" to implement this task
    :param REF_IMAGE_PATH: path/to/reference/image
    :param VIDEO_PATH: path/to/input/seq0.avi
    """
    video = cv2.VideoCapture(VIDEO_PATH)
    ref_image = cv2.imread(REF_IMAGE_PATH)
    h, w, c = ref_image.shape
    film_h, film_w = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    film_fps = video.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    videowriter = cv2.VideoWriter("output2.avi", fourcc, film_fps, (film_w, film_h))
    arucoDict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    arucoParameters = aruco.DetectorParameters_create()
    ref_corns = np.array([[0, 0], [w, 0], [w, h], [0, h]])

    # TODO: find homography per frame and apply backward warp
    pbar = tqdm(total = 353)
    while (video.isOpened()):
        ret, frame = video.read()
        if ret:  ## check whethere the frame is legal, i.e., there still exists a frame
            # TODO: 1.find corners with aruco
            # function call to aruco.detectMarkers()
            (corners, ids, rejected) = aruco.detectMarkers(frame, arucoDict, parameters=arucoParameters)
            
            # TODO: 2.find homograpy
            # function call to solve_homography()
            if ids is not None:
                ids = ids.flatten()
                if 0 in ids:
                    idx = np.where(ids == 0)[0][0]
                    frame_corns = corners[idx].reshape(4, 2)
                    H = solve_homography(ref_corns, frame_corns)
            # TODO: 3.apply backward warp
            # function call to warping()
            x_min = np.min(frame_corns[:, 0]).astype(np.int64)
            x_max = np.max(frame_corns[:, 0]).astype(np.int64)
            y_min = np.min(frame_corns[:, 1]).astype(np.int64)
            y_max = np.max(frame_corns[:, 1]).astype(np.int64)
            frame = warping(ref_image, frame, H, y_min, y_max, x_min, x_max, direction='b')
            

            videowriter.write(frame)
            pbar.update(1)

        else:
            break

    pbar.close()
    video.release()
    videowriter.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    # ================== Part 2: Marker-based planar AR========================
    VIDEO_PATH = '../resource/seq0.mp4'
    # TODO: you can change the reference image to whatever you want
    REF_IMAGE_PATH = '../resource/arknights.png' 
    planarAR(REF_IMAGE_PATH, VIDEO_PATH)