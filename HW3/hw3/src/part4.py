import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None

    # for all images to be stitched:
    for idx in tqdm(range(len(imgs)-1)):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # TODO: 1.feature detection & matching
        # Importing SIFT detector
        sift = cv2.SIFT_create()

        # Detect keypoints and compute descriptors for both images
        kp1, des1 = sift.detectAndCompute(im1, None)
        kp2, des2 = sift.detectAndCompute(im2, None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        U = []
        V = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                U.append(np.float32(kp1[m.queryIdx].pt))
                V.append(np.float32(kp2[m.trainIdx].pt))
        U = np.array(U)
        V = np.array(V)
        # TODO: 2. apply RANSAC to choose best H
        num_iterations = 10000
        threshold = 3
        max_inline = 20
        best_H = np.eye(3)

        ## iterate
        for _ in range(0, num_iterations):
            random_indices = np.random.choice(len(U), size=4, replace=False)
            u_rand = U[random_indices]
            v_rand = V[random_indices]
            H = solve_homography(v_rand, u_rand)
            M = np.concatenate([np.transpose(V), np.ones((1, len(V)))], axis=0)
            W = np.concatenate([np.transpose(U), np.ones((1, len(U)))], axis=0)
            U_hat = np.dot(H, M)
            U_hat = U_hat / U_hat[2, :]
            dist = np.linalg.norm(W - U_hat, axis=0)
            inliers = dist < threshold
            num_inliers = np.count_nonzero(inliers)
            if num_inliers > max_inline:
                max_inline = num_inliers
                best_H = H
            
                
        # TODO: 3. chain the homographies
        last_best_H = np.dot(last_best_H, best_H)
        
        # TODO: 4. apply warping
        out = warping(im2, dst, last_best_H, 0, h_max, 0, w_max, direction='b')
        

    return out 

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)
    