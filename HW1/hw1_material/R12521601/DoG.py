import numpy as np
import cv2

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        gaussian_images = []
        for i in range(self.num_octaves):
            for j in range(self.num_guassian_images_per_octave):
                if(i == 0):
                    if j == 0:
                        gaussian_images.append(image)
                    else:
                        gaussian_images.append(cv2.GaussianBlur(image, (0, 0), self.sigma**j))
                else:
                    if j == 0:
                        gaussian_images.append(cv2.resize(gaussian_images[4], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST))
                    else:
                        gaussian_images.append(cv2.GaussianBlur(gaussian_images[5], (0, 0), self.sigma**j))

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []
        for i in range(self.num_octaves):
            for j in range(self.num_DoG_images_per_octave):
                dog_images.append(cv2.subtract(gaussian_images[i*self.num_guassian_images_per_octave+j+1],
                                               gaussian_images[i*self.num_guassian_images_per_octave+j]))

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        key = []
        for i in range(self.num_octaves):
            for j in range(1,self.num_DoG_images_per_octave-1):
                up = dog_images[i*self.num_DoG_images_per_octave+j+1]
                center = dog_images[i*self.num_DoG_images_per_octave+j]
                down = dog_images[i*self.num_DoG_images_per_octave+j-1]
                for x in range(1, center.shape[0]-1):
                    for y in range(1, center.shape[1]-1):
                        center_patch = center[x-1:x+2, y-1:y+2]
                        up_patch = up[x-1:x+2, y-1:y+2]
                        down_patch = down[x-1:x+2, y-1:y+2]
                        center_point = center_patch[1,1]
                        all_patch = np.hstack((up_patch.flatten(), center_patch.flatten(), down_patch.flatten()))
                        all_patch_wo_center = np.delete(all_patch, 13)
                        is_max = np.all(all_patch_wo_center<center_point)
                        is_min = np.all(all_patch_wo_center>center_point)
                        if ((is_max) or (is_min)) and (abs(center_point) > self.threshold):
                                key.append([x*(i+1),y*(i+1)])
        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(key, axis=0)

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        return keypoints
    
