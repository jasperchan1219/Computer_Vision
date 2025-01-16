
import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        ### TODO ###
        # Calculus look up table for Gs and Gr
        Gs_x, Gs_y = np.meshgrid(np.arange(self.wndw_size) - self.pad_w, np.arange(self.wndw_size) - self.pad_w)
        Gs = np.exp(-np.divide((np.square(Gs_x)+ np.square(Gs_y)) , (2 * self.sigma_s**2)))
        Gr_LUT = np.exp(-np.divide((np.arange(256)/255) * (np.arange(256)/255) , (2 * self.sigma_r**2)))
        output = np.zeros(img.shape)
        for x in range(self.pad_w, img.shape[0] + self.pad_w):
            for y in range(self.pad_w, img.shape[1] + self.pad_w):
                x_l = x - self.pad_w
                x_r = x + self.pad_w + 1
                y_l = y - self.pad_w
                y_r = y + self.pad_w + 1
                # single-channel image
                if  len(guidance.shape) == 2:
                    Gr_difference = abs(padded_guidance[x_l : x_r, y_l : y_r] - padded_guidance[x, y])
                    GsGr = Gs * Gr_LUT[Gr_difference]
                # color image
                elif len(guidance.shape) == 3:
                    Gr_difference_r = abs(padded_guidance[x_l : x_r, y_l : y_r, 0] - padded_guidance[x, y, 0])
                    Gr_difference_g = abs(padded_guidance[x_l : x_r, y_l : y_r, 1] - padded_guidance[x, y, 1])
                    Gr_difference_b = abs(padded_guidance[x_l : x_r, y_l : y_r, 2] - padded_guidance[x, y, 2])
                    # Multiply the Gr_LUT(exponential) for each channel to get the numerator
                    GsGr = Gs * Gr_LUT[Gr_difference_r] * Gr_LUT[Gr_difference_g] * Gr_LUT[Gr_difference_b]
                Denom = np.sum(GsGr)
                padded_img_r = padded_img[x_l : x_r, y_l : y_r, 0]
                padded_img_g = padded_img[x_l : x_r, y_l : y_r, 1]
                padded_img_b = padded_img[x_l : x_r, y_l : y_r, 2]
                # Place the result in the output image starting from (x-self.pad_w, y-self.pad_w)
                output[x_l, y_l, 0] = np.sum(GsGr * padded_img_r) / Denom
                output[x_l, y_l, 1] = np.sum(GsGr * padded_img_g) / Denom
                output[x_l, y_l, 2] = np.sum(GsGr * padded_img_b) / Denom

        return np.clip(output, 0, 255).astype(np.uint8)
