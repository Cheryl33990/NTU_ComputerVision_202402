import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        # Pad the input and guidance image
        # Pad the edges of the image to increase its size. # i.e. hgfedcba|abcdefgh|hgfedcba
        BORDER_TYPE = cv2.BORDER_REFLECT
        # cv2.copyMakeBorder(img, top, bottom, left, right, borderType)
        # convert the image to np.int32 is to avoid potiential numerical overflow and to handle negative values in subtraction operations.
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, 
                                        self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, 
                                             self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        
        ### TODO ###
        """
         Define the Gaussian filter
          - Look-up table (LUT) is defined by Gaussian function
          - The constant coefficient could be ignored -> Only take exponential part  
        """
        # setup a look-up table for spatial kernel
        LUT_s = np.exp(-0.5*(np.arange(self.pad_w+1)**2)/self.sigma_s**2)
        # setup a look-up table for range kernel
        LUT_r = np.exp(-0.5*(np.arange(256)/255)**2/self.sigma_r**2)
        
        # compute the weight of range kernel by rolling the whole image
        # compute the weight of range kernel by rolling the whole image
        """
         Apply JBF
          - weight_sum and result are initialized  as zero matrices with the same shape as the padded image.
          - iterate over each sptial offset from -self.pad_w to self.pad_w
          - dT: the weight array, reflecting the weight of intensity differences betwenn each pizel and its neighbors
            roll the padded_guidence image vertically (along axis 0) by y units,
            and then horizontally (along axis 1) by x units.
          - r_w 
            dT.ndim == 2 -> grayscale    (width & height)
            else (>2)    -> color image  (use np.prod to consolidate color channel information)
          - s_w
            retrieving weights corresponding to the current pixel offsets x and y from the lookup table LUT_s, and then multiplying them
            absolute values -> spatial relationships are symmetric

        """
        weight_sum, result = np.zeros(padded_img.shape), np.zeros(padded_img.shape)
        for x in range(-self.pad_w, self.pad_w+1):
            for y in range(-self.pad_w, self.pad_w+1):
                # dT: intensity difference; 
                dT = LUT_r[np.abs(np.roll(padded_guidance, [y,x], axis=[0,1])-padded_guidance)]
                r_w = dT if dT.ndim==2 else np.prod(dT,axis=2) # range kernel weight
                s_w = LUT_s[np.abs(x)]*LUT_s[np.abs(y)]        # spatial kernel
                t_w = s_w*r_w
                padded_img_roll = np.roll(padded_img, [y,x], axis=[0,1])
                for channel in range(padded_img.ndim):
                    result[:,:,channel] += padded_img_roll[:,:,channel]*t_w
                    weight_sum[:,:,channel] += t_w
        output = (result/weight_sum)[self.pad_w:-self.pad_w, self.pad_w:-self.pad_w,:]
        
        # Ouput image should be in format of uint8
        return np.clip(output, 0, 255).astype(np.uint8)