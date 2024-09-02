import numpy as np
import cv2
import time

class Difference_of_Gaussian(object):
    """
    - threshold: To filter keypoints in subsequent processing.
    - sigma: The standard deviation parameter for Gaussian blurring, controlling the width of the Gaussian function.
    - num_octaves: Each octave corresponds to a halving of the image resolution. 
      In this class, two octaves (num_octaves=2) are set, meaning that images at two different resolutions will be processed.
    - num_DoG_images_per_octave:  To set the number of Difference of Gaussian (DoG) images to be generated per octave.
    - num_guassian_images_per_octave = num_DoG_images_per_octave + 1
      Since each DoG image is produced by subtracting two adjacent Gaussian-blurred images, 
      the number of Gaussian-blurred images required per octave is one more than the number of DoG images.
    """
    def __init__(self, threshold):
        self.threshold = threshold 
        self.sigma = 2**(1/4)      
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        start_time = time.time()
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        #======================================================================================================================================
        # Process the octaves separately
        # The first image in the octave is the original image itself.
        # The following images are created by applying Gaussian blur with progressively increasing values of sigma.
        first_octave = [image] + [cv2.GaussianBlur(image, (0, 0), self.sigma**i) for i in range(1, self.num_guassian_images_per_octave)]
        # Down-sample the image
        # The last image from the first octave, which is the most blurred, is down-sampled to half its original width and height.
        DSImage = cv2.resize(first_octave[-1], 
                             (image.shape[1]//2, image.shape[0]//2), 
                             interpolation = cv2.INTER_NEAREST)
        # Filter down-sampled images in the next octave
        # The second octave is processed in a similar way to the first, but starting with the down-sampled image from the end of the first octave.
        second_octave = [DSImage] + [cv2.GaussianBlur(DSImage, (0, 0), self.sigma**i) for i in range(1, self.num_guassian_images_per_octave)]
        # Combine two octaves
        # This list contains two sub-lists(octaves) 
        gaussian_images = [first_octave, second_octave]

        #======================================================================================================================================
        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []
        # Iterate through each octave for subtraction
        for i in range(self.num_octaves):
            GImage = gaussian_images[i] # 
            dog_image = []
            for j in range(self.num_DoG_images_per_octave): # num_DOG_images_per_octave = 4
                dog = cv2.subtract(GImage[j], GImage[j+1])
                dog_image.append(dog)
                #save DoG images to disk
                Max = max(dog.flatten())
                Min = min(dog.flatten())
                norm = (dog-Min)*255/(Max-Min) # To normalize
                cv2.imwrite(f'testdata/DoG{i+1}-{j+1}.png', norm)
            dog_images.append(dog_image)

        #======================================================================================================================================
        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        # Keep local extremum as a keypoint
        keypoints = []
        # iterate through each ocvate (n=2 here)
        for i in range(self.num_octaves):
            # transform an octave into a 3-dimensional array
            dogs = np.array(dog_images[i])
            height, width = dogs[i].shape
            # examine every 3*3 cube for local extremum
            # iterate through Dog image number 1 and 2 (other than 0 and 3)
            for dog in range(1, self.num_DoG_images_per_octave-1):
                #iterate through every pixel
                for x in range(1, width-2):
                    for y in range(1, height-2):
                        pixel = dogs[dog,y,x]
                        cube = dogs[dog-1:dog+2, y-1:y+2, x-1:x+2]
                        # to check if it's local extremum
                        if (np.absolute(pixel) > self.threshold) and ((pixel >= cube).all() or (pixel <= cube).all()):
                            keypoints.append([y*2, x*2] if i else [y, x])

        #======================================================================================================================================
        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(np.array(keypoints), axis = 0)

        end_time = time.time()
        print(f"Execution Time: {end_time - start_time} seconds")


        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        return keypoints
