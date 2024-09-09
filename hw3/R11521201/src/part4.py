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
    # 全景圖高度為所有影像高度之max；寬度為所有影像寬之和
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0] 
    last_best_H = np.eye(3)
    out = None
    
    # for all images to be stitched:
    for idx in tqdm(range(len(imgs)-1)):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # TODO: 1.feature detection & matching
        # Feature detection & matching
        # Use opencv built-in ORB detector for keypoint matching
        orb = cv2.ORB_create()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        kp1, des1 = orb.detectAndCompute(im1, None)
        kp2, des2 = orb.detectAndCompute(im2, None)
        matches = bf.match(des1, des2) # 透過距離測試篩選好的匹配點
        
        # get the index of match descriptors
        t_idx = [match.trainIdx for match in matches]
        q_idx = [match.queryIdx for match in matches]

        # use the index to find the corresponding key points
        src_pts = np.abs([kp2[idx].pt for idx in t_idx])
        dst_pts = np.abs([kp1[idx].pt for idx in q_idx])

        # TODO: 2. apply RANSAC to choose best H
        # 用RANSAC解決Outlier
        max_Inliers = 0
        best_H = np.eye(3)
        
        num_iteration = 3000
        num_kps_for_H = 4
        threshold = 0.2
        
        for _ in range(num_iteration): 
            rand_idx = random.sample(range(len(src_pts)), num_kps_for_H)
            p1, p2 = src_pts[rand_idx], dst_pts[rand_idx]
            H = solve_homography(p1, p2)
            # use H to get predicted coordinates
            U = np.concatenate((src_pts.T, np.ones((1,src_pts.shape[0]))), axis=0)
            pred = np.dot(H, U)
            pred = (pred/pred[2]).T[:,:2]

            # 計算預測座標與目標座標之歐式距離
            # 小於thres的內點數
            distance = pred-dst_pts
            error = np.linalg.norm(distance, axis=1)

            inliers = (error < threshold).sum() 

            if inliers > max_Inliers :
                best_H = H.copy()
                # update maxInliers
                max_Inliers = inliers
        

        # TODO: 3. chain the homographies
        last_best_H = np.dot(best_H, last_best_H)
        
        # TODO: 4. apply warping
        dst = warping(im2, dst, last_best_H, 0, h_max, 0, w_max, 'b')

        # apply alpha blending
        im1_resized = cv2.resize(im1, (dst.shape[1], dst.shape[0]))
        im2_resized = cv2.resize(im2, (dst.shape[1], dst.shape[0]))

        dst_float = dst.astype(np.float32)
        alpha = np.where(dst_float == 0, 0, 1)
        dst = (alpha * im1_resized + (1 - alpha) * im2_resized).astype(np.uint8)

    return dst

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)