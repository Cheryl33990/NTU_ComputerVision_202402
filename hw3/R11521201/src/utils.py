import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    
    中文：計算單硬性矩陣(Homography Matrix)
    u跟v都是Nx2的矩陣，代表有N個點，每個點都由(x,y)組成。
    u: 原圖點座標
    v: 轉換後對應的座標
    """
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A (參考Lec09 p.12 DLT Algorithm)
    # 公式參照p.35 Importance of Normalization
    A = []
    for i in range(N):
        # 原點
        x, y = u[i]
        # 投影點(Projective)
        x_p, y_p = v[i]
        A.append([x, y, 1, 0, 0, 0, -x*x_p, -y*x_p, -x_p])
        A.append([0, 0, 0, -x, -y, -1, x*y_p, y*y_p, y_p])
    A = np.array(A)

    # TODO: 2.solve H with A
    # 求解參照p.12 (iii) Obtain SVD of A. 
    # U, S, V = np.linalg.svd(A)
    # U: 左奇異向量 / S: 由大到小排序的奇異值 / V: 右奇異向量
    _, _, V = np.linalg.svd(A)

    # 解H只需要用到V的最後一行
    H = V[-1].reshape(3,3) 

    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, ch = src.shape # height & width of source img
    h_dst, w_dst, ch = dst.shape # height & width of destination output img
    H_inv = np.linalg.inv(H) # calculate the inverse of H

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    # 生成網格：透過meshgrid生成座標網格
    xc, yc = np.meshgrid(np.arange(xmin, xmax, 1), np.arange(ymin, ymax, 1)) # default sparse = False
    # 指定範圍內的pixel總數
    xrow = xc.reshape(( 1,(xmax-xmin)*(ymax-ymin) ))
    yrow = yc.reshape(( 1,(xmax-xmin)*(ymax-ymin) ))
    # 為構成齊次矩陣 -> 需要一個全為1的row
    all_onerow =  np.ones(( 1,(xmax-xmin)*(ymax-ymin) ))

    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    # 建立齊次座標：生成Nx3三維的(x, y, 1)矩陣M
    # 縱向拼接 -> axis = 0
    M = np.concatenate((xrow, yrow, all_onerow), axis = 0)
    
    if direction == 'b': # back warping
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        # back warping: destination * H_inv
        V = np.dot(H_inv, M)
        Vx, Vy, _ = V/V[2] # 進行normalize (將第三行都賦值1，故為"_")
        Vx = Vx.reshape(ymax-ymin, xmax-xmin)
        Vy = Vy.reshape(ymax-ymin, xmax-xmin)
        
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        h_src, w_src, ch = src.shape
        # Vx < w_src-1: 不超過原圖右邊介；0 <= Vx: 確保不會是負數
        # Vy < h_src-1: 不超過原圖下邊界；0 <= Vy: 確保不會是負數
        # index從0開始
        mask = (((Vx<w_src-1)&(0<=Vx))&((Vy<h_src-1)&(0<=Vy))) 
        
        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        mask_Vx = Vx[mask]
        mask_Vy = Vy[mask]
        
        # 處理sub-pixel location
        # Step 1: Nearest neighbor
        mask_Vxint = mask_Vx.astype(int)
        mask_Vyint = mask_Vy.astype(int)
        # Step 2: Bilinear interpolation
        dX = (mask_Vx - mask_Vxint).reshape((-1,1))
        dY = (mask_Vy - mask_Vyint).reshape((-1,1))
        p = np.zeros((h_src, w_src, ch))
        # 找尋鄰近的4個點
        # 左上角、左下角、右上角、右下角
        p[mask_Vyint, mask_Vxint, :] += (1-dY)*(1-dX)*src[mask_Vyint, mask_Vxint, :]
        p[mask_Vyint+1, mask_Vxint, :] += dY*(1-dX)*src[mask_Vyint+1, mask_Vxint, :]
        p[mask_Vyint, mask_Vxint+1, :] += (1-dY)*dX*src[mask_Vyint, mask_Vxint+1, :]
        p[mask_Vyint+1, mask_Vxint+1, :] += dY*dX*src[mask_Vyint+1, mask_Vxint+1, :]

        # TODO: 6. assign to destination image with proper masking
        dst[ymin:ymax,xmin:xmax][mask] = p[mask_Vyint,mask_Vxint]
                
        pass

    elif direction == 'f': # forward warping
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        # forward warping: source * H # 較為直觀
        V = np.dot(H,M)
        V = (V/V[2]).astype(int)
        Vx, Vy, _ = V
        Vx = Vx.reshape(ymax-ymin, xmax-xmin)
        Vy = Vy.reshape(ymax-ymin, xmax-xmin)
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        mask = (((Vx<w_dst-1)&(0<=Vx))&((Vy<h_dst-1)&(0<=Vy)))
        
        # TODO: 5.filter the valid coordinates using previous obtained mask
        mask_Vx = Vx[mask]
        mask_Vy = Vy[mask]

        # TODO: 6. assign to destination image using advanced array indicing
        dst[mask_Vy, mask_Vx, :] = src[mask]
        
        pass

    return dst 
