import numpy as np
import cv2.ximgproc as xip
import cv2

def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    #labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32) # Image_L, transform to np.
    Ir = Ir.astype(np.float32) # Image_R, transform to np.
    pad_num = 2 # To keep the image size 
    
    #==================================================================================================
    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both Il to Ir and Ir to Il for later left-right consistency
    
    # For Image_L
    # print("Shape of Image_Left:", Il.shape)     # (375, 450, 3)
    Il_pad = np.pad(Il,((pad_num,pad_num),(pad_num,pad_num),(0,0)))
    # print("Shape of Image_Left_pad:", Il_pad.shape) # (379, 454, 3)
    
    # For Image_R
    # print("Shape of Image_Right:", Ir.shape)     # (375, 450, 3)
    Ir_pad = np.pad(Ir,((pad_num,pad_num),(pad_num,pad_num),(0,0)))
    # print("Shape of Image_Right_pad:", Ir_pad.shape) # (379, 454, 3)
    
    # Cost Computation
    cost_box = np.zeros((h, w, ch, max_disp), dtype=np.float32)
    
    position = [] # To store the position of census pixel
    # range(0, 5) -> 0, 1, 2, 3, 4 -> 5x5 window
    for i in range(0,1+pad_num*2):
        for j in range(0,1+pad_num*2):
            if i!=pad_num or j!=pad_num:
                position.append((i,j)) # 5x5
    
    # Operate Exclusive OR (XOR)
    # If input A  = input B, the output of XOR = 0
    # If input A != input B, the output of XOR = 1
    Il_census = np.zeros((h, w, ch,len(position)), dtype=np.bool)
    for i in range(len(position)):
        pos_row,pos_col = position[i]
        Il_census[:,:,:,i] = Il<Il_pad[pos_row:pos_row+h,pos_col:pos_col+w,:]
    # print("Shape of Image_Left_census:",Il_census.shape)

    Ir_census = np.zeros((h, w, ch,len(position)), dtype=np.bool)
    for i in range(len(position)):
        pos_row,pos_col = position[i]
        Ir_census[:,:,:,i] = Ir<Ir_pad[pos_row:pos_row+h,pos_col:pos_col+w,:]
    # print("Shape of Image_Right_census:",Ir_census.shape)

    for disp in range(max_disp):
        temp_censuscost = np.logical_xor(Il_census[:,disp:,:,:],Ir_census[:,:w-disp,:,:])
        temp_censuscost = np.sum(temp_censuscost,axis=3)
        #print(temp_censuscost.shape)
        
        # axis=0 -> 垂直方向不填充
        # axis=1 -> 水平方向，左側增加寬度為disp的填充，右邊不增加。 -> 將左右視圖對齊
        # axis=2 -> Channel不填充
        temp_censuscost = np.pad(temp_censuscost,((0,0),(disp,0),(0,0),),mode='edge') # edge padding
        #print(temp_censuscost.shape)
        cost_box[:,:,:,disp] = temp_censuscost

    #==================================================================================================
    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)
    # 用JBF進行濾波 # 測試了不同組
    JBF_s = 10 # spatial_sigma
    JBF_d = -1 # diameter_sigma
    JBF_c = 5  # color_sigma
    
    # Image_Left為Guide image
    for disp in range(max_disp):
        cost_box[:,:,:,disp] = xip.jointBilateralFilter(Il, cost_box[:,:,:,disp], JBF_d, JBF_c, JBF_s)
    #print(cost_box.shape)
    cost_box = np.sum(cost_box,axis=2) # channel sum
    #print(cost_box.shape)

    #==================================================================================================
    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    # np.argmin: 找到最小的視差 -> 匹配度最高
    left_disparity = np.argmin(cost_box,axis=2).astype(np.uint8)
    #print(labels.shape)

    #==================================================================================================
    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    
    # compute disparity map for right image
    right_cost_box = np.zeros((h, w, ch, max_disp), dtype=np.float32)
    for disp in range(max_disp):
        temp_censuscost = np.logical_xor(Il_census[:,disp:,:,:],Ir_census[:,:w-disp,:,:])
        temp_censuscost = np.sum(temp_censuscost,axis=3)
        temp_censuscost = np.pad(temp_censuscost,((0,0),(0,disp),(0,0),),mode='edge')
        right_cost_box[:,:,:,disp] = temp_censuscost
    for disp in range(max_disp):
        right_cost_box[:,:,:,disp] = xip.jointBilateralFilter(Ir, right_cost_box[:,:,:,disp], JBF_d, JBF_c, JBF_s)
    right_cost_box = np.sum(right_cost_box,axis=2)
    right_disparity = np.argmin(right_cost_box,axis=2).astype(np.uint8)
    
    # check consistency
    w_value, h_value = np.meshgrid(range(w), range(h))
    #print(hv.shape)
    #print(wv.shape)
    R_idx = np.stack((w_value, h_value))
    #print(left_disparity.shape)
    #print(R_idx[0,:,:].shape)
    
    # 檢查左右視差圖是否匹配
    # R_idx[0,:,:]為橫坐標；# R_idx[1,:,:]為縱坐標
    R_idx[0,:,:] = R_idx[0,:,:] - left_disparity
    R_idx = np.reshape(R_idx,(2,h*w))
    #print(R_idx.shape)
    DR_x_DL = right_disparity[R_idx[1,:],R_idx[0,:]]
    DR_x_DL = np.reshape(DR_x_DL,(h,w))
    #print(DR_x_DL.dtype)
    # 相等為True；步相等為False
    valid = left_disparity==DR_x_DL
    #print(valid.shape)
    
    # hole filling
    valid = np.pad(valid,((0,0),(1,1)),constant_values=True)
    left_disparity = np.pad(left_disparity,((0,0),(1,1)),constant_values=max_disp)
    #print(valid.shape)
    for iw in range(1,w+1):
        for ih in range(h):
            if valid[ih,iw] == False:
                # 向左尋找第一個有效象素
                find_x = iw
                while valid[ih,find_x]==False:
                    find_x = find_x - 1
                FL = left_disparity[ih,find_x]
                
                # 向右尋找第一個有效象素
                find_x = iw
                while valid[ih,find_x]==False:
                    find_x = find_x + 1
                FR = left_disparity[ih,find_x]
                
                # 選擇左右兩側有效象素插值小者作為當前象素的是插值
                left_disparity[ih,iw] = min(FL,FR)
    
    # 移除邊界填充
    left_disparity = left_disparity[:,1:-1]
    
    # Weighted median filtering
    Il_gray = cv2.cvtColor(Il,cv2.COLOR_BGR2GRAY).astype(np.uint8)
    labels = xip.weightedMedianFilter(Il_gray,left_disparity,r = 10,sigma = 10)


    return labels.astype(np.uint8)
    