import numpy as np
import cv2.ximgproc as xip

def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)

    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both Il to Ir and Ir to Il for later left-right consistency\

    h_l, w_l, ch_l = Il.shape
    census_Il = np.zeros((h_l, w_l, ch_l), dtype=np.uint8)
    
    for y in range(h_l):
        for x in range(w_l):
            for c in range(ch_l):
                census_code = 0
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if x + dx < 0 or x + dx >= w_l or y + dy < 0 or y + dy >= h_l:
                            continue
                        if Il[y + dy, x + dx, c] < Il[y, x, c]:
                            census_code |= 1
                        census_code <<= 1
                census_Il[y, x, c] = census_code >> 1

    h_r, w_r, ch_r = Ir.shape
    census_Ir = np.zeros((h_r, w_r, ch_r), dtype=np.uint8)
    
    for y in range(h_r):
        for x in range(w_r):
            for c in range(ch_r):
                census_code = 0
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if x + dx < 0 or x + dx >= w_r or y + dy < 0 or y + dy >= h_r:
                            continue
                        if Ir[y + dy, x + dx, c] < Ir[y, x, c]:
                            census_code |= 1
                        census_code <<= 1
                census_Ir[y, x, c] = census_code >> 1

    cost_l = np.zeros((max_disp+1, h, w), dtype=np.float32)
    cost_r = np.zeros((max_disp+1, h, w), dtype=np.float32)

    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)
    
    for d in range(max_disp+1):
        for x in range(w):
            x_l = max(x-d, 0)
            x_r = min(x+d, w-1)
            for y in range(h):
                cost_l[d, y, x] = np.sum(census_Il[y, x, :] != census_Ir[y, x_l, :]) 
                cost_r[d, y, x] = np.sum(census_Ir[y, x, :] != census_Il[y, x_r, :])
                
        sigma_s = 10
        sigma_c = 48
        cost_l[d, :, :] = xip.jointBilateralFilter(Il, cost_l[d, :, :], sigma_c, sigma_s, sigma_s)
        cost_r[d, :, :] = xip.jointBilateralFilter(Ir, cost_r[d, :, :], sigma_c, sigma_s, sigma_s)

    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    
    win_l = np.argmin(cost_l, axis=0)
    win_r = np.argmin(cost_r, axis=0)
    
    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    
    ## Left-right consistency check
    for y in range(h):
        for x in range(w):
            if x-win_l[y, x] < 0 or win_r[y, x-win_l[y, x]] != win_l[y, x]:
                win_l[y, x] = -1
                             
    ## Hole filling
    for y in range(h):
        for x in range(w):
            if win_l[y, x] == -1:
                l=0
                r=0
                while x + r <= w - 1 and win_l[y, x + r] == -1:
                    r += 1
                if x + r > w - 1:
                    Fr = max_disp 
                else:
                    Fr = win_l[y, x + r]
                
                while x - l >= 0 and win_l[y, x - l] == -1:
                    l += 1
                if x - l < 0:
                    Fl = max_disp 
                else:
                    Fl = win_l[y, x - l]

                win_l[y, x] = min(Fl, Fr)
                    
    ## Weighted median filtering
    labels = xip.weightedMedianFilter(Il.astype(np.uint8), win_l.astype(np.uint8), 20, 0.5)
    
    return labels.astype(np.uint8)