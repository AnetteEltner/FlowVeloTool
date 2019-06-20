# Copyright (c) 2019, Anette Eltner
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met: 
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer. 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution. 
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import sys, math
import numpy as np
import pylab as plt
import pandas as pd

import cv2


class pointImg:
    
    def __init__(self):
        
        self.x = 0
        self.y = 0


class pointAdjusted:
    
    def __init__(self):
        
        self.x = 0
        self.y = 0
        self.s0 = 0
        self.usedObserv = 0
        

def lsm_matching(patch, lsm_search, pointAdjusted, lsm_buffer, thresh=0.001):
# source code from Ellen Schwalbe rewritten for Python
#x1, y1 of patch (template); x2, y2 of search area (little bit bigger)
    add_val = 1 
        
    px = patch.shape[1]
    py = patch.shape[0]
    n = px * py
    
    dif_patch_lsm_size_x = (lsm_search.shape[1] - patch.shape[1]) / 2
    dif_patch_lsm_size_y = (lsm_search.shape[0] - patch.shape[0]) / 2
    
    p_shift_ini = pointImg()
    p_shift_ini.x = np.int(lsm_search.shape[1]/2) 
    p_shift_ini.y =  np.int(lsm_search.shape[0]/2) 
    
    #approximation
    U = np.asarray([np.int(lsm_search.shape[1]/2), np.int(lsm_search.shape[0]/2)], dtype=np.float)
#     #tx, ty, alpha
#     U = np.asarray([np.int(lsm_search.shape[1]/2), np.int(lsm_search.shape[0]/2),
#                     np.float(0)], dtype=np.float)
     
    A = np.zeros((n, U.shape[0]))
    l = np.zeros((n, 1))
    
    for i in range(100):    #number of maximum iterations
        
        lsm_search = contrastAdaption(patch, lsm_search)
        lsm_search = brightnessAdaption(patch, lsm_search)
    
        #calculate gradient at corresponding (adjusting) position U
        count = 0
        img_test_search = np.zeros((lsm_search.shape[0], lsm_search.shape[1]))
        img_test_patch = np.zeros((patch.shape[0], patch.shape[1]))
        for x1 in range(px):
            for y1 in range(py):                 
                if (U[0]-p_shift_ini.x < -(lsm_buffer+dif_patch_lsm_size_x) or U[0]-p_shift_ini.x > lsm_search.shape[1]+lsm_buffer-1 or 
                    U[1]-p_shift_ini.y < -(lsm_buffer+dif_patch_lsm_size_y) or U[1]-p_shift_ini.y > lsm_search.shape[0]+lsm_buffer-1):
                    print(count, i)
                    print('patch out of search area')
                    return 1/0
                
                x2 = x1 + U[0]-p_shift_ini.x + dif_patch_lsm_size_x #shift to coordinate system of lsm_search 
                y2 = y1 + U[1]-p_shift_ini.y + dif_patch_lsm_size_y
#                 #rotation and translation
#                 x2 = x1 * np.cos(U[2]) - y1 * np.sin(U[2]) + U[0]-p_shift_ini.x + dif_patch_lsm_size_x 
#                 y2 = x1 * np.sin(U[2]) + y1 * np.cos(U[2]) + U[1]-p_shift_ini.y + dif_patch_lsm_size_y
                
                g1 = patch[int(y1),int(x1)]
                g2 = interopolateGreyvalue(lsm_search, x2, y2)
                
                img_test_patch[y1,x1] = g1
                img_test_search[int(y2),int(x2)] = g2
                
                plt.ion()
                
                #translation x
                gx1 = interopolateGreyvalue(lsm_search, x2-add_val, y2)
                gx2 = interopolateGreyvalue(lsm_search, x2+add_val, y2)
                
                #translation y
                gy1 = interopolateGreyvalue(lsm_search, x2, y2-add_val)
                gy2 = interopolateGreyvalue(lsm_search, x2, y2+add_val)
                
#                 #rotation
#                 galpha1 = interopolateGreyvalue(lsm_search, x2, y2, 1)
#                 galpha2 = interopolateGreyvalue(lsm_search, x2, y2, -1)
                
                plt.close('all')
    
                if g1 < 0 or g2 < 0 or gx1 < 0 or gy1 < 0 or gx2 < 0 or gy2 < 0:
                    print(count, i)
                    print('error during gradient calculation')
                    return 1/0
                
                l[count] = g2-g1
                
                #translation
                A[count, 0] = gx1-gx2
                A[count, 1] = gy1-gy2
#                 #rotation
#                 A[count, 2] = galpha1-galpha2
                
                count = count + 1
        
        #perform adjustment with gradients
        dx_lsm, s0 = adjustmentGradient(A, l)         
         
        #adds corrections to the values of unknowns
        SUM = 0
        for j in range(U.shape[0]):
            U[j] = U[j] + dx_lsm[j]
            SUM = SUM + np.abs(dx_lsm[j])
#         print SUM, U, dx_lsm
        
        #stops the iteration if sum of additions is very small
        if (SUM < thresh):             
            pointAdjusted.x = U[0]
            pointAdjusted.y = U[1]
            pointAdjusted.s0 = s0
            pointAdjusted.usedObserv = n
            
            return pointAdjusted

    print('adjustment not converging')
    return -1


def adjustmentGradient(A, l):
#A... A-matrix
#l... observation vector l

    A = np.matrix(A)
    l = np.matrix(l)
    
    #adjustment
    N = A.T * A
    Q = np.linalg.inv(N)    #N_inv
    L = A.T * l    
    dx = Q * L    
    v = A * dx - l  #improvements
    
    #error calculation
    s0 = np.sqrt((v.T * v) / (A.shape[0] - A.shape[1])) # sigma-0
    
    #error of unkowns
    error_unknowns = np.zeros((A.shape[1],1))    
    for j in range(error_unknowns.shape[0]):
        error_unknowns[j] = s0 * np.sqrt(Q[j,j])
     
    return dx, s0
    

def interopolateGreyvalue(img, x, y, rot_angle=0): #bilinear interpolation
    x_int = int(x)
    y_int = int(y)
        
    dx = float(x - x_int)
    dy = float(y - y_int)
    
    if y_int < 0 or x_int < 0 or y_int + 1 >= img.shape[0] or x_int + 1 >= img.shape[1]:
        return -1

    if not rot_angle == 0:
        img = rotate_about_center(img, rot_angle)
    
    I = img[y_int, x_int]
    Ixp = img[y_int, x_int+1]
    Iyp = img[y_int+1, x_int]
    Ixyp = img[y_int+1, x_int+1]
    
    g = I * (1-dx) * (1+dy) + Ixp * dx * (1+dy) + Iyp * (1-dx) * dy + Ixyp * dx *dy
    
    return g
    

def contrastAdaption(I1, I2):
#I2 is larger image    
    minI1 = np.float(np.nanmin(I1))
    maxI1 = np.float(np.nanmax(I1))
    minI2 = np.float(np.nanmin(I2))
    maxI2 = np.float(np.nanmax(I2))
       
    #adapt contrast
    I2_adapt = ((maxI1-minI1)/(maxI2-minI2)) * (I2 - np.ones((I2.shape[0], I2.shape[1])) * minI2) + np.ones((I2.shape[0], I2.shape[1])) * minI1
    
    I2_adapt[I2_adapt < 0] = 0
    I2_adapt[I2_adapt > 255] = 255
    
    I2_adapt = np.asarray(I2_adapt, dtype=np.int)
    
    return I2_adapt


def brightnessAdaption(I1, I2):
#I2 is larger image    
    s1 = cv2.mean(I1)
    s2 = cv2.mean(I2)
    
    I2_adapt = I2 + np.ones((I2.shape[0], I2.shape[1])) * s1[0] - s2[0]
    
    I2_adapt[I2_adapt < 0] = 0
    I2_adapt[I2_adapt > 255] = 255
    
    I2_adapt = np.asarray(I2_adapt, dtype=np.int)
    
    return I2_adapt


def rotate_about_center(src, angle, scale=1.):
    src = src.astype(np.uint8)
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    
    #calculate image width and height
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    
    # get rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    
    #calculate move from old center to new center combined with rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    
    #move only affects translation, update translation part of transform 
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))))


def getTemplate(img, tmplPtCoo, template_width=10, template_height=10, forTracking=False):
# consideration that row is y and column is x   
# careful that template extents even to symmetric size around point of interest 

    if template_width > 0:
        template_width_for_cut_left = template_width/2
        template_width_for_cut_right = template_width/2 + 1
    else:
        print('missing template width assignment')        
    if template_height > 0:
        template_height_for_cut_lower = template_height/2
        template_height_for_cut_upper = template_height/2 + 1
    else:
        print('missing template height assignment')    
    
    cut_anchor_x = tmplPtCoo[0] - template_width_for_cut_left
    cut_anchor_y = tmplPtCoo[1] - template_height_for_cut_lower    
    
    #consideration of reaching of image boarders (cutting of templates)
    if tmplPtCoo[1] + template_height_for_cut_upper > img.shape[0]:
        if forTracking:
            print ('template reaches upper border')
            return 1/0        
        template_height_for_cut_upper = np.int(img.shape[0] - tmplPtCoo[1])        
    if tmplPtCoo[1] - template_height_for_cut_lower < 0:
        if forTracking:
            print ('template reaches lower border')
            return 1/0            
        template_height_for_cut_lower = np.int(tmplPtCoo[1])
        cut_anchor_y = 0
        
    if tmplPtCoo[0] + template_width_for_cut_right > img.shape[1]:
        if forTracking:
            print ('template reaches right border')
            return 1/0    
        template_width_for_cut_right = np.int(img.shape[1] - tmplPtCoo[0])        
    if tmplPtCoo[0] - template_width_for_cut_left < 0:
        if forTracking:
            print ('template reaches right border')
            return 1/0    
        template_width_for_cut_left = np.int(tmplPtCoo[0])
        cut_anchor_x = 0
        
    try:
        #cut template from source image
        template = img[np.int(tmplPtCoo[1])-np.int(template_height_for_cut_lower) : np.int(tmplPtCoo[1])+np.int(template_height_for_cut_upper), 
                       np.int(tmplPtCoo[0])-np.int(template_width_for_cut_left) : np.int(tmplPtCoo[0])+np.int(template_width_for_cut_right)]
    except Exception as e:
        _, _, exc_tb = sys.exc_info()
        print(e, 'line ' + str(exc_tb.tb_lineno))    
        
    anchorPt_lowerLeft = np.asarray([cut_anchor_x, cut_anchor_y], dtype=np.float32) 
    
    return template, anchorPt_lowerLeft


def crossCorrelation(SearchImg, PatchImg, xyLowerLeft, illustrate=False, subpixel=False):
    #perform template matching with normalized cross correlation (NCC)
    res = cv2.matchTemplate(SearchImg, PatchImg, cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res) #min_loc for TM_SQDIFF
    match_position_x = max_loc[0] + PatchImg.shape[1]/2
    match_position_y = max_loc[1] + PatchImg.shape[0]/2
    del min_val, min_loc
        
    if subpixel:
#         zoom_factor = 10.0
#         SearchImg_new, xyLowerLeft_upscale = getTemplate(SearchImg, [match_position_x, match_position_y], PatchImg.shape[0]+2, PatchImg.shape[1]+2)
#         SearchImg_upscale = ndimage.zoom(SearchImg_new, zoom_factor)
#         PatchImg_upscale = ndimage.zoom(PatchImg, zoom_factor)
#         res_upscale = cv2.matchTemplate(SearchImg_upscale, PatchImg_upscale, cv2.TM_CCORR_NORMED)
#         min_val, max_val, min_loc, max_loc_upscale = cv2.minMaxLoc(res_upscale) #min_loc for TM_SQDIFF
#         match_position_x_upscale = np.float((max_loc_upscale[0] + PatchImg_upscale.shape[1]/2)) / zoom_factor
#         match_position_y_upscale = np.float((max_loc_upscale[1] + PatchImg_upscale.shape[0]/2)) / zoom_factor 
#         
#         match_position_x = match_position_x_upscale + xyLowerLeft_upscale[0]
#         match_position_y = match_position_y_upscale + xyLowerLeft_upscale[1]
#         
#         if illustrate:        
#             plt.subplot(131),plt.imshow(res_upscale,cmap = 'gray')
#             plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
#             plt.plot(match_position_x_upscale*zoom_factor-PatchImg_upscale.shape[1]/2, 
#                      match_position_y_upscale*zoom_factor-PatchImg_upscale.shape[0]/2, "r.", markersize=10)
#             plt.subplot(132),plt.imshow(SearchImg_upscale,cmap = 'gray')
#             plt.title('Detected Point'), plt.xticks([]), plt.yticks([])    
#             plt.plot(match_position_x_upscale*zoom_factor-3, match_position_y_upscale*zoom_factor+3, "r.", markersize=10)
#             plt.subplot(133),plt.imshow(PatchImg_upscale,cmap = 'gray')
#             plt.title('Template'), plt.xticks([]), plt.yticks([])
#             plt.show()        
#             plt.waitforbuttonpress()
#             plt.cla()
        
        #perform subpixel matching with template and search area in frequency domain
        SearchImg_32, _ = getTemplate(SearchImg, [match_position_x, match_position_y], PatchImg.shape[0], PatchImg.shape[1])
        SearchImg_32 = np.float32(SearchImg_32)
        PatchImg_32 = np.float32(PatchImg)        
        shiftSubpixel, _ = cv2.phaseCorrelate(SearchImg_32,PatchImg_32)    #subpixle with fourier transform
        match_position_x = match_position_x - shiftSubpixel[0]  #match_position_x - shiftSubpixel[1]
        match_position_y = match_position_y - shiftSubpixel[1]  #match_position_y + shiftSubpixel[0]
               
    if illustrate:    
        plt.subplot(131),plt.imshow(res,cmap = 'gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.plot(match_position_x-PatchImg.shape[1]/2, match_position_y-PatchImg.shape[0]/2, "r.", markersize=10)
        plt.subplot(132),plt.imshow(SearchImg,cmap = 'gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])    
        plt.plot(match_position_x, match_position_y, "r.", markersize=10)
        plt.subplot(133),plt.imshow(PatchImg,cmap = 'gray')
        plt.title('Template'), plt.xticks([]), plt.yticks([])
        plt.show()        
        plt.waitforbuttonpress()
        plt.cla()
        plt.close('all')
        print('correlation value: ' + str(max_val))
    
    del res
        
    if max_val > 0.9:#998:
        #keep only NCC results with high correlation values
        xyMatched = np.asarray([match_position_x + xyLowerLeft[0], match_position_y + xyLowerLeft[1]], dtype=np.float32)
        return xyMatched
    
    else:
        print('NCC matching not successful')
        return [-999,-999]


def performFeatureTracking(template_size, search_area, initCooTemplate, 
                           templateImage, searchImage, shiftSearchArea,
                           performLSM=True, lsm_buffer=3, thresh=0.001,
                           subpixel=False, plot_result=False):
#template_size: np.array([template_width, template_height])
#search_area: np.array([search_area_x_CC, search_area_y_CC])
#initCooTemplate: np.array([x,y])
#shiftSearchArea: np.array([shiftFromCenter_x, shiftFromCenter_y])
    template_width = template_size[0]
    template_height = template_size[1]
    search_area_x = search_area[0]
    search_area_y = search_area[1]    
    shiftSearchArea_x = shiftSearchArea[0]
    shiftSearchArea_y = shiftSearchArea[1]

    #check if template sizes even and correct correspondingly
    if int(template_width) % 2 == 0:
        template_width = template_width + 1
    if int(template_height) % 2 == 0:
        template_height = template_height + 1
    if int(search_area_x) % 2 == 0:
        search_area_x = search_area_x + 1
    if int(search_area_y) % 2 == 0:
        search_area_y = search_area_y + 1    
    
    #get patch clip
    if plot_result:
        plt.imshow(templateImage)
        plt.plot(initCooTemplate[0], initCooTemplate[1], "r.", markersize=10)
        plt.waitforbuttonpress()
        plt.cla()
        plt.close('all')
        
    try:
        patch, _ = getTemplate(templateImage, initCooTemplate, template_width, template_height, True)
    except Exception as e:
#        _, _, exc_tb = sys.exc_info()
#        print(e, 'line ' + str(exc_tb.tb_lineno))
        print('template patch reaches border')
        return 1/0
    
    #shift search area to corresponding position considering movement direction
    templateCoo_init_shift =  np.array([initCooTemplate[0] + shiftSearchArea_x, initCooTemplate[1] + shiftSearchArea_y])
        
    #get lsm search clip
    try:
        search_area, lowerLeftCoo_lsm_search = getTemplate(searchImage, templateCoo_init_shift, search_area_x, search_area_y, True)
    
    except Exception as e:
#        _, _, exc_tb = sys.exc_info()
#        print(e, 'line ' + str(exc_tb.tb_lineno))
        print('search patch reaches border')
        return 1/0

    if plot_result:
        plt.ion() 
    
    CC_xy = crossCorrelation(search_area, patch, lowerLeftCoo_lsm_search, plot_result, subpixel)
    if CC_xy[0] == -999:
        return 1/0
    
    if plot_result:
        plt.close('all')
        print(CC_xy)
    
    TrackedFeature = CC_xy
    
    if performLSM:
        #perform least square matching (subpixel accuracy possible)
        try:
            lsm_search, lowerLeftCoo_lsm_search = getTemplate(searchImage, CC_xy, search_area_x, search_area_y, True)            
        except Exception as e:
#            _, _, exc_tb = sys.exc_info()
#            print(e, 'line ' + str(exc_tb.tb_lineno))
            print('lsm patch reaches border')
            return 1/0
        
        if plot_result:
            plt.imshow(lsm_search)
            plt.waitforbuttonpress()
            plt.close('all')
                      
        pointAdjusted_ = pointAdjusted()
        
        try:
            result_lsm = lsm_matching(patch, lsm_search, pointAdjusted_, lsm_buffer, thresh)            
            print ('sigma LSM tracking: ' + str(result_lsm.s0)) 
        
            if plot_result:
                plt.imshow(searchImage, cmap='gray')
                plt.plot(result_lsm.y + lowerLeftCoo_lsm_search[0], result_lsm.x + lowerLeftCoo_lsm_search[1], "b.", markersize=10)
                plt.waitforbuttonpress()
                plt.close('all')
    
            TrackedFeature = np.asarray([result_lsm.x, result_lsm.y])
        
        except Exception as e:
#            _, _, exc_tb = sys.exc_info()
#            print(e, 'line ' + str(exc_tb.tb_lineno))
            print('lsm failed')
    
    return TrackedFeature


def performFeatureTrackingLK(startImg, searchImg, featuresToSearch, useApprox=False, initialEstimateNewPos=None,
                             searchArea_x=150, searchArea_y=150, maxDistBackForward_px=1):
# use grey scale images    
    featuresToSearchFloat = np.asarray(featuresToSearch, dtype=np.float32)
    
    #parameters for lucas kanade optical flow
    lk_params = dict(winSize  = (searchArea_x,searchArea_y), maxLevel=2, 
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.003),  #15,15 2 0.03
                     flags = cv2.OPTFLOW_USE_INITIAL_FLOW)

    #calculate optical flow
    if useApprox:
        #work with initial estimates, i.e. pre-set shift of search window
        initialEstimateNewPosFloat = np.asarray(initialEstimateNewPos, dtype=np.float32)
        trackedFeatures, status, _ = cv2.calcOpticalFlowPyrLK(startImg, searchImg, featuresToSearchFloat, initialEstimateNewPosFloat,
                                                              None, **lk_params)
        #check backwards
        initialEstimateNewPosFloatCheck = trackedFeatures + (featuresToSearchFloat - initialEstimateNewPosFloat)
        trackedFeaturesCheck, _, _ = cv2.calcOpticalFlowPyrLK(searchImg, startImg, trackedFeatures, initialEstimateNewPosFloatCheck,
                                                              None, **lk_params)        
    else:
        #...or not
        trackedFeatures, status, _ = cv2.calcOpticalFlowPyrLK(startImg, searchImg, featuresToSearchFloat, featuresToSearchFloat, 
                                                              None, **lk_params)
        
        #check backwards
        trackedFeaturesCheck, status, _ = cv2.calcOpticalFlowPyrLK(searchImg, startImg, trackedFeatures, trackedFeatures, 
                                                                   None, **lk_params)         
    
    #set points that fail backward forward tracking test to nan
    distBetweenBackForward = abs(featuresToSearch-trackedFeaturesCheck).reshape(-1, 2).max(-1)
    keepGoodTracks = distBetweenBackForward < maxDistBackForward_px    
    trackedFeaturesDF = pd.DataFrame(trackedFeatures, columns=['x','y'])
    trackedFeaturesDF.loc[:,'check'] = keepGoodTracks
    trackedFeaturesDF = trackedFeaturesDF.where(trackedFeaturesDF.check == True)
    trackedFeaturesDF = trackedFeaturesDF.drop(['check'], axis=1)
    trackedFeatures = np.asarray(trackedFeaturesDF)
    
    cv2.destroyAllWindows()
    
    return trackedFeatures, status


def performDenseFeatureTracking(startImg, searchImg):   
    #perform dense optical flow measurement
    flow = cv2.calcOpticalFlowFarneback(startImg,searchImg, None, 0.5, 3, 5, 3, 5, 1.2, 0)
    
    return flow

    
    