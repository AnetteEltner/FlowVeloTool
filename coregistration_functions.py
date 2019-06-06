# Copyright (c) 2012, Jan Erik Solem
# All rights reserved.
#
# Copyright (c) 2019, Anette Eltner
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


import os, csv
import numpy as np
import pylab as plt
import matplotlib

import cv2

import draw_functions as drawF


'''perform coregistration'''
def coregistration(image_list, directory_out, kp_nbr=None, sift_vers=False, 
                   feature_match_twosided=False, nbr_good_matches=10, 
                   master_0 = True):
    
    if not os.path.exists(directory_out):
        os.system('mkdir ' + directory_out)
    
    master_img_name = image_list[1]
    master_img_dirs = master_img_name.split("/")    
    img_master = cv2.imread(master_img_name)
    #cv2.imwrite(os.path.join(directory_out, master_img_dirs[-1])[:-4] + '_coreg.jpg', img_master)         
    
    if master_0 == True:    #matchin to master
        '''detect Harris keypoints in master image'''
        keypoints_master, _ = HarrisCorners(master_img_name, kp_nbr, False)        
        
        '''calculate ORB or SIFT descriptors in master image'''
        if not sift_vers:
            keypoints_master, descriptor_master = OrbDescriptors(master_img_name, keypoints_master)
            print('ORB descriptors calculated for master ' + master_img_dirs[-1])
        else: 
            keypoints_master, descriptor_master = SiftDescriptors(master_img_name, keypoints_master)    
            print('SIFT descriptors calculated for master ' + master_img_dirs[-1])
    
    
    '''border mask preparation (for temp texture)'''
    maskForBorderRegion_16UC1 = np.ones((img_master.shape[0], img_master.shape[1]))
    maskForBorderRegion_16UC1 = maskForBorderRegion_16UC1.astype(np.uint16)
    
    
    '''perform co-registration for each image'''
    i = 0
    while i < len(image_list):

        slave_img_name = image_list[i-1]
        slave_img_dirs = slave_img_name.split("/") 
                
        if master_0 == False:   #matching always to subsequent frame (no master)
            '''skip first image (because usage of subsequent images)'''
            if i == 0:
                i = i + 1
                continue   
            
            '''detect Harris keypoints in master image'''
            keypoints_master, _ = HarrisCorners(slave_img_name, kp_nbr, False)           
            
            '''calculate ORB or SIFT descriptors in master image'''
            if not sift_vers:
                keypoints_master, descriptor_master = OrbDescriptors(slave_img_name, keypoints_master)
                print('ORB descriptors calculated for master ' + slave_img_dirs[-1])
            else: 
                keypoints_master, descriptor_master = SiftDescriptors(slave_img_name, keypoints_master)    
                print('SIFT descriptors calculated for master ' + slave_img_dirs[-1])
        
         
        '''skip first image (because already read as master)'''
        if slave_img_dirs[-1] == master_img_dirs[-1]:
            i = i + 1
            continue
    
        slave_img_name_1 = image_list[i]
        slave_img_dirs_1 = slave_img_name_1.split("/") 

        '''detect Harris keypoints in image to register'''
        keypoints_image, _ = HarrisCorners(slave_img_name_1, kp_nbr, False)
    
        '''calculate ORB or SIFT descriptors in image to register'''
        if not sift_vers:
            keypoints_image, descriptor_image = OrbDescriptors(slave_img_name_1, keypoints_image)
            print('ORB descriptors calculated for image ' + slave_img_dirs_1[-1])
        else:
            keypoints_image, descriptor_image = SiftDescriptors(slave_img_name_1, keypoints_image)
            print('SIFT descriptors calculated for image ' + slave_img_dirs_1[-1])
        
        
        '''match images to master using feature descriptors (SIFT)'''  
        if not sift_vers:
            matched_pts_master, matched_pts_img = match_DescriptorsBF(descriptor_master, descriptor_image, keypoints_master, keypoints_image,
                                                                      True,feature_match_twosided)
            matched_pts_master = np.asarray(matched_pts_master, dtype=np.float32)
            matched_pts_img = np.asarray(matched_pts_img, dtype=np.float32)
        else:
            if feature_match_twosided:        
                matched_pts_master, matched_pts_img = match_twosidedSift(descriptor_master, descriptor_image, keypoints_master, keypoints_image, "FLANN")    
            else:
                matchscores = SiftMatchFLANN(descriptor_master, descriptor_image)
                matched_pts_master = np.float32([keypoints_master[m[0].queryIdx].pt for m in matchscores]).reshape(-1,2)
                matched_pts_img = np.float32([keypoints_image[m[0].trainIdx].pt for m in matchscores]).reshape(-1,2)
        
        print('number of matches: ' + str(matched_pts_master.shape[0]))
        
        
        '''calculate homography from matched image points and co-register images with estimated 3x3 transformation'''
        if matched_pts_master.shape[0] > nbr_good_matches:
            # Calculate Homography
            H_matrix, _ = cv2.findHomography(matched_pts_img, matched_pts_master, cv2.RANSAC, 3)
            
            # Warp source image to destination based on homography
            img_src = cv2.imread(slave_img_name_1)
            img_coregistered = cv2.warpPerspective(img_src, H_matrix, (img_master.shape[1],img_master.shape[0]))      #cv2.PerspectiveTransform() for points only
            
            #save co-registered image
            cv2.imwrite(os.path.join(directory_out, slave_img_dirs_1[-1])[:-4] + '_coreg.jpg', img_coregistered)
            
            
            '''Mask for border region'''
            currentMask = np.ones((img_master.shape[0], img_master.shape[1]))
            currentMask = currentMask.astype(np.uint16)
            currentMask = cv2.warpPerspective(currentMask, H_matrix, (img_master.shape[1],img_master.shape[0]))
            maskForBorderRegion_16UC1 = maskForBorderRegion_16UC1 * currentMask
            
        i = i + 1   
    
    
    write_file = open(directory_out + 'mask_border.txt', 'wb')        
    writer = csv.writer(write_file, delimiter=",")
    writer.writerows(maskForBorderRegion_16UC1)
    write_file.close()
    

#detect Harris corner features
def HarrisCorners(image_file, kp_nbr=None, visualize=False, img_import=False):
    
    if img_import:
        image_gray = image_file
    else:
        image = cv2.imread(image_file)
        image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                                                         
    image_gray = np.uint8(image_gray) 
    
    '''detect Harris corners'''
    keypoints = cv2.cornerHarris(image_gray,2,3,0.04)
    keypoints = cv2.dilate(keypoints,None)                                                              
    
    #reduce keypoints to specific number
    thresh_kp_reduce = 0.01
    keypoints_prefilt = keypoints
    keypoints = np.argwhere(keypoints > thresh_kp_reduce * keypoints.max())

    if not kp_nbr == None:
        keypoints_reduced = keypoints
        while len(keypoints_reduced) >= kp_nbr:
            thresh_kp_reduce = thresh_kp_reduce + 0.01
            keypoints_reduced = np.argwhere(keypoints_prefilt > thresh_kp_reduce * keypoints_prefilt.max())
    else:
        keypoints_reduced = keypoints       
        
    keypoints = [cv2.KeyPoint(x[1], x[0], 1) for x in keypoints_reduced]
    
    return keypoints, keypoints_reduced #keypoints_reduced for drawing


#calculate ORB descriptors at detected features (using various feature detectors)
def OrbDescriptors(image_file, keypoints):
    image = cv2.imread(image_file)
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)                                             
    image_gray = np.uint8(image_gray) 
    
    '''perform ORB'''
    if "3." in cv2.__version__:
        orb = cv2.ORB_create()
    else:
        orb = cv2.ORB()
    keypoints, descriptors = orb.compute(image_gray, keypoints)
    
    return keypoints, descriptors


#calculate SIFT descriptors at detected features (using various feature detectors)
def SiftDescriptors(image_file, keypoints):
    image = cv2.imread(image_file)
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)                                             
    image_gray = np.uint8(image_gray) 
        
    '''perform SIFT'''
    if "3." in cv2.__version__:
        siftCV2 = cv2.xfeatures2d.SIFT_create()
        #siftCV2 = cv2.SIFT_create()
    else:
        siftCV2 = cv2.SIFT()
    keypoints, descriptors = siftCV2.compute(image_gray, keypoints)
    descriptors = descriptors.astype(np.uint8)
    
    return keypoints, descriptors


#match SIFT features using SIFT matching
#source code from Jan Erik Solem
def match_SIFT(desc1, desc2):
    '''For each descriptor in the first image, select its match in the second image.
    input: desc1 (descriptors for the first image),
    desc2 (same for the second image).'''
    
    desc1 = np.array([d/plt.linalg.norm(d) for d in desc1])
    desc2 = np.array([d/plt.linalg.norm(d) for d in desc2])
    
    dist_ratio = 0.6
    desc1_size = desc1.shape
    
    matchscores = np.zeros((desc1_size[0],1),'int')
    desc2t = desc2.T #precompute matrix transpose
    for i in range(desc1_size[0]):
        dotprods = np.dot(desc1[i,:], desc2t)   #vector of dot products
        dotprods = 0.9999*dotprods
        # inverse cosine and sort, return index for features in second Image
        indx = np.argsort(plt.arccos(dotprods))
        
        # check if nearest neighbor has angle less than dist_ratio times 2nd
        if plt.arccos(dotprods)[indx[0]] < dist_ratio * plt.arccos(dotprods)[indx[1]]:
            matchscores[i] = np.int(indx[0])
            
    return matchscores


#match SIFT features using SIFT matching and perform two-sided
#source code from Jan Erik Solem
def match_twosided_SIFT(desc1, desc2):
    '''Two-sided symmetric version of match().'''
    
    matches_12 = match_SIFT(desc1,desc2)
    matches_21 = match_SIFT(desc2,desc1)
    
    ndx_12 = matches_12.nonzero()[0]
    
    # remove matches that are not symmetric
    for n in ndx_12:
        if matches_21[int(matches_12[n])] != n:
            matches_12[n] = 0
            
    return matches_12


#match SIFT image features using FLANN matching
#source code from Jan Erik Solem    
def SiftMatchFLANN(des1,des2):
    max_dist = 0
    min_dist = 100
    
    # FLANN parameters   
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    
    if des1.dtype != np.float32:
        des1 = des1.astype(np.float32)
    if des2.dtype != np.float32:
        des2 = des2.astype(np.float32)
    
    matches = flann.knnMatch(des1,des2,k=2)
       
    # ratio test as per Lowe's paper
    for m,n in matches:
        if min_dist > n.distance:
            min_dist = n.distance
        if max_dist < n.distance:
            max_dist = n.distance
    
    good = []
    for m,n in matches:
        #if m.distance < 0.75*n.distance:
        if m.distance <= 3*min_dist:
            good.append([m])
    
    return good


#match SIFT image features using FLANN matching and perform two-sided matching
#source code from Jan Erik Solem
def match_twosidedSift(desc1, desc2, kp1, kp2, match_Variant="FLANN"):
    '''Two-sided symmetric version of match().'''    
    if match_Variant == "FLANN":
        matches_12 = SiftMatchFLANN(desc1,desc2)
        matches_21 = SiftMatchFLANN(desc2,desc1)
    elif match_Variant == "BF":
        matches_12 = SiftMatchBF(desc1,desc2)
        matches_21 = SiftMatchBF(desc2,desc1)

    pts1 = []
    pts2 = []
    for m in matches_12:
        pts1.append(kp1[m[0].queryIdx].pt)
        pts2.append(kp2[m[0].trainIdx].pt)

    pts1_b = []
    pts2_b = []    
    for m in matches_21:
        pts2_b.append(kp1[m[0].trainIdx].pt)
        pts1_b.append(kp2[m[0].queryIdx].pt)
    
    pts1_arr = np.asarray(pts1)
    pts2_arr = np.asarray(pts2)
    pts_12 = np.hstack((pts1_arr, pts2_arr))
    pts1_arr_b = np.asarray(pts1_b)
    pts2_arr_b = np.asarray(pts2_b)        
    pts_21 = np.hstack((pts1_arr_b, pts2_arr_b))
       
    pts1_ts = []
    pts2_ts = []        
    for pts in pts_12:
        pts_comp = np.asarray(pts, dtype = np.int)
        for pts_b in pts_21:
            pts_b_comp = np.asarray(pts_b, dtype = np.int)
            if ((int(pts_comp[0]) == int(pts_b_comp[2])) and (int(pts_comp[1]) == int(pts_b_comp[3]))
                and (int(pts_comp[2]) == int(pts_b_comp[0])) and (int(pts_comp[3]) == int(pts_b_comp[1]))):
                pts1_ts.append(pts[0:2].tolist())
                pts2_ts.append(pts[2:4].tolist())                
                break
    
    pts1 = np.asarray(pts1_ts, dtype=np.float32)
    pts2 = np.asarray(pts2_ts, dtype=np.float32)
    
    #print('Matches twosided calculated')
        
    return pts1, pts2


#match STAR image features using bruce force matching
#source code from Jan Erik Solem
def match_DescriptorsBF(des1,des2,kp1,kp2,ratio_test=True,twosided=True):
    '''Match STAR descriptors between two images'''
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors.
    matches = bf.match(des1,des2)                    
    
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)     
    
    pts1 = []
    pts2 = []        
    
    if ratio_test: 
        # ratio test as per Lowe's paper
        good = []
        for m in matches:
            if m.distance < 100:
                good.append(m)
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)
    else:
        for m in matches:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            
    if twosided:
        pts1_b = []
        pts2_b = []
        
        matches_back = bf.match(des2,des1)
        for m in matches_back:
            pts2_b.append(kp1[m.trainIdx].pt)
            pts1_b.append(kp2[m.queryIdx].pt)
        
        pts1_arr = np.asarray(pts1)
        pts2_arr = np.asarray(pts2)
        pts_12 = np.hstack((pts1_arr, pts2_arr))
        pts1_arr_b = np.asarray(pts1_b)
        pts2_arr_b = np.asarray(pts2_b)        
        pts_21 = np.hstack((pts1_arr_b, pts2_arr_b))
       
        
        pts1_ts = []
        pts2_ts = []        
        for pts in pts_12:
            pts_comp = np.asarray(pts, dtype = np.int)
            for pts_b in pts_21:
                pts_b_comp = np.asarray(pts_b, dtype = np.int)
                if ((int(pts_comp[0]) == int(pts_b_comp[2])) and (int(pts_comp[1]) == int(pts_b_comp[3]))
                    and (int(pts_comp[2]) == int(pts_b_comp[0])) and (int(pts_comp[3]) == int(pts_b_comp[1]))):
                    pts1_ts.append(pts[0:2].tolist())
                    pts2_ts.append(pts[2:4].tolist())
                    
                    break
        
        pts1 = pts1_ts
        pts2 = pts2_ts      
        
        #print('Matches calculated')
            
    return pts1, pts2


#match SIFT image features using bruce force matching    
#source code from Jan Erik Solem
def SiftMatchBF(des1, des2):
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    
    return good
    
    
def accuracy_coregistration(image_list_coreg, check_pts_img, template_size, output_dir):
    
    write_all = open(os.path.join(output_dir, 'stat_img_coreg_all.txt'), 'wb')
    writer_all = csv.writer(write_all, delimiter="\t")
    write_stat = open(os.path.join(output_dir, 'stat_img_coreg.txt'), 'wb')
    writer_stat = csv.writer(write_stat, delimiter="\t")
    
    first_image = True

    distance_matched_points_for_stat = np.ones((check_pts_img.shape[0],1))
    image_list_coreg_dirs = image_list_coreg[0].split("/")   
    frame_ids = image_list_coreg_dirs[-1]
    frame_ids = np.asarray(frame_ids)
    frame_ids = frame_ids.reshape(1,1)
    
    for image in image_list_coreg:
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
        if first_image:
            '''template matching --> create templates from first image pairs with approximate position due to img point information'''
            #include template size to approximate template position table for subsequent template extraction
            template_approx_pos = check_pts_img
            
            #calculate template with corresponding template size
            template_img, _ = getTemplateAtImgpoint(img, template_approx_pos, template_size, template_size)
            
            first_image = False 
    
        else:  
            '''approximation of template position for subsequent images'''
            master_pts = check_pts_img
            
            # Apply template Matching --> print with img with new pts
            template_approx_size = check_pts_img
            approx_pos_template, anchor_pts = getTemplateAtImgpoint(img, template_approx_size, template_size*3, template_size*3)
            check_pts_img = performTemplateMatch(approx_pos_template, template_img, anchor_pts)
            
            dist_check_master = np.sqrt(np.square(master_pts[:,0] - check_pts_img[:,0]) + np.square(master_pts[:,1] - check_pts_img[:,1]))
            frame_id_dirs = image.split("/")   
            frame_id = frame_id_dirs[-1]

            if frame_id == 'output_033.jpg':
                print()
            frame_id = np.asarray(frame_id)
            frame_id = frame_id.reshape(1,1)

            dist_check_master = dist_check_master.reshape(dist_check_master.shape[0],1)
            
            distance_matched_points_for_stat = np.hstack((distance_matched_points_for_stat, dist_check_master))
            frame_ids = np.hstack((frame_ids, frame_id))   
    
    distance_matched_points = distance_matched_points_for_stat[:,1:distance_matched_points_for_stat.shape[1]]
    
    distance_matched_points_for_stat = np.vstack((frame_ids, distance_matched_points_for_stat))
    distance_matched_points_for_stat = distance_matched_points_for_stat[:,1:distance_matched_points_for_stat.shape[1]]
    
    
    '''calculate statistics'''
    point_nbr = []
    for i in range(distance_matched_points.shape[0]):
        point_nbr.append(i+1)
    writer_stat.writerow(['id', point_nbr])    
    
    average_dist_per_point = np.mean(distance_matched_points, axis=1)
    writer_stat.writerow(['mean', average_dist_per_point])
    std_dist_per_point = np.std(distance_matched_points, axis=1)
    writer_stat.writerow(['std', std_dist_per_point])    
    max_dist_per_point = np.max(distance_matched_points, axis=1)
    writer_stat.writerow(['max', max_dist_per_point])    
    min_dist_per_point = np.min(distance_matched_points, axis=1)
    writer_stat.writerow(['min', min_dist_per_point])    
   
    writer_all.writerows(distance_matched_points_for_stat)

        
    '''draw errorbar'''
    matplotlib.rcParams.update({'font.family': 'serif',
                                'font.size' : 12,})    
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)    
    ax.spines["right"].set_visible(False)  
    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left()  
    ax.set_xlabel('point ID')
    ax.set_ylabel('point deviations [pixel]')
    ax.set_xlim(0, len(point_nbr)+1)
    ax.set_xticks(np.arange(0, len(point_nbr)+1, 1))
    distance_matched_points = distance_matched_points.T    
    ax.errorbar(point_nbr, average_dist_per_point, xerr=0, yerr=std_dist_per_point, 
                 fmt='o', ecolor='g')

    plt.savefig(os.path.join(output_dir, 'errorbar_checkPts.jpg'),
                bbox_inches="tight", dpi=600)
    
    
    '''draw check point locations'''
    drawF.draw_points_onto_image(cv2.imread(image_list_coreg[0], 0), 
                                 check_pts_img, point_nbr, 5, 15)
    plt.savefig(os.path.join(output_dir, 'accuracy_coreg_checkPts_location.jpg'), dpi=600)
    
    
#define template at image point position (of corresponding GCP)
def getTemplateAtImgpoint(img, img_pts, template_width=10, template_height=10):
#consideration that row is y and column is x   
#careful that template extent even to symmetric size around point of interest 
    
    template_img = []
    anchor_pts = []
    for pt in img_pts:
        if img_pts.shape[1] > 2:
            template_width_for_cut_left = pt[2]/2
            template_width_for_cut_right = pt[2]/2 + 1
        elif template_width > 0:
            template_width_for_cut_left = template_width/2
            template_width_for_cut_right = template_width/2 + 1
        else:
            print('missing template size assignment')
        
        if img_pts.shape[1] > 2:
            template_height_for_cut_lower = pt[3]/2
            template_height_for_cut_upper = pt[3]/2 + 1
        elif template_height > 0:
            template_height_for_cut_lower = template_height/2
            template_height_for_cut_upper = template_height/2 + 1
        else:
            print('missing template size assignment')
        
        cut_anchor_x = pt[0] - template_width_for_cut_left
        cut_anchor_y = pt[1] - template_height_for_cut_lower
        
        #consideration of reaching of image boarders (cutting of templates)
        if pt[1] + template_height_for_cut_upper > img.shape[0]:
            template_height_for_cut_upper = np.int(img.shape[0] - pt[1])
        if pt[1] - template_height_for_cut_lower < 0:
            template_height_for_cut_lower = np.int(pt[1])
            cut_anchor_y = 0
        if pt[0] + template_width_for_cut_right > img.shape[1]:
            template_width_for_cut_right = np.int(img.shape[1] - pt[0])
        if pt[0] - template_width_for_cut_left < 0:
            template_width_for_cut_left = np.int(pt[0])
            cut_anchor_x = 0
        
        template = img[np.int(pt[1]-template_height_for_cut_lower) : np.int(pt[1]+template_height_for_cut_upper), 
                       np.int(pt[0]-template_width_for_cut_left) : np.int(pt[0]+template_width_for_cut_right)]
        
        #template_img = np.dstack((template_img, template))
        template_img.append(template)
        
        anchor_pts.append([cut_anchor_x, cut_anchor_y])
        
    anchor_pts = np.asarray(anchor_pts, dtype=np.float32) 
    #template_img = np.delete(template_img, 0, axis=2) 
    
    return template_img, anchor_pts #anchor_pts defines position of lower left of template in image


#template matching for automatic detection of image coordinates of GCPs
def performTemplateMatch(img_extracts, template_img, anchor_pts, plot_results=False):
    new_img_pts = []
    template_nbr = 0
    
    count_pts = 0
    while template_nbr < len(template_img):
        template_array = np.asarray(template_img[template_nbr])
        if (type(img_extracts) is list and len(img_extracts) > 1) or (type(img_extracts) is tuple and len(img_extracts.shape) > 2):      
            img_extract = img_extracts[template_nbr]
        else:
            img_extract = img_extracts
        res = cv2.matchTemplate(img_extract, template_array, cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res) #min_loc for TM_SQDIFF
        match_position_x = max_loc[0] + template_array.shape[1]/2
        match_position_y = max_loc[1] + template_array.shape[0]/2
        del min_val, min_loc
        
        if max_val > 0.9:
            new_img_pts.append([match_position_x + anchor_pts[template_nbr,0], 
                                match_position_y + anchor_pts[template_nbr,1]])
            count_pts = count_pts + 1
             
        template_nbr = template_nbr + 1

        if plot_results:    
            plt.subplot(131),plt.imshow(res,cmap = 'gray')
            plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
            plt.plot(match_position_x-template_array.shape[1]/2, match_position_y-template_array.shape[0]/2, "r.", markersize=10)
            plt.subplot(132),plt.imshow(img_extract,cmap = 'gray')
            plt.title('Detected Point'), plt.xticks([]), plt.yticks([])    
            plt.plot(match_position_x, match_position_y, "r.", markersize=10)
            plt.subplot(133),plt.imshow(template_array,cmap = 'gray')
            plt.title('Template'), plt.xticks([]), plt.yticks([])
            plt.show()
        
    new_img_pts = np.asarray(new_img_pts, dtype=np.float32)
    new_img_pts = new_img_pts.reshape(count_pts, 2)
         
    return new_img_pts
