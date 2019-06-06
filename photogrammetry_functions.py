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


import sys, os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.spatial
import matplotlib
import cv2

import draw_functions as drawTools


def getExteriorCameraGeometry(gcpImgPts, gcpObjPts, interior_orient, unit_gcp=1, max_orient_diff = 0.1, ransacApprox=True,
                              exteriorApprox=np.asarray([0, 0, 0, 0, 0, 0]).reshape(6,1), writeToFile=False, dirOut=None):
    #assuming distortion free lens
    #interior_orient: imported with read_aicon function
    #gcpObjPts and gcpImgPts: arrays with coordinates AND point IDs
    
    if ransacApprox:
        #use RANSAC to get initial values of exterior orientation parameters
        try:
            gcpImgPts_pix_id = gcpImgPts[:,0].reshape(gcpImgPts.shape[0],1)
            gcpImgPts_pix = metric_to_pixel(gcpImgPts[:,1:3], interior_orient.resolution_x, interior_orient.resolution_y, 
                                                     interior_orient.sensor_size_x, interior_orient.sensor_size_y)
            gcpImgPts_pix = np.hstack((gcpImgPts_pix_id, gcpImgPts_pix))
            
            #perform estimation with RANSAC
            exteriorApprox_rans, rot_mat_rans, position_rans, inlier_ids = getExteriorCameraGeometry_withRANSAC(gcpImgPts_pix, gcpObjPts, interior_orient, unit_gcp)
            
            #keep only inlier GCPs for further improvement of initial RANSAC values with adjustment
            gcpImgPts_filtered = []
            for inlier in inlier_ids:
                gcpImgPts_filtered.append(gcpImgPts[np.int(inlier),:])
            gcpImgPts = np.asarray(gcpImgPts_filtered)

        except Exception as e:
            _, _, exc_tb = sys.exc_info()
            print(e, 'line ' + str(exc_tb.tb_lineno))
            print('estimation exterior orientation with RANSAC failed')
            return 1/0
   
    '''using resection with adjustment'''
    try:
        #refine RANSAC exterior orientation or calculate exterior orientation from provided initial values
        cam_file_forResection = [interior_orient.ck, 0, 0,    #note that ck has to be negative
                                 0, 0, 0, 0, 0, 0, 0, 0]
        
        #fit points in image space to GCP coordinates in object space
        img_pts, gcp_coos, _ = assign_ImgToObj_Measurement(gcpObjPts, gcpImgPts)
        ImgGCPCoo = np.hstack((img_pts, gcp_coos))
        ImgGCPCoo[:,2:5] = ImgGCPCoo[:,2:5] * unit_gcp
        
        #perform adjustment
        if ransacApprox:
            calib_results, s0, nbrObserv = resection(cam_file_forResection, exteriorApprox_rans, ImgGCPCoo, 0.0001, False)
        else:
            calib_results, s0, nbrObserv = resection(cam_file_forResection, exteriorApprox, ImgGCPCoo, 0.0001, False)
        
        print('camera position accuracy: ' + str(s0))
        print('exterior orientation with adjustment: ' + str(calib_results))

        #proceed with estimated values if they are valid
        if calib_results[0,0] != -9999 and np.isinf(s0) == False:        
            position = calib_results[0:3,0] / unit_gcp            
            
            #convert angles into rotation matrix
            rot_mat = rot_Matrix(calib_results[3,0], calib_results[4,0], calib_results[5,0], 'radians').T
            multipl_array = np.array([[-1,-1,-1],[1,1,1],[-1,-1,-1]])
            rot_mat = rot_mat * multipl_array         
            
            if writeToFile:
                calib_results_out = np.vstack((np.array([[s0[0,0], -999]]), calib_results))
                calib_results_out = np.vstack((np.array([[nbrObserv, -999]]), calib_results_out))
                calib_results_out = pd.DataFrame(calib_results_out, columns=['val', 'std'])            
                id_calib_results = pd.DataFrame({'id': ['nbr_observations', 's0', 'X', 'Y', 'Z', 'omega', 'phi', 'kappa']})
                calib_results_out = pd.concat([id_calib_results, calib_results_out], axis=1, join_axes=[id_calib_results.index])
                calib_results_out.to_csv(dirOut + 'Accuracy_CameraOrient.txt', sep='\t', index=False)
                
        elif ransacApprox:        
            #if resection fails use RANSAC
            s0 = np.zeros((1,1))
            
            rot_mat = rot_Matrix(exteriorApprox_rans[3,0], exteriorApprox_rans[4,0], exteriorApprox_rans[5,0], 'radians').T
            multipl_array = np.array([[-1,-1,-1],[1,1,1],[-1,-1,-1]])
            rot_mat = rot_mat * multipl_array            
            
            position = position_rans[:]
            print('resection adjustment failed and only exterior orientation from RANSAC used')
    
        else:
            print('referencing failed and thus skipped')
            return 1/0
    
        #return exterior orientation where referencing at least within "100% - min_accuracy" of good registration
        if not exteriorApprox.all() == 0:   #use only if exteriorApprox values are given
            position_ref_neg = exteriorApprox[:,0:3] - max_orient_diff * exteriorApprox[:,0:3]   
            position_ref_pos = exteriorApprox[:,0:3] + max_orient_diff * exteriorApprox[:,0:3]   
            print('orient range: ' + str(position_ref_neg) + str(position_ref_pos))
        
            if (position_ref_neg[0] > position[0] or position_ref_pos[0] < position[0] or
                position_ref_neg[1] > position[1] or position_ref_pos[1] < position[1] or
                position_ref_neg[2] > position[2] or position_ref_pos[2] < position[2]):        
                print('orientation too large deviations to approximation')
                return
        
        #shape rotation and position matrix into transformation matrix
        eor_mat = np.hstack((rot_mat.T, position.reshape(position.shape[0],1))) #if rotation matrix received from opencv transpose rot_mat
        eor_mat = np.vstack((eor_mat, [0,0,0,1]))
         
        if position[0] < 0 or position[1] < 0 or position[2] < 0:   #projection center needs to be positive
            print('failed image referencing because projection center negative')
            return
         
        print('image taken from position ' + str(position))
        
        return(eor_mat)
    
    except Exception as e:
        _, _, exc_tb = sys.exc_info()
        print(e, 'line ' + str(exc_tb.tb_lineno))
        print('resection with adjustment failed')
        return 1/0


def getExteriorCameraGeometry_withRANSAC(gcpImgPts, gcpObjPts, interior_orient, unit_gcp):
    #assuming distortion free lens
    #interior_orient: imported with read_aicon function
    #gcpObjPts and gcpImgPts: arrays with coordinates AND point IDs
    
    #convert camera parameters of undistorted image to pixel value for opencv
    pixel_size = interior_orient.sensor_size_x / interior_orient.resolution_x
    ck = -1 * interior_orient.ck / pixel_size
    xh = interior_orient.resolution_x / 2
    yh = interior_orient.resolution_y / 2
    cam_file_forOpenCV = [ck, xh, yh, 0, 0, 0, 0, 0]
    
    #get camera pose 
    rot_mat, position, inliers = register_frame(gcpImgPts, gcpObjPts, cam_file_forOpenCV, False, None, 20)   #True, img_to_read
    inliers = np.asarray(inliers).flatten()

    rot_mat_rans = np.zeros((rot_mat.shape[0], rot_mat.shape[1]))
    rot_mat_rans[:] = rot_mat[:]
    position_rans = np.zeros((position.shape[0], position.shape[1]))
    position_rans[:] = position[:]
    
    try:
        print('nbr of GCP outliers after RANSAC: ' + str(gcpImgPts.shape[0] - inliers.shape[0]))
        print('inlier point ids for RANSAC: ' + str(gcpImgPts[inliers,0]))
        
    except Exception as error:
        print(error)
        print('could not calculate outlier number')
    
    #convert rotation matrix into angles
    multipl_array = np.array([[1,0,0],[0,-1,0],[0,0,1]])  
    rot_matrix = -1 * (np.matrix(rot_mat) * np.matrix(multipl_array))
    rot_matrix = np.asarray(rot_matrix)
    omega, phi, kappa = rot_matrix_to_euler(rot_matrix, 'radian')
    rotation = np.asarray([omega, phi, kappa]) #note that kappa needs to be multiplied with -1 to rotate correctly (-1*omega, phi, kappa)  
    exteriorApprox_rans = np.vstack((position.reshape(position.shape[0],1) * unit_gcp, rotation.reshape(rotation.shape[0],1)))
    print('exterior orientation with RANSAC: ' + str(exteriorApprox_rans.flatten()))       
    
    return exteriorApprox_rans, rot_mat_rans, position_rans, inliers


def project_pts_into_img(eor_mat, ior_mat, point_cloud, plot_results=False, neg_x=False):
    #point cloud including RGB
    #ior_mat from read_aicon_ior

    if point_cloud.shape[1] > 3:
        RGB_included = True
    else:
        RGB_included = False
    
    #exterior image orientation parameters
    img_transform = eor_mat     
    
    '''transform point cloud into camera coordinate system'''
    point_cloudXYZ = point_cloud[:,0:3]
    
    if RGB_included:
        point_cloudRGB = point_cloud[:,3:6]    
    point_cloud = np.matrix(np.linalg.inv(img_transform)) * np.matrix(np.vstack((point_cloudXYZ.T, np.ones(point_cloudXYZ.shape[0])))) 
    point_cloud = point_cloud.T

    if RGB_included:        
        point_cloud = np.hstack((point_cloud[:,0:3], point_cloudRGB)) #reunite transformed point cloud with RGB
    del point_cloudXYZ
    
    #remove points behind the camera
    df_points = pd.DataFrame(point_cloud)  
    df_points = df_points.loc[df_points[2] > 0] 
    point_cloud = np.asarray(df_points)
    del df_points    
    
    '''inbetween coordinate system'''
    x = point_cloud[:,0] / point_cloud[:,2]
    y = point_cloud[:,1] / point_cloud[:,2]
    d = point_cloud[:,2]
    
    if neg_x:
        ptCloud_img = np.hstack((x.reshape(x.shape[0],1)*-1, y.reshape(y.shape[0],1)))
    else:
        ptCloud_img = np.hstack((x.reshape(x.shape[0],1), y.reshape(y.shape[0],1)))
    ptCloud_img = np.hstack((ptCloud_img, d.reshape(d.shape[0],1)))
    if not ptCloud_img.shape[0] > 0:    #take care if img registration already erroneous
        return None
    if RGB_included:
        ptCloud_img = np.hstack((ptCloud_img, point_cloudRGB))
        del point_cloudRGB

    if plot_results:
        if point_cloud.shape[1] > 3:
            rgb = ptCloud_img[:,3:6] / 256
        _, ax = plt.subplots()
        if RGB_included:
            ax.scatter(x, y, s=5, edgecolor=None, lw = 0, facecolors=rgb)
        else:
            ax.scatter(x, y, s=5, edgecolor=None, lw = 0)
        plt.title('3D point cloud in image space')
        plt.show()
        plt.close('all')
        del ax
  
    '''calculate depth map but no interpolation (solely for points from point cloud'''
    ptCloud_img_x = ptCloud_img[:,0] * -1 * ior_mat.ck
    ptCloud_img_y = ptCloud_img[:,1] * ior_mat.ck
    ptCloud_img_proj = np.hstack((ptCloud_img_x.reshape(ptCloud_img_x.shape[0],1), ptCloud_img_y.reshape(ptCloud_img_y.shape[0],1))) 
    ptCloud_img_px = metric_to_pixel(ptCloud_img_proj, ior_mat.resolution_x, ior_mat.resolution_y, 
                                     ior_mat.sensor_size_x, ior_mat.sensor_size_y)
    
    if RGB_included:
        ptCloud_img_px = np.hstack((ptCloud_img_px, ptCloud_img[:,2:6]))
    else:
        z_vals = ptCloud_img[:,2]
        ptCloud_img_px = np.hstack((ptCloud_img_px, z_vals.reshape(z_vals.shape[0],1)))
          
    return ptCloud_img_px


class camera_interior:

    def __init__(self):
        self.xh = 0
        self.yh = 0
        self.ck = 0
        self.A1 = 0
        self.A2 = 0
        self.A3 = 0
        self.B1 = 0
        self.B2 = 0
        self.C1 = 0
        self.C2 = 0
        self.resolution_x = 0        
        self.resolution_y = 0
        self.sensor_size_x = 0
        self.sensor_size_y = 0
        self.r0 = 0
            
            
    def camera_mat(self):
        camera_mat = np.zeros((14,1))
        camera_mat[0] = np.float32(self.xh)        
        camera_mat[1] = np.float32(self.yh)
        camera_mat[2] = np.float32(self.ck)
        camera_mat[3] = np.float32(self.A1)
        camera_mat[4] = np.float32(self.A2)
        camera_mat[5] = np.float32(self.A3)
        camera_mat[6] = np.float32(self.B1)
        camera_mat[7] = np.float32(self.B2)
        camera_mat[8] = np.float32(self.C1)
        camera_mat[9] = np.float32(self.C2)
        camera_mat[10] = np.float32(self.resolution_x)
        camera_mat[11] = np.float32(self.resolution_y)
        camera_mat[12] = np.float32(self.sensor_size_x)
        camera_mat[13] = np.float32(self.sensor_size_y)
        camera_mat[14] = np.float32(self.r0)
                
        return camera_mat     


def read_aicon_ior(directory, ior_file=None):
    #read aicon interior geometry in mm
    if ior_file == None:
        file_read = open(directory)
    else:
        file_read = open(os.path.join(directory, ior_file))
        
    ior_table = file_read.read().split(' ')      #normals created in CC
    file_read.close()

    ior_mat = camera_interior()
    
    ior_mat.ck = np.float(ior_table[2])
    ior_mat.xh = np.float(ior_table[3])
    ior_mat.yh = np.float(ior_table[4])
    ior_mat.A1 = np.float(ior_table[5])
    ior_mat.A2 = np.float(ior_table[6])
    ior_mat.A3 = np.float(ior_table[8])
    ior_mat.r0 = np.float(ior_table[7])
    ior_mat.B1 = np.float(ior_table[9])
    ior_mat.B2 = np.float(ior_table[10])
    ior_mat.C1 = np.float(ior_table[11])
    ior_mat.C2 = np.float(ior_table[12])
    ior_mat.sensor_size_x = np.float(ior_table[13])
    ior_mat.sensor_size_y = np.float(ior_table[14])
    ior_mat.resolution_x = np.int(ior_table[15])
    ior_mat.resolution_y = np.int(ior_table[16])

    return ior_mat


def pixel_to_metric(img_pts, x_resolution, y_resolution, x_size, y_size):
    #convert pixel coordinates into metric camera coordinate system    
    center_x = x_resolution/2 + 0.5
    center_y = y_resolution/2 + 0.5
    pixel_size = x_size/x_resolution
    
    pixel_size_control = y_size/y_resolution
    if not pixel_size > (pixel_size_control - pixel_size * 0.1) and pixel_size < (pixel_size_control + pixel_size * 0.1):
        sys.exit('error with pixel size: x not equal y')        
    
    img_pts_mm_x = np.asarray((img_pts[:,0] - 0.5 - center_x) * pixel_size)
    img_pts_mm_x = img_pts_mm_x.reshape(img_pts_mm_x.shape[0],1)
    img_pts_mm_y = np.asarray((img_pts[:,1] - 0.5 - center_y) * (-1 * pixel_size))
    img_pts_mm_y = img_pts_mm_y.reshape(img_pts_mm_y.shape[0],1)
    img_pts_pixel = np.hstack((img_pts_mm_x, img_pts_mm_y))
    
    return img_pts_pixel


def metric_to_pixel(img_pts, x_resolution, y_resolution, x_size, y_size):
    #convert metric image measurements into pixel
    pixel_size = x_size/x_resolution
    
    pixel_size_control = y_size/y_resolution
    if not pixel_size > (pixel_size_control - pixel_size * 0.1) and pixel_size < (pixel_size_control + pixel_size * 0.1):
        sys.exit('error with pixel size: x not equal y')    
    
    img_pts_pix_x = img_pts[:,0] / pixel_size + np.ones(img_pts.shape[0]) * (x_resolution/2)
    img_pts_pix_x = img_pts_pix_x.reshape(img_pts_pix_x.shape[0], 1)
    img_pts_pix_y = y_resolution - (img_pts[:,1] / pixel_size + np.ones(img_pts.shape[0]) * (y_resolution/2))
    img_pts_pix_y = img_pts_pix_y.reshape(img_pts_pix_y.shape[0], 1)
    img_pts_mm = np.hstack((img_pts_pix_x, img_pts_pix_y))
    
    return img_pts_mm    
        
    
def undistort_img_coos(img_pts, interior_orient, mm_val=False):
# source code from Hannes Sardemann rewritten for Python
    #img_pts: array with x and y values in pixel (if in mm state this, so can be converted prior to pixel)
    #interior_orient: list with interior orientation parameters in mm
    #output: in mm
    
    ck = -1 * interior_orient.ck

    #transform pixel values into mm-measurement
    if mm_val == False:    
        img_pts = pixel_to_metric(img_pts, interior_orient.resolution_x, interior_orient.resolution_y, 
                                  interior_orient.sensor_size_x, interior_orient.sensor_size_y)
        
    x_img = img_pts[:,0]
    y_img = img_pts[:,1]
    x_img_1 = img_pts[:,0]
    y_img_1 = img_pts[:,1]    
    
    #start iterative undistortion
    iteration = 0
    
    test_result = [10, 10]
    
    try:
        while np.max(test_result) > 1e-14:            
            if iteration > 1000:
                #sys.exit('No solution for un-distortion')
                print('Undistortion for this point-set failed. Using original points.')
                
                break
            
            iteration = iteration + 1
            
            camCoo_x = x_img
            camCoo_y = y_img
            
            if interior_orient.r0 == 0:
                x_dash = camCoo_x / (-1 * ck)
                y_dash = camCoo_y / (-1 * ck)
                r2 = x_dash**2 + y_dash**2  #img radius
            else:
                x_dash = camCoo_x
                y_dash = camCoo_y
                if x_dash.shape[0] < 2:
                    r2 = np.float(x_dash**2 + y_dash**2)  #img radius
                else:
                    r2 = x_dash**2 + y_dash**2
                r = np.sqrt(r2)  
                    
            '''extended Brown model'''        
            #radial distoriton   
            if interior_orient.r0 == 0:
                p1 = ((interior_orient.A3 * r2 + (np.ones(r2.shape[0]) * interior_orient.A2)) * r2 + (np.ones(r2.shape[0]) * interior_orient.A1)) * r2            
            else:
                p1 = (interior_orient.A1 * (r**2 - (interior_orient.r0**2)) + interior_orient.A2 * (r**4 - interior_orient.r0**4) + 
                      interior_orient.A3 * (r**6 - interior_orient.r0**6))
                
            dx_rad = x_dash * p1
            dy_rad = y_dash * p1            
            
            #tangential distortion
            dx_tan = (interior_orient.B1 * (r2 + 2 * x_dash**2)) + 2 * interior_orient.B2 * x_dash * y_dash
            dy_tan = (interior_orient.B2 * (r2 + 2 * y_dash**2)) + 2 * interior_orient.B1 * x_dash * y_dash           
            
            #combined distortion
            dx = dx_rad + dx_tan
            dy = dy_rad + dy_tan
            
            x_roof = x_dash + dx
            y_roof = y_dash + dy
                       
            #adding up distortion to recent distorted coordinate
            if interior_orient.r0 == 0:
                x_img_undistort = np.ones(x_dash.shape[0]) * interior_orient.xh - ck * (np.ones(x_roof.shape[0]) + interior_orient.C1) * x_roof - ck * interior_orient.C2 * y_roof
                y_img_undistort = np.ones(y_roof.shape[0]) * interior_orient.yh - ck * y_roof
            else:
                x_img_undistort = np.ones(x_dash.shape[0]) * interior_orient.xh + (np.ones(x_roof.shape[0]) + interior_orient.C1) * x_roof + interior_orient.C2 * y_roof
                y_img_undistort = np.ones(y_roof.shape[0]) * interior_orient.yh + y_roof
                            
            #subtracting distortion from original coordinate
            x_img = x_img_1 - (x_img_undistort - x_img)
            y_img = y_img_1 - (y_img_undistort - y_img)
            
            
            #test result if difference between re-distorted (undistorted) coordinates fit to original img coordinates
            test_result[0] = np.max(np.abs(x_img_undistort - img_pts[:,0]))
            test_result[1] = np.max(np.abs(y_img_undistort - img_pts[:,1]))
    
    except Exception as e:
        print(e)
    
    x_undistort = x_img #in mm
    y_undistort = y_img #in mm
        
    x_undistort = x_undistort.reshape(x_undistort.shape[0],1)
    y_undistort = y_undistort.reshape(y_undistort.shape[0],1)
    img_pts_undist = np.hstack((x_undistort, y_undistort))
    
    return img_pts_undist   #in mm


def assign_ImgToObj_Measurement(obj_pts, img_pts):
#obj_pts: object coordinate (ID, X, Y, Z)
#img_pts: image coordinates (ID, x, y)
    img_coos = []
    gcp_coos = []
    pt_id = []
    nbr_rows = 0
    for row_gcp in obj_pts:
        for row_pts in img_pts:
            if row_gcp[0] == row_pts[0]:
                img_coos.append([row_pts[1], row_pts[2]])
                gcp_coos.append([row_gcp[1], row_gcp[2], row_gcp[3]])
                pt_id.append(row_pts[0])
                nbr_rows = nbr_rows + 1
                break 
    img_coos = np.float32(img_coos).reshape(nbr_rows,2)
    gcp_coos = np.float32(gcp_coos).reshape(nbr_rows,3)   
    
    return img_coos, gcp_coos, pt_id


# source code for least square adjustment from Danilo Schneider rewritten for Python
'''generates observation vector for least squares adjustment'''
def l_vector_resection(ImgCoos_GCPCoos, camera_interior, camera_exterior):
#ImgCoos_GCPCoos: assigned image coordinates and object coordinates of ground control points 
# (numpy array [x_vec, y_vec, X_vec, Y_vec, Z_vec])

    l_vec = np.zeros((2*ImgCoos_GCPCoos.shape[0],1))
    
    i = 0
    for point in ImgCoos_GCPCoos:        
        x, y = model_resection(camera_interior, point[2:5], camera_exterior)
        l_vec[2*i] = point[0]-x
        l_vec[2*i+1] = point[1]-y  
        
        i = i + 1
        
    return l_vec


'''generates design matrix for least squares adjustment'''
def A_mat_resection(ImgCoos_GCPCoos, camera_exterior, camera_interior, e=0.0001, param_nbr=6):
#ImgCoos_GCPCoos: assigned image coordinates and object coordinates of ground control points 
#camera_exterior: coordinates of projection centre  and angles of ration matrix (numpy array [X0, Y0, Z0, omega, phi, kappe])
#camera_interior: interior camera orientation (numpy array [ck, xh, yh, A1, A2, A3, B1, B2, C1, C2, r0]), Brown (aicon) model
#param_nbr: define number of parameters, which are adjusted (standard case only XYZ, OmegaPhiKappa)
#e: epsilon
   
    #generates empty matrix
    A = np.zeros((2*ImgCoos_GCPCoos.shape[0], param_nbr))
    
    #fills design matrix
    camera_exterior = camera_exterior.reshape(camera_exterior.shape[0],1)
    i = 0
    for point in ImgCoos_GCPCoos:     
        for j in range(param_nbr):
            #numerical adjustment (mini distance above and below point to estimate slope)
            parameter1 = np.zeros((camera_exterior.shape[0],1))
            parameter2 = np.zeros((camera_exterior.shape[0],1))
            parameter1[:] = camera_exterior[:]
            parameter1[j] = camera_exterior[j] - e
            parameter2[:] = camera_exterior[:]
            parameter2[j] = camera_exterior[j] + e
            
            x2, y2 = model_resection(camera_interior, point[2:5], parameter2)
            x1, y1 = model_resection(camera_interior, point[2:5], parameter1)
            
            A[2*i,j] = (x2-x1)/(2*e)
            A[2*i+1,j] = (y2-y1)/(2*e)
            
        i = i + 1
        
    return A
            

def rotmat_1(omega, phi, kappa):
    R = np.zeros((3, 3))
    
    R[0,0] = math.cos(phi)*math.cos(kappa)+math.sin(phi)*math.sin(omega)*math.sin(kappa)
    R[1,0] = math.sin(phi)*math.cos(kappa)-math.cos(phi)*math.sin(omega)*math.sin(kappa)
    R[2,0] = math.cos(omega)*math.sin(kappa)
    R[0,1] = -math.cos(phi)*math.sin(kappa)+math.sin(phi)*math.sin(omega)*math.cos(kappa)
    R[1,1] = -math.sin(phi)*math.sin(kappa)-math.cos(phi)*math.sin(omega)*math.cos(kappa)
    R[2,1] = math.cos(omega)*math.cos(kappa)
    R[0,2] = math.sin(phi)*math.cos(omega)
    R[1,2] = -math.cos(phi)*math.cos(omega)
    R[2,2] = -math.sin(omega)
    
    return R

  
def rotmat_2(omega, phi, kappa):
    R = np.zeros((3, 3))
    
    R[0,0] =  math.cos(phi)*math.cos(kappa)
    R[0,1] = -math.cos(phi)*math.sin(kappa)
    R[0,2] =  math.sin(phi)
    R[1,0] =  math.cos(omega)*math.sin(kappa)+math.sin(omega)*math.sin(phi)*math.cos(kappa)
    R[1,1] =  math.cos(omega)*math.cos(kappa)-math.sin(omega)*math.sin(phi)*math.sin(kappa)
    R[1,2] = -math.sin(omega)*math.cos(phi)
    R[2,0] =  math.sin(omega)*math.sin(kappa)-math.cos(omega)*math.sin(phi)*math.cos(kappa)
    R[2,1] =  math.sin(omega)*math.cos(kappa)+math.cos(omega)*math.sin(phi)*math.sin(kappa)
    R[2,2] =  math.cos(omega)*math.cos(phi)
    
    return R


'''general camera model (collinearity/telecentric equations)'''  
def model_resection(camera_interior, GCP, camera_exterior, rot_mat_dir_v1=True):
#camera_exterior: coordiantes of projection centre  and angles of ration matrix (numpy array [X0, Y0, Z0, omega, phi, kappe])
#GCP: ground control point coordinates (numpy array [X, Y, Z])
#camera_interior: interior camera orientation (numpy array [ck, xh, yh, A1, A2, A3, B1, B2, C1, C2, r0]), Brown (aicon) model
#rot_mat_dir_v1: choose rotation matrix version

    ck, xh, yh, A1, A2, A3, B1, B2, C1, C2, r0 = camera_interior

    ProjCentre = camera_exterior[0:3]
    RotMat = camera_exterior[3:6]
    
    if rot_mat_dir_v1:
        R = rotmat_2(RotMat[0], RotMat[1], RotMat[2])
        N = R[0,2]*(GCP[0]-ProjCentre[0]) + R[1,2]*(GCP[1]-ProjCentre[1]) + R[2,2]*(GCP[2]-ProjCentre[2])
    else: 
        R = rotmat_2(RotMat[0], RotMat[1], RotMat[2])
        N = -1

    kx = R[0,0]*(GCP[0]-ProjCentre[0]) + R[1,0]*(GCP[1]-ProjCentre[1]) + R[2,0]*(GCP[2]-ProjCentre[2])
    ky = R[0,1]*(GCP[0]-ProjCentre[0]) + R[1,1]*(GCP[1]-ProjCentre[1]) + R[2,1]*(GCP[2]-ProjCentre[2])
    
    x = -1*ck*(kx/N)
    y = -1*ck*(ky/N)
        
    r = np.sqrt(x*x+y*y)
    
    x = xh + x;
    x = x + x * (A1*(r**2-r0**2)+A2*(r**4-r0**4)+A3*(r**6-r0**6))
    x = x + B1*(r*r+2*x*x) + 2*B2*x*y
    x = x + C1*x + C2*y
    
    y = yh + y;
    y = y + y * (A1*(r**2-r0**2)+A2*(r**4-r0**4)+A3*(r**6-r0**6))
    y = y + B2*(r*r+2*y*y) + 2*B1*x*y
    y = y + 0   
    
    return x, y


''' MAIN FUNCTION FOR SPATIAL RESECTION'''
def resection(camera_interior, camera_exterior, ImgCoos_GCPCoos, e=0.0001, plot_results=False, dir_plot=None):
#source code from Danilo Schneider rewritten for Python
#camera_exterior: estimate of exterior orientation and position (XYZOmegaPhiKappa)
#camera_interior: interior camera orientation (numpy array [ck, xh, yh, A1, A2, A3, B1, B2, C1, C2, r0]), Brown (aicon) model
#ImgCoos_GCPCoos: assigned image coordinates and object coordinates of ground control points 
#e: epsilon

    '''iterative calculation of parameter values'''
    s0 = 0
    restart = False
    camera_exterior_ori = np.zeros((camera_exterior.shape[0],1))
    camera_exterior_ori[:] = camera_exterior[:]                               
    for iteration in range(200):
        
        #only if outlier in image measurement detected
        if restart:
            camera_exterior = np.zeros((camera_exterior_ori.shape[0],1))
            camera_exterior[:] = camera_exterior_ori[:]
            iteration = 0
            restart = False
        
        try:
        
            l = l_vector_resection(ImgCoos_GCPCoos, camera_interior, camera_exterior)
            A = A_mat_resection(ImgCoos_GCPCoos, camera_exterior, camera_interior, e)
        
            '''least squares adjustment'''
            N  = np.matrix(A.T) * np.matrix(A)
            L  = np.matrix(A.T) * np.matrix(l)
            Q  = np.matrix(np.linalg.inv(N))
            dx = Q * L  #N\L
            v  = np.matrix(A) * dx - np.matrix(l)
            s0 = np.sqrt((v.T * v) / (A.shape[0] - A.shape[1])) # sigma-0
            
#             if iteration == 0:
#                 print(v)
        
            ''''adds corrections to the values of unknowns'''
            SUM = 0
            for par_nbr in range(camera_exterior.shape[0]):
                camera_exterior[par_nbr] = camera_exterior[par_nbr] + dx[par_nbr]
                SUM = SUM + np.abs(dx[par_nbr])
            
            ''''stops the iteration if sum of additions is very small'''
            if (SUM < 0.00001):
                break
            
            '''calculate std of corrections to check for outliers'''
            std_v = np.std(v)
            mean_v = np.mean(v)
            
            #remove point if larger 2*std
            for k in range(v.shape[0]):
                if mean_v + 3 * std_v < v[k]:
                    #if k % 2 == 0:
                    print('outlier during resection detected: ', ImgCoos_GCPCoos[int(k/2)])
                    ImgCoos_GCPCoos = np.delete(ImgCoos_GCPCoos, (int(k/2)), axis=0)
                    restart = True
                    break
            if restart:
                continue
            
        except Exception as error:
            print(error)
            return np.asarray([[-9999],[0]]), s0, -9999
        

#     '''Output per iteration (check on convergence)'''
#     print('Iteration ' + str(iteration))
#     print('  Sigma-0: ' + str(s0) + ' mm')
#     print('  Sum of additions: ' + str(SUM))
    
    if plot_results:
        '''Generation of vector field (residuals of image coordinates)'''
        #splits the x- and y-coordinates in two different vectors
        x = ImgCoos_GCPCoos[:,0]
        y = ImgCoos_GCPCoos[:,1]
        vx = np.zeros((v.shape[0],1))
        vy = np.zeros((v.shape[0],1))
        for i in range(ImgCoos_GCPCoos.shape[0]):
            vx[i] = v[i*2]
            vy[i] = v[i*2+1]
    
        #displays residuals in a seperate window
        set_markersize = 2    
        fontProperties_text = {'size' : 10, 
                               'family' : 'serif'}
        matplotlib.rc('font', **fontProperties_text)    
        fig = plt.figure(frameon=False) 
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        fig.add_axes(ax)     
        ax.plot(x, y, 'go')    
        
        a_scale = 1
        a_width = 0.05
        for xl, yl, v_x, v_y in zip(x, y, vx, vy):
            ax.arrow(xl, yl , v_x[0] * a_scale, v_y[0] * a_scale, head_width=a_width, 
                     head_length=a_width*1, fc='k', ec='k')  
    
        plt.show()
        plt.close('all')
        del ax, fig
    
    '''Calculation of standard deviations of estimated parameters'''
    calibration_results = np.zeros((camera_exterior.shape[0],2))
    
    for j in range(camera_exterior.shape[0]):
        calibration_results[j,0] = camera_exterior[j]
        calibration_results[j,1] = s0 * np.sqrt(Q[j,j])
    
    #displays the estimated parameters incl. their standard deviation
    return calibration_results, s0, ImgCoos_GCPCoos.shape[0]


''' Umrechnung Winkel in Rotationsmatrix '''
def rot_Matrix(omega,phi,kappa,unit='grad'):        #radians
# unit: rad = radians, gon, grad
    # gon to radian
    if unit == 'gon':
        omega = omega * (math.pi/200)
        phi = phi * (math.pi/200)
        kappa = kappa * (math.pi/200)
     
    # grad to radian
    elif unit == 'grad':
        omega = omega * (math.pi/180)
        phi = phi * (math.pi/180)
        kappa = kappa * (math.pi/180)
    
    # radian    
    elif unit == 'rad':
        omega = omega
        phi = phi
        kappa = kappa
    
    r11 = math.cos(phi) * math.cos(kappa)
    r12 = -math.cos(phi) * math.sin(kappa)
    r13 = math.sin(phi)
    r21 = math.cos(omega) * math.sin(kappa) + math.sin(omega) * math.sin(phi) * math.cos(kappa)
    r22 = math.cos(omega) * math.cos(kappa) - math.sin(omega) * math.sin(phi) * math.sin(kappa)
    r23 = -math.sin(omega) * math.cos(phi)
    r31 = math.sin(omega) * math.sin(kappa) - math.cos(omega) * math.sin(phi) * math.cos(kappa)
    r32 = math.sin(omega) * math.cos(kappa) + math.cos(omega) * math.sin(phi) * math.sin(kappa)
    r33 = math.cos(omega) * math.cos(phi)
    
    rotMat = np.array(((r11,r12,r13),(r21,r22,r23),(r31,r32,r33)))        
    return rotMat


'''Umrechnung Rotationsmatrix in Winkel'''
def rot_matrix_to_euler(R, unit='grad'):
    y_rot = math.asin(R[2][0]) 
    x_rot = math.acos(R[2][2]/math.cos(y_rot))    
    z_rot = math.acos(R[0][0]/math.cos(y_rot))
    if unit == 'grad':
        y_rot_angle = y_rot *(180/np.pi)
        x_rot_angle = x_rot *(180/np.pi)
        z_rot_angle = z_rot *(180/np.pi)    
    else: #unit is radiant
        y_rot_angle = y_rot
        x_rot_angle = x_rot
        z_rot_angle = z_rot     
    return x_rot_angle,y_rot_angle,z_rot_angle  #omega, phi, kappa


def register_frame(img_pts, gcp_pts, cam_file, plot_results=False, image=None, reprojectionError=5):
    #img_to_read: directory + image name, img file with point ID, x, y
    #img_pts: img point measurement (in pixel) file with point ID, x, y
    #gcp_pts: gcp point measurement file with point ID, X, Y, Z
    #cam_file: interior camera parameters in pixel
    
    gcp_table = np.zeros((gcp_pts.shape[0], gcp_pts.shape[1]))
    gcp_table[:] = gcp_pts[:]
    pts_table = np.zeros((img_pts.shape[0], img_pts.shape[1]))
    pts_table[:] = img_pts[:]
   
    '''read camera file with interior orientation information'''   
    #transform metric values to pixel values
    ck, cx, cy, k1, k2, k3, p1, p2 = cam_file

    ''' give information about interior camera geometry'''
    #camera matrix opencv
    camMatrix = np.zeros((3,3),dtype=np.float32)
    camMatrix[0][0] = ck
    camMatrix[0][2] = cx
    camMatrix[1][1] = ck
    camMatrix[1][2] = cy
    camMatrix[2][2] = 1.0           
    distCoeff = np.asarray([k1, k2, p1, p2, k3], dtype=np.float32)          
    
    '''re-organise coordinates to numpy matrix with assigned pt ids'''
    img_pts, gcp_coos, pt_id = assign_ImgToObj_Measurement(gcp_table, pts_table)
    
    if plot_results:
        img = cv2.imread(image, 0)
        plot = drawTools.draw_points_onto_image(img, img_pts, pt_id, 2, 12, False)
        plot.title('GCPs used for registration')
        plot.show()
        plot.close('all')
        del plot    
    
    '''resolve for exterior camera parameters'''
    #solve for exterior orientation
    img_pts = np.ascontiguousarray(img_pts[:,:2]).reshape((img_pts.shape[0],1,2)) #new for CV3
    _, rvec_solved, tvec_solved, inliers = cv2.solvePnPRansac(gcp_coos, img_pts, camMatrix, distCoeff, reprojectionError) # iterationsCount=2000, reprojectionError=5
#     _, rvec_solved, tvec_solved = cv2.solvePnP(gcp_coos, img_pts, camMatrix, distCoeff,
#                                                rvec_solved, tvec_solved, useExtrinsicGuess=True)
    
    '''convert to angles and XYZ'''
    np_rodrigues = np.asarray(rvec_solved[:,:],np.float64)
    rot_matrix = cv2.Rodrigues(np_rodrigues)[0]
    
    position = -np.matrix(rot_matrix).T * np.matrix(tvec_solved)
    
#     multipl_array = np.array([[1,0,0],[0,-1,0],[0,0,-1]])  
#     rot_matrix = -1*(np.matrix(rot_matrix) * np.matrix(multipl_array))
#     rot_matrix = np.asarray(rot_matrix)
#     omega, phi, kappa = rot_matrix_to_euler(rot_matrix)
#     rotation = np.asarray([omega, phi, kappa])
        
    return rot_matrix, position, inliers


def NN_pts(reference_pts, target_pts, max_NN_dist=1, plot_results=False,
           closest_to_cam=False, ior_mat=None, eor_mat=None):   #ior_mat, eor_mat,     
    #get nearest neighbors
    reference_pts_xy_int = np.asarray(reference_pts[:,0:2], dtype = np.int)
    target_pts_int = np.asarray(target_pts, dtype = np.int)    
    points_list = list(target_pts_int)

    #define kd-tree
    mytree = scipy.spatial.cKDTree(reference_pts_xy_int)
#    dist, indexes = mytree.query(points_list)
#    closest_ptFromPtCloud = reference_pts[indexes,0:3]
    
    #search for nearest neighbour
    indexes = mytree.query_ball_point(points_list, max_NN_dist)   #find points within specific distance (here in pixels)
    
    #filter neighbours to keep only point closest to camera if several NN found
    NN_points_start = True
    NN_skip = 0
    NN_points = None
    dist_to_pz_xy = None
    for nearestPts_ids in indexes:
        if not nearestPts_ids:  #if no nearby point found, skip
            NN_skip = NN_skip + 1
            continue
        
        #select all points found close to waterline point
        nearestPtsToWaterPt_d = reference_pts[nearestPts_ids,0:3]
        nearestPts_ids = np.asarray(nearestPts_ids)
        
        if closest_to_cam:
            '''select only point closest to camera'''             
            #transform image measurement into object space
            imgPts_mm = pixel_to_metric(nearestPtsToWaterPt_d[:,0:2], ior_mat.resolution_x, ior_mat.resolution_y, 
                                        ior_mat.sensor_size_x, ior_mat.sensor_size_y)        
            imgPts_mm_d = nearestPtsToWaterPt_d[:,2]
            xyd_map = np.hstack((imgPts_mm, imgPts_mm_d.reshape(imgPts_mm_d.shape[0],1)))
            xyd_map_mm = imgDepthPts_to_objSpace(xyd_map, eor_mat, ior_mat.resolution_x, ior_mat.resolution_y, 
                                                 ior_mat.sensor_size_x/ior_mat.resolution_x, ior_mat.ck)
            xyd_map_mm = drop_dupl(xyd_map_mm[:,0], xyd_map_mm[:,1], xyd_map_mm[:,2])
              
            #calculate shortest distance to camera centre
            pz_coo = eor_mat[0:3,3]
            pz_ones = np.ones((xyd_map_mm.shape[0], pz_coo.shape[0]))
            pz_coo = pz_coo.reshape(1,pz_coo.shape[0]) * pz_ones
            dist_to_pz = np.sqrt(np.square(pz_coo[:,0]-xyd_map_mm[:,0]) + np.square(pz_coo[:,1]-xyd_map_mm[:,1]) + np.square(pz_coo[:,2]-xyd_map_mm[:,2]))
            dist_to_pz_xy = np.hstack((xyd_map_mm, dist_to_pz.reshape(dist_to_pz.shape[0],1)))
            dist_to_pz_xy_df = pd.DataFrame(dist_to_pz_xy)
            closest_pt_to_cam = dist_to_pz_xy_df.loc[dist_to_pz_xy_df[3].idxmin()]
            closestCameraPt = np.asarray(closest_pt_to_cam)

        df_nearestPtsToWaterPt_d = pd.DataFrame(nearestPtsToWaterPt_d)        
        id_df_nearestPtsToWaterPt_d = df_nearestPtsToWaterPt_d.loc[df_nearestPtsToWaterPt_d[2].idxmin()]
        closestCameraPt = np.asarray(id_df_nearestPtsToWaterPt_d)
        
        if NN_points_start:
            NN_points_start = False
            NN_points = closestCameraPt
        else:
            NN_points = np.vstack((NN_points, closestCameraPt))               


    print('NN skipped: ' + str(NN_skip))

    return NN_points    #, np.min(dist_to_pz_xy[:,2]), np.max(dist_to_pz_xy[:,2])
    
    
def imgDepthPts_to_objSpace(img_pts_xyz, eor_mat, x_resolution, y_resolution, pixel_size, ck):
        
    '''calculate inbetween coordinate system'''
    ptCloud_img_x = img_pts_xyz[:,0] / (-1 * ck)
    ptCloud_img_y = img_pts_xyz[:,1] / ck
    
    x = ptCloud_img_x * img_pts_xyz[:,2]
    y = ptCloud_img_y * img_pts_xyz[:,2]
    z = img_pts_xyz[:,2]
    
    imgPts_xyz = np.hstack((x.reshape(x.shape[0],1), y.reshape(y.shape[0],1)))
    imgPts_xyz = np.hstack((imgPts_xyz, z.reshape(z.shape[0],1)))
    
    '''transform into object space'''
    imgPts_XYZ = np.matrix(eor_mat) * np.matrix(np.vstack((imgPts_xyz.T, np.ones(imgPts_xyz.shape[0])))) 
    imgPts_XYZ = np.asarray(imgPts_XYZ.T)
    
    return imgPts_XYZ[:,0:3]


def exteriorFromFile(calib_results):
    #set correct unit
    position = calib_results[0:3,0]    
    #convert angles into rotation matrix
    rot_mat = rot_Matrix(calib_results[3,0], calib_results[4,0], calib_results[5,0], 'radians').T
    multipl_array = np.array([[-1,-1,-1],[1,1,1],[-1,-1,-1]])
    rot_mat = rot_mat * multipl_array
    #join position and rotation to transformation matrix
    eor_mat = np.hstack((rot_mat.T, position.reshape(position.shape[0],1))) #if rotation matrix received from opencv transpose rot_mat
    eor_mat = np.vstack((eor_mat, [0,0,0,1]))
    
    return eor_mat


def drop_dupl(x,y,z):
    df = pd.DataFrame({'x':x, 'y':y, 'z':z})
    dupl_dropped = df.drop_duplicates(cols=['x', 'y', 'z'])    
    return np.asarray(dupl_dropped)