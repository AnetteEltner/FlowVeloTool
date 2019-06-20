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


import numpy as np
import pandas as pd
import pylab as plt
import scipy.ndimage as ndimage
import scipy.spatial
from sklearn.neighbors import KDTree
from PIL import Image, ImageDraw

import cv2

import photogrammetry_functions as photo_tool
import draw_functions as draw_tool

def defineFeatureSearchArea(pointCloud, cameraGeometry_interior, cameraGeometry_exterior, plot_results=False,
                            savePlot=False, dirOut=None, img_name=None):      

    #project points into depth image
    xyd_rgb_map = photo_tool.project_pts_into_img(cameraGeometry_exterior, cameraGeometry_interior, pointCloud, False)
    if xyd_rgb_map is None:
        print('point projection into image failed')
        return
        
    print('point cloud with ' + str(pointCloud.shape[0]) + ' points projected into img')    
    
    #plot projected point cloud
    if plot_results:
        if pointCloud.shape[1] <= 3:
            print('drawing point cloud to image not possible because rgb info missing')
        else:
            rgb = xyd_rgb_map[:,3:6] / 256
            fig, ax = plt.subplots()
            ax.scatter(xyd_rgb_map[:,0], -1*xyd_rgb_map[:,1], s=5, edgecolor=None, lw = 0, facecolors=rgb)
            plt.title('point cloud in image space')
            plt.show()
            plt.close('all')
            del fig, ax
    
    xyd = xyd_rgb_map[:,0:3]

    if savePlot:
        fig, ax = plt.subplots()
        ax.scatter(xyd_rgb_map[:,0], -1*xyd_rgb_map[:,1], s=5, edgecolor=None, lw = 0)
        plt.title('point cloud in image space')
        plt.savefig(dirOut+img_name[:-4] + '_PtcldImg.png', dpi=600, pad_inches=0)
        plt.close('all')
        del fig, ax
    
    
    del xyd_rgb_map
    
    #find border coordinates of features search mask (is within image frame and greater negative values)
    xyd_index = np.asarray(xyd[:,0:2], dtype=np.int)
    xyd_index = xyd_index[xyd_index[:,0] < cameraGeometry_interior.resolution_x]
    xyd_index = xyd_index[xyd_index[:,0] > 0]
    xyd_index = xyd_index[xyd_index[:,1] < cameraGeometry_interior.resolution_y]
    xyd_index = xyd_index[xyd_index[:,1] > 0]
    
    xyd_img = np.zeros((cameraGeometry_interior.resolution_y, cameraGeometry_interior.resolution_x))
    xyd_img = np.uint8(xyd_img)
    xyd_img[xyd_index[:,1], xyd_index[:,0]] = 1

    #convert into binary image
    (_, xyd_img) = cv2.threshold(xyd_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)    
    
    #perform closing to get closed water surface area
    kernel = np.ones((20,20),np.uint8)
    xyd_img_close = cv2.morphologyEx(xyd_img, cv2.MORPH_CLOSE, kernel)    
    
#     #invert image to get contour
#     xyd_img_close_inverted = cv2.bitwise_not(xyd_img_close)
    
    #get contour
    _, cnts, _ = cv2.findContours(xyd_img_close, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  #(cnts, _) 
    del xyd_img_close
    contours = sorted(cnts, key = cv2.contourArea, reverse = True)
    
    #extract largest contour
    MaskBorderPts = contours[0]
    MaskBorderPts = MaskBorderPts.reshape(MaskBorderPts.shape[0],2)
    
    if savePlot: 
        plt.figure()
        plt.gray()
        plt.imshow(xyd_img)
        plt.plot([p[0] for p in MaskBorderPts],
                    [p[1] for p in MaskBorderPts],
                    marker='o', ms=2, color='none', markeredgecolor='green', markeredgewidth=2)
        plt.axis('off')
        plt.savefig(dirOut+img_name[:-4] + '_FeatureSearchArea.png', dpi=600, pad_inches=0)
        plt.close('all')

    return MaskBorderPts


def featureDetection(dirImg, img_name, border_pts, minimum_thresh=100, neighbor_search_radius=50,
                     maximum_neighbors=10, max_ft_nbr=500, sensitive_FD=0.02, savePlot=False, 
                     dir_out='', improve_img=False):
    
    '''Load image and clip image'''
    img = cv2.imread(dirImg + img_name, 0)
    
    if improve_img:
        img_eq = np.asarray(img[:], dtype=np.uint8)
        img = cv2.equalizeHist(img_eq)                
    
    img_clipped = raster_clip(img, 0, border_pts, False, False, False)
    img_clipped = np.asarray(img_clipped, dtype=np.uint8)
                
    
    '''---- detect features ---- -> Good Features to Track'''
    goodFtTr = cv2.goodFeaturesToTrack(img_clipped, max_ft_nbr, sensitive_FD, 10)
    goodFtTr = goodFtTr.reshape(goodFtTr.shape[0], goodFtTr.shape[2])
    goodFtTr_x, goodFtTr_y = goodFtTr[:,1], goodFtTr[:,0] 
    goodFtTr = np.hstack((goodFtTr_x.reshape(goodFtTr_x.shape[0], 1), goodFtTr_y.reshape(goodFtTr_y.shape[0], 1)))
    
    
    '''---- filter features ---- -> Filter detected features to remove wrong large particles'''
    '''remove features where value below threshold (i.e. too dark features)'''
    goodFtTr_x_int = np.asarray(goodFtTr_x, dtype=np.int).reshape(goodFtTr_x.shape[0],1)
    goodFtTr_y_int = np.asarray(goodFtTr_y, dtype=np.int).reshape(goodFtTr_y.shape[0],1)
    pt_vals_from_array = img_clipped[goodFtTr_x_int, goodFtTr_y_int]
    
    FtAboveThresh = np.hstack((goodFtTr, pt_vals_from_array))
    FtAboveThresh = FtAboveThresh[FtAboveThresh[:,2] > minimum_thresh]
    print('Removing too dark features. Remaining features: ' + str(FtAboveThresh.shape[0]))
    
    
    '''remove features detected along border of (clipped) image'''
    goodFtTr_x_int_min = np.asarray(FtAboveThresh[:,0], dtype=np.int).reshape(FtAboveThresh.shape[0],1)
    goodFtTr_y_int_min = np.asarray(FtAboveThresh[:,1], dtype=np.int).reshape(FtAboveThresh.shape[0],1)
    
    minimum_filtered_img = ndimage.minimum_filter(img_clipped, size=10)
    pt_vals_from_array_min = minimum_filtered_img[goodFtTr_x_int_min, goodFtTr_y_int_min]
    FtAboveMinThresh = np.hstack((FtAboveThresh[:,0:2], pt_vals_from_array_min))
    FtAboveMinThresh = FtAboveMinThresh[FtAboveMinThresh[:,2] > 20]
    print('Removing features along border. Remaining features: ' + str(FtAboveMinThresh.shape[0]))
    
    
    '''Cluster Analysis (looking for nearest neighbors using kdtree)'''
    #FtAboveMinThresh_x = np.asarray(FtAboveMinThresh[:,1], dtype=np.int).reshape(FtAboveMinThresh.shape[0],1)
    #FtAboveMinThresh_y = np.asarray(FtAboveMinThresh[:,0], dtype=np.int).reshape(FtAboveMinThresh.shape[0],1)
    #FtAboveMinThresh_forCluster = np.hstack((FtAboveMinThresh_x, FtAboveMinThresh_y))
    
    kdtree = KDTree(FtAboveMinThresh[:,0:2], leaf_size=2)  
    neighbors = []
    max_neighbors = 0
    for FeaturePt in FtAboveMinThresh[:,0:2]:
        FeaturePt = FeaturePt.reshape(1,FeaturePt.shape[0])        
        neighbor_count = kdtree.query_radius(FeaturePt, r=neighbor_search_radius, count_only=True)
        if neighbor_count < maximum_neighbors:
            neighbors.append(FeaturePt.reshape(FeaturePt.shape[1],1))
        if neighbor_count > max_neighbors:
            max_neighbors = neighbor_count
    neighbors = np.asarray(neighbors, np.uint32)
    
    border_row_min = np.min(np.asarray(border_pts)[:,0])
    border_col_min = np.min(np.asarray(border_pts)[:,1])
    
    neighbors[:,0] = neighbors[:,0] + border_col_min
    neighbors[:,1] = neighbors[:,1] + border_row_min  
    print('Removing features in clusters. Remaining features: ' + str(neighbors.shape[0]))
    
    del kdtree
    
    
    '''write images'''
    if savePlot:       
        #filtered via amount nearest neighbors criteria
        plot = draw_tool.drawPointsToImg(img, neighbors)
        plot.savefig(dir_out+img_name[:-4] + '_circles_NN.png', dpi=600)
        plot.close('all')
                
    return neighbors


def NN_pts_FD(reference_pts, target_pts, max_NN_dist=1, plot_results=False):     
    reference_pts_xy_int = np.asarray(reference_pts[:,0:2], dtype = np.int)
    target_pts_int = np.asarray(target_pts[:,0:2], dtype = np.int)
    
    points_list = list(target_pts_int)

    #define kd-tree
    mytree = scipy.spatial.cKDTree(reference_pts_xy_int)
    
    #search for nearest neighbour
    distances, indexes = mytree.query(points_list, k=1)   #find points within specific distance (here in pixels)
    
    #filter neighbours to keep only point closest to camera if several NN found
    NN_skip = 0
    points_target_final = []
    points_NN_final = []
    
    i = 0
    for nearestPts_id in indexes:
        if distances[i] > max_NN_dist:  #no nearby point found, and thus skip
            NN_skip = NN_skip + 1
            i = i + 1  
            continue
        
        #select points closest to target point        
        points_NN_final.append(reference_pts[nearestPts_id,:])
        points_target_final.append(target_pts[i,:])
        
        i = i + 1  
                       
    if NN_skip > 0:
        print('NN skipped: ' + str(NN_skip))

    return np.asarray(points_NN_final), np.asarray(points_target_final)


def NN_pts(reference_pts, target_pts, max_NN_dist=1, plot_results=False):     
    reference_pts_xy_int = np.asarray(reference_pts[:,0:2], dtype = np.int)
    target_pts_int = np.asarray(target_pts, dtype = np.int)
    
    points_list = list(target_pts_int)

    #define kd-tree
    mytree = scipy.spatial.cKDTree(reference_pts_xy_int)
    
    #search for nearest neighbour
    indexes = mytree.query_ball_point(points_list, max_NN_dist)   #find points within specific distance (here in pixels)
    
    #filter neighbours to keep only point closest to camera if several NN found
    NN_skip = 0
    points_target_final = []
    points_NN_final = []
    
    i = 0
    for nearestPts_ids in indexes:
        if not nearestPts_ids:  #if no nearby point found, skip
            NN_skip = NN_skip + 1
            continue
        
        #select all points found close to waterline point
        nearestPtsToWaterPt_d = reference_pts[nearestPts_ids,0:3]
        nearestPts_ids = np.asarray(nearestPts_ids)

        df_nearestPtsToWaterPt_d = pd.DataFrame(nearestPtsToWaterPt_d)        
        id_df_nearestPtsToWaterPt_d = df_nearestPtsToWaterPt_d.loc[df_nearestPtsToWaterPt_d[2].idxmin()]
        closestCameraPt = np.asarray(id_df_nearestPtsToWaterPt_d)
        
        points_NN_final.append(closestCameraPt)
        points_target_final.append(points_list[i][:])
        
        i = i + 1  
                       
    if NN_skip > 0:
        print('NN skipped: ' + str(NN_skip))

    return np.asarray(points_NN_final), np.asarray(points_target_final)
    
    
def getWaterborderXYZ(borderPts, ptCloud, exteriorOrient, interiorOrient):
    
    #project points into depth image
    xyd_rgb_map = photo_tool.project_pts_into_img(exteriorOrient, interiorOrient, ptCloud, False)
    if xyd_rgb_map.any() == None:
        print('point projection into image failed')
        return
    
    #undistort border points
    borderPts_undist = photo_tool.undistort_img_coos(borderPts, interiorOrient, False)  
    borderPts_undist_px = photo_tool.metric_to_pixel(borderPts_undist, interiorOrient.resolution_x, 
                                                     interiorOrient.resolution_y, interiorOrient.sensor_size_x, 
                                                     interiorOrient.sensor_size_y)
    
    #find nearest depth value to border points in depth image
    borderPts_xyd, borderPtsNN_undist_px = NN_pts(xyd_rgb_map, borderPts_undist_px, 5, False)
    if borderPts_xyd.any() == None:
        print('no NN for border found')
        return    
        
    borderPts_xyd_mm = photo_tool.pixel_to_metric(borderPts_xyd[:,0:2], interiorOrient.resolution_x, interiorOrient.resolution_y, 
                                                  interiorOrient.sensor_size_x, interiorOrient.sensor_size_y)
    
    borderPts_mm_d = borderPts_xyd[:,2]
    xyd_map = np.hstack((borderPts_xyd_mm, borderPts_mm_d.reshape(borderPts_mm_d.shape[0],1)))
    xyd_map_mm = photo_tool.imgDepthPts_to_objSpace(xyd_map, exteriorOrient, interiorOrient.resolution_x, 
                                                    interiorOrient.resolution_y, interiorOrient.sensor_size_x / interiorOrient.resolution_x, 
                                                    interiorOrient.ck)
    
    return  xyd_map_mm, borderPtsNN_undist_px


def world2Pixel(geoMatrix, xy):
#estimation of raster position according to point coordinates (e.g. for raster clipping with shape)
    ulX = geoMatrix[0]  #origin X
    ulY = geoMatrix[3]  #origin Y
    xDist = geoMatrix[1]    #pixel width
    yDist = geoMatrix[5]    #pixel height

    xy_len = np.shape(xy)[0]
    row = np.rint((xy[:,0] - np.ones(xy_len) * ulX) / xDist)
    col = np.rint((xy[:,1] - np.ones(xy_len) * ulY) / (yDist))
        
    return np.array(list([row, col])).T   #integer


''' clip raster with given polygon to keep information only within clip '''
def raster_clip(ras_to_clip, geotrans, polygon, visualize=False, flipped_rows=False, world2Pix=True,
                return_rasClip=False):
#polygon is list of X and Y coordinates
    #transform coordinates of polygon vertices to pixel coordinates
    if world2Pix:
        poly_coo = world2Pixel(geotrans, polygon)
    else:
        poly_coo = np.asarray(polygon, dtype=np.uint)
    
    #determine min and max for image extent setting
    x_min = np.nanmin(poly_coo[:,0])
    if x_min < 0:
        x_min = 0
    y_min = np.nanmin(poly_coo[:,1])
    if y_min < 0:
        y_min = 0
    x_max = np.nanmax(poly_coo[:,0])
    if x_max > ras_to_clip.shape[1]:
        x_max = ras_to_clip.shape[1]
    y_max = np.nanmax(poly_coo[:,1])
    if y_max > ras_to_clip.shape[0]:
        y_max = ras_to_clip.shape[0]
        
    if y_min > y_max or x_min > x_max:
        print('error with raster extent')
        return         
    
    else:
        #define image with corresponding size
        img = Image.new('1', (int(x_max - x_min), int(y_max - y_min)))
        
        #set minimal necessary image extent according to polygon size
        poly_coo_x = poly_coo[:,0] - np.ones(poly_coo.shape[0])*x_min
        poly_coo_y = poly_coo[:,1] - np.ones(poly_coo.shape[0])*y_min
        poly_coo_sm = np.array([poly_coo_x, poly_coo_y]).T
        
        #draw image with mask as 1 and outside as 0
        poly_coo_flat = [y for x in poly_coo_sm.tolist() for y in x]    
        ImageDraw.Draw(img).polygon(poly_coo_flat, fill=1)
        del poly_coo_x, poly_coo_y, poly_coo_sm, poly_coo_flat
        
        #convert image to array, consider that rows and columns are switched in image format
        mask_list = []
        for pixel in iter(img.getdata()):
            mask_list.append(pixel)
        mask = np.array(mask_list).reshape(int(y_max-y_min), int(x_max-x_min))
        del img, mask_list
    
        #add offset rows and columns to obtain original raster size
        if flipped_rows:
            add_rows_down = np.zeros((int(y_min), int(x_max-x_min)))
            add_rows_up = np.zeros((int(ras_to_clip.shape[0]-y_max), int(x_max-x_min)))
        else:
            add_rows_down = np.zeros((int(ras_to_clip.shape[0]-y_max), int(x_max-x_min)))
            add_rows_up = np.zeros((int(y_min), int(x_max-x_min)))
            
        add_cols_left = np.zeros((int(ras_to_clip.shape[0]), int(x_min)))
        add_cols_right = np.zeros((int(ras_to_clip.shape[0]), int(ras_to_clip.shape[1]-x_max)))
        mask_final = np.vstack((add_rows_up, mask))
        mask_final = np.vstack((mask_final, add_rows_down))
        mask_final = np.hstack((add_cols_left, mask_final))
        mask_final = np.hstack((mask_final, add_cols_right))
                
        #extract values within clip (from poylgon)                                  
        mask_final[mask_final==0]=np.nan
        ras_clipped = mask_final * ras_to_clip.reshape(mask_final.shape[0], mask_final.shape[1])
        
        ras_clipped_to_extent = np.delete(ras_clipped, np.s_[mask.shape[0] + add_rows_up.shape[0] : ras_clipped.shape[0]], 0)
        ras_clipped_to_extent = np.delete(ras_clipped_to_extent, np.s_[0 : add_rows_up.shape[0]], 0)
        ras_clipped_to_extent = np.delete(ras_clipped_to_extent, np.s_[mask.shape[1] + add_cols_left.shape[1] : ras_clipped.shape[1]], 1)
        ras_clipped_to_extent = np.delete(ras_clipped_to_extent, np.s_[0 : add_cols_left.shape[1]], 1)

        del mask
        
        if visualize:
            #plt.imshow(ras_to_clip)
            plt.imshow(ras_clipped)
            plt.show()
            plt.close('all')

        if not return_rasClip:       
            return ras_clipped_to_extent
        else:
            return ras_clipped_to_extent, ras_clipped, np.asarray([x_min, y_min])

        
def LSPIV_features(dirImg, img_name, border_pts, pointDist_x, pointDist_y, savePlot=False, dir_out=''):
    
    '''Load image and clip image'''
    img = cv2.imread(dirImg + img_name, 0)             
    
    grid = np.indices((img.shape[0], img.shape[1]))
    img_id_x =  grid[1]
    img_id_y =  grid[0]
    
    img_clipped_x = raster_clip(img_id_x, 0, border_pts, False, False, False)
    img_clipped_y = raster_clip(img_id_y, 0, border_pts, False, False, False)
    
    '''define features'''
    features_col = img_clipped_x[np.int(pointDist_x/2)::np.int(pointDist_x/2),np.int(pointDist_y/2)::np.int(pointDist_y/2)]
    features_row = img_clipped_y[np.int(pointDist_x/2)::np.int(pointDist_x/2),np.int(pointDist_y/2)::np.int(pointDist_y/2)]
    
    features = np.hstack((features_row.reshape(features_row.shape[0]*features_row.shape[1],1), 
                          features_col.reshape(features_col.shape[0]*features_col.shape[1],1)))
    
    features = np.asarray(features, np.uint32)
    
    if savePlot:
        plot = draw_tool.drawPointsToImg(img, features)
        plot.savefig(dir_out+img_name[:-4] + '_circles_NN.png', dpi=600)
        plot.close('all')
    
    return features
    