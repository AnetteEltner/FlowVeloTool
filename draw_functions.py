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

import os
import numpy as np
import pylab as plt
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.colors as colors
import matplotlib.cm as cmx
import itertools
import scipy.spatial
import pandas as pd
    

def assignPtsBasedOnID(pts1, pts2):
#pts1: ID, x, y
#pts2: ID, x, y

    pts1_coos = []
    pts2_coos = []
    pt_id = []
    nbr_rows = 0
    for row_pts2 in pts2:
        for row_pts1 in pts1:
            if row_pts2[0] == row_pts1[0]:
                pts1_coos.append([row_pts1[1], row_pts1[2]])
                pts2_coos.append([row_pts2[1], row_pts2[2]])
                pt_id.append(row_pts1[0])
                nbr_rows = nbr_rows + 1
                break 
    pts1_coos = np.float32(pts1_coos).reshape(nbr_rows,2)
    pts2_coos = np.float32(pts2_coos).reshape(nbr_rows,2)   
    
    return pts1_coos, pts2_coos, pt_id


def drawPointsToImg(img, points, switchCol=False):
    fig = plt.figure(frameon=False) #dpi of screen resolution

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax) 
    ax.axis('equal')      
    ax.set_axis_off()
              
    if switchCol:
        ax.plot([p[0] for p in points],
                    [p[1] for p in points],
                    marker='o', ms=3, color='none', markeredgecolor='blue', markeredgewidth=1)        
    else:
        ax.plot([p[1] for p in points],
                    [p[0] for p in points],
                    marker='o', ms=3, color='none', markeredgecolor='blue', markeredgewidth=1)
     
    ax.imshow(img, cmap='gray')#, aspect='normal')
    
    return plt


def drawArrowsOntoImg(image, imagePtsStart, imgPtsEnd, arrowHeadSize=0.9, fontSize=12):
# draw image points into image and label the point id
# image_points: array with 2 columns
# point_id: list of point ids in same order as corresponding image_points file; if empty no points labeled
# dpi from screen resolution

    
    fontProperties_text = {'size' : fontSize, 
                           'family' : 'sans-serif'}
    matplotlib.rc('font', **fontProperties_text)
    
    fig = plt.figure(frameon=False) #dpi of screen resolution
    
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('equal')  
    ax.plot([p[0] for p in imagePtsStart], [p[1] for p in imagePtsStart],
            marker='o', ms=2, color='none', markeredgecolor='yellow', markeredgewidth=1)
        
    ax.set_axis_off()
    fig.add_axes(ax)
     
    scale_value = 0.25#0.1
    qv = ax.quiver(imagePtsStart[:,0], imagePtsStart[:,1],  
                    (imgPtsEnd[:,0]-imagePtsStart[:,0]), (imgPtsEnd[:,1]-imagePtsStart[:,1]),   #imgPtsEnd[:,0]-imagePtsStart[:,0]), (imgPtsEnd[:,1]-imagePtsStart[:,1]
                    facecolor='red', linewidth=.1, width=.001, headwidth=5, 
                    headlength=5, angles='xy', scale_units='xy', scale=scale_value, edgecolor='')
    qk = ax.quiverkey(qv, arrowHeadSize, arrowHeadSize, 1, 'arrow scale 1:' + str(int(1/scale_value)), 
                      coordinates='figure', fontproperties=fontProperties_text)
    
    t = qk.text.set_color('w')

    ax.imshow(image, cmap='gray')#, aspect='normal')
        
    return plt
    

def draw_tracks(Final_Vals, image, dir_out, outputImgName, variableToDraw, log_norm=False,
                label_data=False, variableToLabel=None):
    try:
        '''visualize'''
        #sort after flow velocity
        image_points = Final_Vals.sort_values(variableToDraw)
        image_points = image_points.reset_index(drop=True)        
        
        #set colors
        jet = plt.get_cmap('Spectral') 
        # cNorm  = colors.SymLogNorm(linthresh=0.003, linscale=1,
        #                            vmin=image_points['velo'].min(), vmax=image_points['velo'].max())
        if log_norm:
            cNorm  = colors.LogNorm(vmin=image_points[variableToDraw].min(), vmax=image_points[variableToDraw].max())
        else:
            cNorm  = colors.Normalize(vmin=image_points[variableToDraw].min(), vmax=image_points[variableToDraw].max())
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
        
        #set font size
        fontProperties_text = {'size' : 12, 
                               'family' : 'serif'}
        matplotlib.rc('font', **fontProperties_text)
        
        #draw figure
        fig = plt.figure(frameon=False) #dpi of screen resolution
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        ax.axis('equal')  
        fig.add_axes(ax)
    #    ax.plot(image_points['x'], image_points['y'], "g.", markersize=3, markeredgecolor='none')
        
        image_points = image_points.sort_values('id')
        image_points = image_points.reset_index(drop=True)
        
        #add arrows
        if len(image_points['id']) > 1:
            point_n = 0
            label_criteria = 0

            while point_n < image_points.shape[0]:
                try:
                    if label_data:
                        label, xl, yl, arr_x, arr_y = image_points[variableToLabel][point_n], image_points['x'][point_n], image_points['y'][point_n], image_points['x_tr'][point_n], image_points['y_tr'][point_n]
                    else:
                        xl, yl, arr_x, arr_y = image_points['x'][point_n], image_points['y'][point_n], image_points['x_tr'][point_n], image_points['y_tr'][point_n]
                 
                    ax.arrow(xl, yl, arr_x-xl, arr_y-yl, color=scalarMap.to_rgba(image_points[variableToDraw][point_n]), head_width = 3, head_length=3, width=1.5) # arr_x-xl, arr_y-yl
                    point_n = point_n + 1
                
                except Exception as e:
#                    print(e)
#                    print ('skipped point: ' + str(point_n))
#                    print(xl, yl, arr_x-xl, arr_y-yl)
                    point_n = point_n + 1
     
                if label_data:            
                    if label_criteria == 0:
                        ax.annotate(str("{0:.2f}".format(label)), xy = (xl, yl), color='black', **fontProperties_text)
                         
                if point_n == image_points.shape[0]:
                    continue
                 
                if label_data:
                    label_next = image_points[variableToLabel][point_n]
                    if int(label_next) == label:
                        label_criteria = 1
                    else:
                        label_criteria = 0
    
                
        #ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), aspect='normal')
        ax.imshow(image, cmap = 'gray')#, aspect='normal') 
        
        image_points = image_points.sort_values(variableToDraw)
        image_points = image_points.reset_index(drop=True)
        
    #    cmap = mcolors.LinearSegmentedColormap.from_list('my_cmap', scalarMap.to_rgba(image_points[variableToDraw]))
        norm  = mcolors.Normalize(min(image_points[variableToDraw]), max(image_points[variableToDraw]))
        sm = matplotlib.cm.ScalarMappable(cmap=jet, norm=norm) #cmap=cmap
        sm.set_array([])
        
        fig.colorbar(sm, fraction=0.1, pad=0, shrink=0.5)
        
        #save figure
        plt.savefig(os.path.join(dir_out, outputImgName),  dpi=600)
        
    except Exception as e:
        print(e)
    

'''draw points on image'''
def draw_points_onto_image(image, image_points, point_id, markSize=2, fontSize=8, switched=False):
# draw image points into image and label the point id
# image_points: array with 2 columns
# point_id: list of point ids in same order as corresponding image_points file; if empty no points labeled
# dpi from screen resolution
    
    set_markersize = markSize
    
    fontProperties_text = {'size' : fontSize, 
                           'family' : 'serif'}
    matplotlib.rc('font', **fontProperties_text)
    
    fig = plt.figure(frameon=False) #dpi of screen resolution
    
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.axis('equal')  
    fig.add_axes(ax)
     
    if switched:
        ax.plot([p[1] for p in image_points],
                [p[0] for p in image_points],
                marker='o', ms=set_markersize, color='green', markeredgecolor='green', markeredgewidth=1)
    else:
        ax.plot([p[0] for p in image_points],
                 [p[1] for p in image_points],
                 marker='o', ms=set_markersize, color='red', markeredgecolor='black', markeredgewidth=1,
                 linestyle=' ')
               
    #ax.plot(image_points[:,0], image_points[:,1], "r.", markersize=set_markersize, markeredgecolor='black')
    if len(point_id) > 1:
        if not switched:
            for label, xl, yl in zip(point_id, image_points[:,0], image_points[:,1]):
                ax.annotate(str((label)), xy = (xl, yl), xytext=(xl+5, yl+1), color='blue', **fontProperties_text)
        else:
            for label, xl, yl in zip(point_id, image_points[:,1], image_points[:,0]):
                ax.annotate(str((label)), xy = (xl, yl), xytext=(xl+5, yl+1), color='blue', **fontProperties_text)           #str(int(label)

    ax.imshow(image, cmap='gray')#, aspect='normal')
        
    return plt


def prepDrawFlowVelosRaster(img, velos, cell_size, maxNN_dist):

    xy = np.asarray(Pixel2xy(img, cell_size)).T
    
    velos_xy = np.asarray([velos.x.values,velos.y.values,velos.velo.values]).T
    
    min_x = np.min(velos_xy[:,0])
    min_y = np.min(velos_xy[:,1])
    max_x = np.max(velos_xy[:,0])
    max_y = np.max(velos_xy[:,1])
    
    xy = xy[xy[:,0]>=min_x]
    xy = xy[xy[:,0]<=max_x]
    xy = xy[xy[:,1]>=min_y]
    xy = xy[xy[:,1]<=max_y]
    
    NN_veloPt = NN_pts(velos_xy, xy, maxNN_dist)
    NN_veloPt = np.asarray(NN_veloPt)

    NN_veloPt = pd.DataFrame(NN_veloPt)
    NN_veloPt.columns = ['x','y','velo','i']
    NN_veloPt_median = NN_veloPt.groupby('i', as_index=True).velo.median()
    NN_veloPt_x = NN_veloPt.groupby('i', as_index=True).x.head(1)
    NN_veloPt_y = NN_veloPt.groupby('i', as_index=True).y.head(1)
    
    NN_veloPt_x = NN_veloPt_x.values / cell_size
    NN_veloPt_y = NN_veloPt_y.values / cell_size
    NN_veloPt_median = NN_veloPt_median.values
    NN_veloPt_arr = np.hstack((NN_veloPt_x.reshape(NN_veloPt_x.shape[0],1), NN_veloPt_y.reshape(NN_veloPt_y.shape[0],1)))
    NN_veloPt_arr = np.hstack((NN_veloPt_arr, NN_veloPt_median.reshape(NN_veloPt_median.shape[0],1)))

    print('data for illustration prepared')
 
    return NN_veloPt_arr


def draw_tracks_raster(Final_Vals, image, dir_out, outputImgName, variableToDraw, 
                       cell_size, plt_title=None, log_norm=False):

    '''visualize'''
    #sort after flow velocity
    image_points = Final_Vals.sort_values(variableToDraw)
    image_points = image_points.reset_index(drop=True)  

    #set font size
    fontProperties_text = {'size' : 12, 
                           'family' : 'serif'}
    matplotlib.rc('font', **fontProperties_text)
    
    #draw figure
    fig = plt.figure(frameon=False) #dpi of screen resolution
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.axis('equal')  
    fig.add_axes(ax)  

#     edgecolor='black'
#     markeredgewidths=0
#     markersize=10
    transparency=.8      
    
    #set colors
    jet = plt.get_cmap('plasma') 

    # cNorm  = colors.SymLogNorm(linthresh=0.003, linscale=1,
    #                            vmin=image_points['velo'].min(), vmax=image_points['velo'].max())
#     if log_norm:
#         cNorm  = colors.LogNorm(vmin=image_points[variableToDraw].min(), vmax=image_points[variableToDraw].max())
#     else:
#         cNorm  = colors.Normalize(vmin=image_points[variableToDraw].min(), vmax=image_points[variableToDraw].max())
#     scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
       
    #add points
#     point_n = 0
#     while point_n < image_points.shape[0]:
#         plt.plot(image_points.x.values[point_n], image_points.y[point_n], marker='s', ms=markersize, 
#                  color=scalarMap.to_rgba(image_points[variableToDraw][point_n]), 
#                  lw=0, markeredgecolor=edgecolor, markeredgewidth=markeredgewidths, alpha=transparency)
#         point_n = point_n + 1

    
    array_pts = np.zeros((image.shape[0] / cell_size, image.shape[1] / cell_size))
    array_pts[:] = np.nan
    row_ind = np.asarray(image_points.y.values, dtype=np.int) - 1
    col_ind = np.asarray(image_points.x.values, dtype=np.int) - 1
    rowcol_ind = np.vstack((row_ind,col_ind))
    rowcol_ind = rowcol_ind[rowcol_ind[:,0]>=0]
    rowcol_ind = rowcol_ind[rowcol_ind[:,1]>=0]
    row_ind = rowcol_ind[0,:]
    col_ind = rowcol_ind[1,:]
    array_pts[row_ind,col_ind] = image_points.velo.values #pts[:,2].flatten()

    #ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), aspect='normal')
    ax.imshow(image, cmap = 'gray', extent=[0,image.shape[1],0,image.shape[0]])#, aspect='normal') 
    
    
    ax.imshow(array_pts, cmap='plasma', extent=[0,image.shape[1],0,image.shape[0]], interpolation='bilinear', alpha=transparency)
    
    image_points = image_points.sort_values(variableToDraw)
    image_points = image_points.reset_index(drop=True)
    
#    cmap = mcolors.LinearSegmentedColormap.from_list('my_cmap', scalarMap.to_rgba(image_points[variableToDraw]))
    norm  = mcolors.Normalize(min(image_points[variableToDraw]), max(image_points[variableToDraw]))
    sm = matplotlib.cm.ScalarMappable(cmap=jet, norm=norm) #cmap=cmap
    sm.set_array([])
    
    fig.colorbar(sm, fraction=0.1, pad=0, shrink=0.5)
    
    fig.suptitle(plt_title)
    
    #save figure
    plt.savefig(os.path.join(dir_out, outputImgName),  dpi=600)
    

def Pixel2xy(array, cell_size):
    row = []
    col = []
    i = 0
    j = 0
    row_len = np.shape(array)[0]
    col_len = np.shape(array)[1]
    cols = []
    while i < col_len: #row_len
        cols.append(i)
        i += 1
    while j < row_len:  #col_len
        rows = (np.ones(len(cols)) * j).tolist()
        row.append(rows)
        col.append(cols)
        j += 1

    row_merged = list(itertools.chain(*row))
    col_merged = list(itertools.chain(*col))                 
    row = np.array(row_merged)
    col = np.array(col_merged)
    y = np.asarray(row * cell_size, dtype=np.int) 
    x = np.asarray(col * cell_size, dtype=np.int)
 
    return [x, y]


def NN_pts(ref_pts, target_pts, max_NN_dist):
#input ref_pts, target_pts ... array
    
    reference_pts_xy_int = np.asarray(ref_pts[:,0:2], dtype = np.int)
    target_pts_int = np.asarray(target_pts[:,0:2], dtype = np.int)
    
    points_list = list(target_pts_int)

    #define kd-tree
    mytree = scipy.spatial.cKDTree(reference_pts_xy_int)
    
    #search for nearest neighbour
    indexes = mytree.query_ball_point(points_list, max_NN_dist)   #find points within specific distance (here in pixels)

    NN_to_crosssec = []
    i = -1
    for NNpts in indexes:
        i = i + 1
        
        if not NNpts:  #if no nearby point found, skip
            continue
        
        velos = ref_pts[NNpts,2]
        pixel_coos = np.ones((velos.shape[0],1)) * target_pts[i,:]
        i_arr = np.ones((velos.shape[0],1)) * i
        ref_pts_arr = np.hstack((pixel_coos, velos.reshape(velos.shape[0],1)))
        ref_pts_arr = np.hstack((ref_pts_arr, i_arr))
        j = 0
        while j < ref_pts_arr.shape[0]:
            NN_to_crosssec.append(ref_pts_arr[np.int(j),:])
            j = j + 1
        
    return np.asarray(NN_to_crosssec)


def NN_difference(ref_pts, target_pts, max_NN_dist, singlePoint=True):
#input ref_pts, target_pts ... array
    
    reference_pts_xy = np.asarray(ref_pts[:,0:2], dtype = np.float)
    target_pts_xy = np.asarray(target_pts[:,0:2], dtype = np.float)
    
    points_list = list(target_pts_xy)

    #define kd-tree
    mytree = scipy.spatial.cKDTree(reference_pts_xy)
    
    #search for nearest neighbour
    if singlePoint:    
        dist, indexes = mytree.query(points_list)
        
        distFilteredTargets = np.asarray([dist, indexes]).T
        distFilteredTargets = np.hstack((distFilteredTargets, target_pts))
        distFilteredTargets = distFilteredTargets[distFilteredTargets[:,0]<max_NN_dist]
        
        distFilteredTargets = pd.DataFrame(distFilteredTargets)
        distFilteredTargets.columns = ['dist','index','x','y','velo']
        distFilteredShortestToTarget = distFilteredTargets.iloc[distFilteredTargets.groupby('index')['dist'].idxmin(), :]
        distFilteredShortestToTarget = np.asarray(distFilteredShortestToTarget)
        
        indexes = distFilteredShortestToTarget[:,1]
        distFilteredShortestToTarget = distFilteredShortestToTarget[:,2:]
        
    else:
        indexes = mytree.query_ball_point(points_list, max_NN_dist)   #find points within specific distance (here in pixels)

    NN_diff = []
    i = -1
    for NNpts in indexes:
        i = i + 1
        if not NNpts:  #if no nearby point found, skip
            continue        
        
        velo_ref = ref_pts[np.int(NNpts),2]
        
        if singlePoint:
            NN_diff.append([ref_pts[np.int(NNpts),0],ref_pts[np.int(NNpts),1],
                            velo_ref,distFilteredShortestToTarget[i,2]])
        else:
            velo_target = np.ones((velo_ref.shape[0])) * target_pts[i,2]       
            
            velo_diff = velo_ref - velo_target
            velo_diff_std = np.nanstd(velo_diff)
            velo_diff_median = np.nanmedian(velo_diff)
            velo_diff_mean = np.nanmean(velo_diff)
            velo_diff_count_bool = np.isnan(velo_diff)
            velo_diff_count = np.count_nonzero(velo_diff)
            
            NN_diff.append([target_pts[i,0],target_pts[i,1],target_pts[i,2],
                            velo_diff_median,velo_diff_mean,velo_diff_std,velo_diff_count])
        
    return np.asarray(NN_diff)



def rotate_pts(origin, points, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    ox = ox * np.ones((points.shape[0],1))
    oy = oy * np.ones((points.shape[0],1))
    cos_angle = np.cos(angle) * np.ones((points.shape[0],1))
    sin_angle = np.sin(angle) * np.ones((points.shape[0],1))
    px = points[:,0]
    px = px.reshape(px.shape[0],1)
    py = points[:,1]
    py = py.reshape(py.shape[0],1)

    qx = ox + cos_angle * (px - ox) - sin_angle * (py - oy)
    qy = oy + sin_angle * (px - ox) + cos_angle * (py - oy)
    return np.hstack((qx, qy))  