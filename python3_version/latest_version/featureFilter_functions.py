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


'''filter feature tracks'''

import sys
import numpy as np
import pandas as pd
import featureDetect_functions as detectF
import scipy.spatial
import draw_functions as drawF
from sklearn.neighbors import NearestNeighbors
import math


def TrackFilterMinCount(image_points, minCount):
    try:
        image_pointsId = image_points.copy()
        image_pointsId["id_copy"] = image_pointsId.id
        image_pointsId = image_pointsId.groupby('id_copy', as_index=False).id.count()
        pts_count = np.asarray(image_pointsId)

        #keep only tracks, where feature is tracked across minimum number of frames
        PtsToKeep_TrackLongEnough = pts_count[pts_count[:,1] >= minCount]
        image_points = image_points[image_points.id.isin(PtsToKeep_TrackLongEnough[:,0])]
        image_points = image_points.reset_index(drop=True)
    except Exception as e:
        print(e)
        _, _, exc_tb = sys.exc_info()
        print('line ' + str(exc_tb.tb_lineno))
        print('count filter failed')
    
    return image_points


def TrackFilterSteadiness(image_points, threshAngle):
    try:
        threshAngle_rad = np.radians(threshAngle)
        StdAnglePerTrack = image_points.groupby('id', as_index=True).angle.std()
        StdAnglePerTrack = StdAnglePerTrack.fillna(0)
        print("Average std dev flow direction: " + str(np.degrees(np.nanmean(StdAnglePerTrack))))

        #keep only tracks, where tracked feature across frames are steady enough
        id_steady = StdAnglePerTrack[StdAnglePerTrack < threshAngle_rad]
        id_steady = id_steady.dropna()
        if len(id_steady) > 0:
            image_points = image_points[image_points.id.isin(id_steady.index.values)]
            angleStd_out = np.degrees(np.average(StdAnglePerTrack))
        else:
            angleStd_out = np.nan

        image_points = image_points.reset_index(drop=True)
    except Exception as e:
        print(e)
        _, _, exc_tb = sys.exc_info()
        print('line ' + str(exc_tb.tb_lineno))
        print('steadiness filter failed')

    return image_points, angleStd_out  #export tracks
    

def TrackFilterAngleRange(image_points, threshAngleRange):
    try:
        threshAngleRange_rad = np.radians(threshAngleRange)
        MaxAnglePerTrack = image_points.groupby('id', as_index=True).angle.max()
        MinAnglePerTrack = image_points.groupby('id', as_index=True).angle.min()
        RangeAnglePerTrack = MaxAnglePerTrack - MinAnglePerTrack
        print("Average range flow direction: " + str(np.degrees(np.nanmean(RangeAnglePerTrack))))

        #keep only tracks, where range of directions of tracked feature across frames are below threshold
        id_range = RangeAnglePerTrack[RangeAnglePerTrack < threshAngleRange_rad]
        id_range = id_range.dropna()
        if len(id_range) > 0:
            image_points = image_points[image_points.id.isin(id_range.index.values)]
            angleStd_out = np.degrees(np.average(RangeAnglePerTrack))
        else:
            angleStd_out = np.nan

        image_points = image_points.reset_index(drop=True)
    except Exception as e:
        print(e)
        _, _, exc_tb = sys.exc_info()
        print('line ' + str(exc_tb.tb_lineno))
        print('angle range filter failed')

    return image_points, angleStd_out  #export tracks 


def angleBetweenVecAndXaxis(track):
    #prepare for calculation
    ones = np.ones((track.shape[0],1))
    zeros = np.zeros((track.shape[0],1))
    image_points_xy = pd.DataFrame(np.hstack((ones, zeros)))
    image_points_xy.columns = ['x','y']
    image_points_xy_tr = pd.DataFrame(track.reshape(track.shape[0],2))
    image_points_xy_tr.columns = ['x_tr','y_tr']
    
    #calculate angel between track and x-axis
    dotProd_vect = image_points_xy.x*image_points_xy_tr.x_tr + image_points_xy.y*image_points_xy_tr.y_tr
    len_vec_xy_tr = np.sqrt(np.square(image_points_xy_tr.y_tr)+np.square(image_points_xy_tr.x_tr))
    len_vec_xy = np.sqrt(np.square(image_points_xy.y)+np.square(image_points_xy.x))
    angle = np.arccos(dotProd_vect/(len_vec_xy*len_vec_xy_tr))
        
    for i, value in image_points_xy_tr.x_tr.iteritems():
        if value < 0: #3. Quadrant
            if image_points_xy_tr.y_tr[i] < 0:
                angle[i] = angle[i] + math.pi
        if value > 0: # 4. Quadrant
            if image_points_xy_tr.y_tr[i] < 0:
                angle[i] = angle[i] + math.pi
    
    return angle


def TrackFilterMainflowdirection(image_points, binNbr, angle_buffer=10):
    try:
        image_points = image_points.drop(columns='angle')

        #get first and last position vector (track) of tracked feature across frames
        LastTrackPerTrack_x = image_points.groupby('id', as_index=True).x_tr.last()
        FirstTrackPerTrack_x = image_points.groupby('id', as_index=True).x.first()
        LastTrackPerTrack_y = image_points.groupby('id', as_index=True).y_tr.last()
        FirstTrackPerTrack_y = image_points.groupby('id', as_index=True).y.first()
        x_track = LastTrackPerTrack_x.values - FirstTrackPerTrack_x.values
        y_track = LastTrackPerTrack_y.values - FirstTrackPerTrack_y.values
        track = np.hstack((x_track.reshape(x_track.shape[0],1), y_track.reshape(y_track.shape[0],1)))

        #preparation filter first-last vectors (tracks)
        angle = angleBetweenVecAndXaxis(track).values
        index_angle = np.asarray(LastTrackPerTrack_x.index)
        angle = np.hstack((index_angle.reshape(angle.shape[0],1), angle.reshape(angle.shape[0],1)))
        angle_df = pd.DataFrame(angle)
        angle_df.columns = ['index', 'angle']
        MedianAnglePerTrack = angle_df.set_index('index')
        print("Median angle flow direction: " + str(np.degrees(np.median(MedianAnglePerTrack.values))))

        #filter tracks outside main flow direction
        if binNbr != 0:
            #use histogram analysis to find main direction of flow
            flow_dirs_hist, bin_edges = np.histogram(MedianAnglePerTrack, bins=binNbr)
            bin_edges_RollMean = np.convolve(bin_edges, np.ones((2,))/2, mode='valid')
            flow_dirs_hist = np.hstack((flow_dirs_hist.reshape(flow_dirs_hist.shape[0],1), bin_edges_RollMean.reshape(bin_edges_RollMean.shape[0],1)))
            main_dir_index = np.where(flow_dirs_hist[:,0] == np.max(flow_dirs_hist[:,0]))
            main_dir = flow_dirs_hist[main_dir_index,1]
            print('main flow direction: ' + str(main_dir))

            if np.asarray(main_dir_index).shape[0] == 1:
                main_dir_ind = np.asarray(main_dir_index)
            else:
                print('False index for main flow direction')
                sys.exit()

            angle_array = np.asarray(MedianAnglePerTrack).reshape(MedianAnglePerTrack.shape[0],1)
            angle_id = np.asarray(MedianAnglePerTrack.index.values).reshape(MedianAnglePerTrack.shape[0],1)
            angle_array = np.hstack((angle_id, angle_array))
            id_below_main_dir = np.where(angle_array[:,1] < bin_edges[main_dir_ind])
            id_above_main_dir = np.where(angle_array[:,1] > bin_edges[int(main_dir_ind+1)])

            if np.asarray(id_below_main_dir).size == False and np.asarray(id_above_main_dir).size == False:
                print('Error filtering flow direction (angle index)')
                sys.exit()
            elif np.asarray(id_below_main_dir).size and np.asarray(id_above_main_dir).size:
                id_below_main_dir = np.asarray(id_below_main_dir, dtype=int)[1,:]
                id_above_main_dir = np.asarray(id_above_main_dir, dtype=int)
                angle_filtered_IdForImgPtsDfs = np.vstack((id_below_main_dir.reshape(id_below_main_dir.shape[0],1),
                                                           id_above_main_dir.T))
                angle_filtered_IdForImgPtsDf = angle_array[angle_filtered_IdForImgPtsDfs]
                angle_filtered_IdForImgPtsDf = angle_filtered_IdForImgPtsDf.reshape(angle_filtered_IdForImgPtsDf.shape[0],2)
            elif np.asarray(id_below_main_dir).size:
                angle_filtered_IdForImgPtsDf = angle_array[np.asarray(id_below_main_dir, dtype=int)[0,:]]
            elif np.asarray(id_above_main_dir).size:
                angle_filtered_IdForImgPtsDf = angle_array[np.asarray(id_above_main_dir, dtype=int)[0,:]]

            grouped_img_pts = pd.concat([MedianAnglePerTrack, MedianAnglePerTrack], axis=1)
            grouped_img_pts = grouped_img_pts.loc[np.asarray(angle_filtered_IdForImgPtsDf, dtype=int)[:,0]]

            image_points = image_points[~image_points.id.isin(grouped_img_pts.index.values)]

        else:
            #...or use median angle for all first-last vectors
            angle_buffer_rad = np.radians(angle_buffer)
            threshAngle = [np.median(MedianAnglePerTrack) - angle_buffer_rad, np.median(MedianAnglePerTrack) + angle_buffer_rad]

            MedianAnglePerTrack_filt = MedianAnglePerTrack[MedianAnglePerTrack > threshAngle[0]]
            MedianAnglePerTrack_filt = MedianAnglePerTrack_filt[MedianAnglePerTrack_filt < threshAngle[1]]
            MedianAnglePerTrack_filt = MedianAnglePerTrack_filt.dropna()
            image_points = image_points[image_points.id.isin(MedianAnglePerTrack_filt.index.values)]

        image_points = image_points.reset_index(drop=True)
    except Exception as e:
        print(e)
        _, _, exc_tb = sys.exc_info()
        print('line ' + str(exc_tb.tb_lineno))
        print('main flow direction filter failed')
    
    return image_points, np.degrees(np.median(MedianAnglePerTrack.values))
   

def FilteredTracksGroupPerID(image_points):  
    #get stastics for each filtered track
    MeanVeloPerTrack = image_points.groupby('id', as_index=True).velo.mean()
    MeanVeloPerTrack.name = 'velo_mean'
    StdVeloPerTrack = image_points.groupby('id', as_index=True).velo.std()
    StdVeloPerTrack.name = 'velo_std'
    DistancePerTrack = image_points.groupby('id', as_index=True).dist.mean()
    x_Track_start = image_points.groupby('id', as_index=True).x.first()
    y_Track_start = image_points.groupby('id', as_index=True).y.first()
    x_Track_last = image_points.groupby('id', as_index=True).x_tr.last()
    y_Track_last = image_points.groupby('id', as_index=True).y_tr.last()
    X_track_start = image_points.groupby('id', as_index=True).X_start.first()
    Y_track_start = image_points.groupby('id', as_index=True).Y_start.first()
    X_track_end = image_points.groupby('id', as_index=True).X_end.last()
    Y_track_end = image_points.groupby('id', as_index=True).Y_end.last()    
    id_Track = image_points.groupby('id', as_index=True).id.last()
    filteredTracksOutput = pd.concat([id_Track, x_Track_start, y_Track_start, x_Track_last, y_Track_last,
                                      X_track_start, Y_track_start, X_track_end, Y_track_end,
                                      MeanVeloPerTrack, StdVeloPerTrack, DistancePerTrack], axis=1)
    
    return filteredTracksOutput


def DefineRFeatures_forRasterbasedFilter(img, border_pts, cell_size):    
    '''Clip image'''
    grid = np.indices((img.shape[0], img.shape[1]))
    img_id_x =  grid[1]
    img_id_y =  grid[0]
    
    img_clipped_x = detectF.raster_clip(img_id_x, 0, border_pts, False, False, False)
    img_clipped_y = detectF.raster_clip(img_id_y, 0, border_pts, False, False, False)
    
    '''define features'''
    features_col = img_clipped_x[np.int(cell_size/2)::np.int(cell_size/2),np.int(cell_size/2)::np.int(cell_size/2)]
    features_row = img_clipped_y[np.int(cell_size/2)::np.int(cell_size/2),np.int(cell_size/2)::np.int(cell_size/2)]
    
    features = np.hstack((features_col.reshape(features_col.shape[0]*features_col.shape[1],1),
                          features_row.reshape(features_row.shape[0]*features_row.shape[1],1)))
    
    features = np.asarray(features, np.uint32)
    
    features = features[features[:,0]<img.shape[1]]
    features = features[features[:,1]<img.shape[0]]

    return features

    
def NN_filter(velo_points, targeting_pts, max_NN_dist, points3D=False):
    
    if points3D: 
        target_pts = np.asarray(targeting_pts[['X','Y','Z']], dtype = np.float)
        search_points = np.asarray(velo_points[['X','Y','Z']], dtype = np.float)
    else:
        target_pts = np.asarray(targeting_pts[['x','y']], dtype = np.float)
        search_points = np.asarray(velo_points[['x','y']], dtype = np.float)       

    #define kd-tree
    velotree = scipy.spatial.cKDTree(search_points)
    targettree = scipy.spatial.cKDTree(target_pts)
    
    #search for nearest neighbour
    indexes = targettree.query_ball_tree(velotree, max_NN_dist)   #find points within specific distance (here in pixels)
    
    NN_diff = []
    i = -1

    for NNpts in indexes:
        i = i + 1
        if not NNpts:  #if no nearby point found, skip
            continue
        try:
            NNpts = np.asarray(NNpts, dtype=np.int)
            velosPerTarget = velo_points.iloc[NNpts,:]

            velo_std = velosPerTarget.velo.std()
            velo_median = velosPerTarget.velo.median()
            dist_median = velosPerTarget.dist_metric.median()
            velo_count = velosPerTarget.dist.count()
            x_median = velosPerTarget.x.median()
            y_median = velosPerTarget.y.median()
            x_tr_median = velosPerTarget.x_tr.median()
            y_tr_median = velosPerTarget.y_tr.median()

            xtr_cell = targeting_pts.x[i] + (x_tr_median-x_median)
            ytr_cell = targeting_pts.y[i] + (y_tr_median-y_median)
            NN_diff.append([targeting_pts.X[i], targeting_pts.Y[i], targeting_pts.Z[i],
                            targeting_pts.id[i], targeting_pts.x[i],targeting_pts.y[i], xtr_cell, ytr_cell,
                            dist_median,velo_median,velo_std,velo_count])
        except Exception as e:
            print(e)
            _, _, exc_tb = sys.exc_info()
            print('line ' + str(exc_tb.tb_lineno))
            print('NN filter failed')

    NN_diff = pd.DataFrame(NN_diff)
    NN_diff.columns = ['X','Y','Z','id','x','y','x_tr','y_tr','dist','velo','veloStd','count']
        
    return NN_diff


def weightGauss(value, sigma, mue=0.0):
    weight = np.exp(-(np.square(value - mue) / (2 * np.square(sigma))))

    return weight


def filterGaussian(veloTable, searchRadius=100, NN_nbr=30):
    neighborsTree = NearestNeighbors(n_neighbors=NN_nbr, algorithm='kd_tree').fit(veloTable.loc[:, ['X', 'Y', 'Z']])
    distances, indices = neighborsTree.kneighbors(veloTable.loc[:, ['X', 'Y', 'Z']])

    for i in range(indices.shape[0]):
        # keep only values within search radius
        velos = np.asarray(veloTable.loc[indices[i, :], ['velo']]).flatten()
        distancesVelo = distances[i]
        veloDist = np.vstack((velos, distancesVelo)).T
        veloDist = veloDist[veloDist[:, 1] < searchRadius]
        velos = veloDist[:, 0]
        veloDistances = veloDist[:, 1]

        # get std velocities
        stdVelo = np.std(veloDist[:, 0])
        medianVelo = np.median(veloDist[:, 0])

        # get weights for velocity based on velocity distribution
        weightsVeloGauss = []
        for velocity in velos:
            weightsVeloGauss.append(weightGauss(velocity, sigma=stdVelo, mue=medianVelo))
        weightsVeloGauss = np.asarray(weightsVeloGauss)
        weightsVeloGauss = weightsVeloGauss / (np.ones((weightsVeloGauss.shape[0])) * np.sum(weightsVeloGauss))

        # get weights for velocity based on distance
        weightsDistGauss = []
        for distanceVelo in veloDistances:
            weightsDistGauss.append(weightGauss(distanceVelo, sigma=0.2))
        weightsDistGauss = np.asarray(weightsDistGauss)
        weightsDistGauss = weightsDistGauss / (np.ones((weightsDistGauss.shape[0])) * np.sum(weightsDistGauss))

        # combine distance and velocity based weights to filter velocity
        velos = veloDist[:, 0]
        filteredVelo = np.sum(velos * ((1 / 3 * weightsDistGauss + 2 / 3 * weightsVeloGauss)))
        veloTable.loc[i, ['velo']] = filteredVelo

        return veloTable
