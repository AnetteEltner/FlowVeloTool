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


def TrackFilterMinCount(image_points, minCount):
    id_pts = np.unique(image_points.id)
    pts_count = []
    for ids in id_pts:
        count_id = 0
        for id_pt in image_points.id:
            if id_pt == ids:
                count_id = count_id + 1
        pts_count.append([ids, count_id])
    pts_count = np.asarray(pts_count)
    
    #keep only tracks, where feature is tracked across minimum number of frames
    PtsToKeep_TrackLongEnough = pts_count[pts_count[:,1] >= minCount]
    image_points = image_points[image_points.id.isin(PtsToKeep_TrackLongEnough[:,0])]
    image_points = image_points.reset_index(drop=True)
    
    return image_points


def TrackFilterSteadiness(image_points, threshAngle):
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

    return image_points, angleStd_out  #export tracks
    

def TrackFilterAngleRange(image_points, threshAngleRange):
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
    
    return angle


def TrackFilterMainflowdirection(image_points, binNbr, angle_buffer=10):    
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