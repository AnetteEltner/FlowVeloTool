import os, csv
import numpy as np
import pandas as pd
import cv2


import featureTracking_functions as trackF
import photogrammetry_functions as photogrF
import featureDetect_functions as detectF
import draw_functions as drawF
import featureFilter_functions as filterF
import featureReference_functions as refF


def EstimateExterior(gcpCoo_file, imgCoo_GCP_file, interior_orient, estimate_exterior,
                     unit_gcp, max_orientation_deviation, ransacApprox, angles_eor, pos_eor,
                     directoryOutput):
    try:
        #read object coordinates of GCP (including point ID)
        gcpObjPts_table = np.asarray(pd.read_table(gcpCoo_file, header=None, delimiter='\t'))
    except:
        print('failed reading GCP file (object space)')
            
    try:
        #read pixel coordinates of image points of GCPs (including ID)
        gcpImgPts_table = np.asarray(pd.read_table(imgCoo_GCP_file, header=None, delimiter='\t'))
    except:
        print('failed reading GCP file (imgage space)')
    gcpPts_ids = gcpImgPts_table[:,0]
    gcpPts_ids = gcpPts_ids.reshape(gcpPts_ids.shape[0],1)
    gcpImgPts_to_undist = gcpImgPts_table[:,1:3]
        
    #undistort image measurements of GCP
    gcpImgPts_undist = photogrF.undistort_img_coos(gcpImgPts_to_undist, interior_orient, False)  
    gcpImgPts_undist = np.hstack((gcpPts_ids, gcpImgPts_undist))
            
    #get exterior orientation
    try:
        #estimate exterior orientation from GCPs
        if estimate_exterior:
            if ransacApprox:
                exteriorApprox = np.asarray([0,0,0,0,0,0]).reshape(6,1)
            else:
                exteriorApprox = np.vstack((pos_eor, angles_eor)) * unit_gcp
        
            eor_mat = photogrF.getExteriorCameraGeometry(gcpImgPts_undist, gcpObjPts_table, interior_orient, unit_gcp, max_orientation_deviation, 
                                                         ransacApprox, exteriorApprox, True, directoryOutput)
        
        #...or use predefined camera pose information
        else:        
            rot_mat = photogrF.rot_Matrix(angles_eor[0], angles_eor[1], angles_eor[2], 'radians').T
            rot_mat = rot_mat * np.array([[-1,-1,-1],[1,1,1],[-1,-1,-1]])
            
            eor_mat = np.hstack((rot_mat.T, pos_eor)) #if rotation matrix received from opencv transpose rot_mat
            eor_mat = np.vstack((eor_mat, [0,0,0,1]))
            print(eor_mat)
                 
        eor_mat[0:3,3] = eor_mat[0:3,3] * unit_gcp
        
    except Exception as e:        
        print(e)
        print('Referencing image failed\n')
        
    return eor_mat


def searchMask(waterlevel_pt, waterlevel_buffer, AoI_file, ptCloud, unit_gcp, interior_orient,
               eor_mat, savePlotData, directoryOutput, img_list, preDefAoI=False):
    waterlevel = waterlevel_pt - waterlevel_buffer  #read waterlevel
    #use search mask from file
    if preDefAoI:
        try:
            searchMask = pd.read_table(AoI_file, header=None, delimiter=',')
        except:
            print('reading search mask file failed')
        searchMask = np.asarray(searchMask)      
        
    #...or calculate from water level information and 3D point cloud
    else:
        #select points only below water level to extract river area to search for features...
        pointsBelowWater = ptCloud[ptCloud[:,2] < waterlevel] * unit_gcp
        searchMask = detectF.defineFeatureSearchArea(pointsBelowWater, interior_orient, eor_mat, False, savePlotData, directoryOutput, img_list[1])  #xy        
            
    searchMask = np.asarray(searchMask)
    print('search mask with ' + str(searchMask.shape[0]) + ' points defined\n')
    
    return searchMask


def FeatureDetectionPTV(dir_imgs, img_list, frameCount, minimumThreshBrightness, neighborSearchRadius_FD, searchMask,
                        maximumNeighbors_FD, maxFtNbr_FD, sensitiveFD, savePlotData, directoryOutput, FD_everyIthFrame,
                        first_loop, feature_ID_max=None):
    #detect features in search area with feature detection
    featuresToTrack = detectF.featureDetection(dir_imgs, img_list[frameCount], searchMask, minimumThreshBrightness, neighborSearchRadius_FD, 
                                               maximumNeighbors_FD, maxFtNbr_FD, sensitiveFD, savePlotData, directoryOutput, False)
    feature_ID = np.array(range(featuresToTrack.shape[0]))        
    
    #assign corresponding IDs to features
    if first_loop:
        feature_ID_max = featuresToTrack.shape[0] + 1 
        first_loop = False             
    else: 
        feature_ID = feature_ID_max + feature_ID
        feature_ID_max = feature_ID_max + featuresToTrack.shape[0] + 1

    featuresToTrack_id = np.hstack((feature_ID.reshape(feature_ID.shape[0],1), featuresToTrack[:,1].reshape(featuresToTrack.shape[0],1)))
    featuresToTrack = np.hstack((featuresToTrack_id, featuresToTrack[:,0].reshape(featuresToTrack.shape[0],1)))
    
    print('nbr features detected: ' + str(featuresToTrack.shape[0]) + '\n')
    
    #write detected feature to file
    outputFileFD = open(os.path.join(directoryOutput, 'FD_every_' + str(FD_everyIthFrame) + '_' + img_list[frameCount][:-4]) + '.txt', 'wb')
    writer = csv.writer(outputFileFD, delimiter="\t")
    writer.writerow(['id','x', 'y'])
    writer.writerows(featuresToTrack)
    outputFileFD.flush()
    outputFileFD.close()
    del writer                
                
    return featuresToTrack, first_loop, feature_ID_max


def FeatureDetectionLSPIV(dir_imgs, img_list, frameCount, template_width, template_height, searchMask, FD_everyIthFrame,
                          savePlotData, directoryOutput, first_loop, feature_ID_max=None):
    #extract features in search area with pre-defined grid
    featuresToTrack = detectF.LSPIV_features(dir_imgs, img_list[frameCount], searchMask, template_width, template_height, savePlotData,
                                             directoryOutput)
    feature_ID = np.array(range(featuresToTrack.shape[0]))        
           
    if first_loop:
        feature_ID_max = featuresToTrack.shape[0] + 1 
        first_loop = False             
    else: 
        feature_ID = feature_ID_max + feature_ID
        feature_ID_max = feature_ID_max + featuresToTrack.shape[0] + 1

    featuresToTrack_id = np.hstack((feature_ID.reshape(feature_ID.shape[0],1), featuresToTrack[:,1].reshape(featuresToTrack.shape[0],1)))
    featuresToTrack = np.hstack((featuresToTrack_id, featuresToTrack[:,0].reshape(featuresToTrack.shape[0],1)))
    
    print('nbr features detected: ' + str(featuresToTrack.shape[0]) + '\n')
    
    #write detected feature to file
    outputFileFD = open(os.path.join(directoryOutput, 'FD_every_' + str(FD_everyIthFrame) + '_' + img_list[frameCount][:-4]) + '.txt', 'wb')
    writer = csv.writer(outputFileFD, delimiter="\t")
    writer.writerow(['id','x', 'y'])
    writer.writerows(featuresToTrack)
    outputFileFD.flush()
    outputFileFD.close()
    del writer                
                
    return featuresToTrack, first_loop, feature_ID_max


def FeatureTracking(template_width, template_height, search_area_x_CC, search_area_y_CC, shiftSearchFromCenter_x, shiftSearchFromCenter_y,
                    frameCount, FT_forNthNberFrames, TrackEveryNthFrame, dir_imgs, img_list, featuresToTrack, interior_orient,
                    performLSM, lsmBuffer, threshLSM, subpixel, trackedFeaturesOutput_undist, save_gif, imagesForGif, directoryOutput,
                    lk, initialEstimatesLK):
    #prepare function input
    template_size = np.asarray([template_width, template_height])
    search_area = np.asarray([search_area_x_CC, search_area_y_CC])
    shiftSearchArea = np.asarray([shiftSearchFromCenter_x, shiftSearchFromCenter_y])
    
    #loop through images
    img_nbr_tracking = frameCount
    while img_nbr_tracking < frameCount+FT_forNthNberFrames:
        #read images
        templateImg = cv2.imread(dir_imgs + img_list[img_nbr_tracking], 0)
        searchImg = cv2.imread(dir_imgs + img_list[img_nbr_tracking+TrackEveryNthFrame], 0)
        
        print('template image: ' + img_list[img_nbr_tracking] + ', search image: ' + 
                      img_list[img_nbr_tracking+TrackEveryNthFrame] + '\n')
        
        #track features per image sequence        
        if lk:
            #tracking (matching templates) with Lucas Kanade
            try:
                #consider knowledge about flow velocity and direction (use shift of search window)
                if initialEstimatesLK == True:
                    featureEstimatesNextFrame = featuresToTrack[:,1:]
                    x_initialGuess, y_initialGuess = featureEstimatesNextFrame[:,0], featureEstimatesNextFrame[:,1]
                    x_initialGuess = x_initialGuess.reshape(x_initialGuess.shape[0],1) + np.ones((featureEstimatesNextFrame.shape[0],1)) * shiftSearchFromCenter_x
                    y_initialGuess = y_initialGuess.reshape(y_initialGuess.shape[0],1) + np.ones((featureEstimatesNextFrame.shape[0],1)) * shiftSearchFromCenter_y
                    featureEstimatesNextFrame = np.hstack((x_initialGuess, y_initialGuess))
                #...or not
                else:
                    featureEstimatesNextFrame = None
                
                #perform tracking
                trackedFeaturesLK, status = trackF.performFeatureTrackingLK(templateImg, searchImg, featuresToTrack[:,1:],
                                                                            initialEstimatesLK, featureEstimatesNextFrame,
                                                                            search_area_x_CC, search_area_y_CC)
                
                featuresId = featuresToTrack[:,0]
                trackedFeaturesLKFiltered = np.hstack((featuresId.reshape(featuresId.shape[0],1), trackedFeaturesLK))
                trackedFeaturesLKFiltered = np.hstack((trackedFeaturesLKFiltered, status))
                trackedFeaturesLK_px = trackedFeaturesLKFiltered[~np.all(trackedFeaturesLKFiltered == 0, axis=1)]
                
                trackedFeatures = trackedFeaturesLK_px[:,0:3]
                
                #undistort tracked feature measurement
                trackedFeature_undist = photogrF.undistort_img_coos(trackedFeaturesLK_px[:,1:3], interior_orient)
                trackedFeature_undist_px = photogrF.metric_to_pixel(trackedFeature_undist, interior_orient.resolution_x, interior_orient.resolution_y, 
                                                                    interior_orient.sensor_size_x, interior_orient.sensor_size_y)    

                frameName = np.asarray([img_list[img_nbr_tracking+TrackEveryNthFrame] for x in range(trackedFeaturesLK_px.shape[0])])
                trackedFeaturesOutput_undistArr = np.hstack((frameName, trackedFeaturesLK_px[:,0]))
                trackedFeaturesOutput_undistArr = np.hstack((trackedFeaturesOutput_undistArr, trackedFeature_undist_px[:,0]))
                trackedFeaturesOutput_undistArr = np.hstack((trackedFeaturesOutput_undistArr, trackedFeature_undist_px[:,1]))
                trackedFeaturesOutput_undistArr = trackedFeaturesOutput_undistArr.reshape(4, frameName.shape[0]).T
                
                trackedFeaturesOutput_undist.extend(trackedFeaturesOutput_undistArr)          
                
            except Exception as e:
                print(e)
                print('stopped tracking features with LK after frame ' + img_list[img_nbr_tracking])              
        
        else:
            #tracking (matching templates) with NCC
            trackedFeatures = []     
            for featureToTrack in featuresToTrack:
                
                try:
                    #perform tracking
                    trackedFeature_px = trackF.performFeatureTracking(template_size, search_area, featureToTrack[1:], templateImg, searchImg, 
                                                                      shiftSearchArea, performLSM, lsmBuffer, threshLSM, subpixel, False)
                    trackedFeatures.append([featureToTrack[0], trackedFeature_px[0], trackedFeature_px[1]])
                    
                    #undistort tracked feature measurement
                    trackedFeature_undist = photogrF.undistort_img_coos(trackedFeature_px.reshape(1,2), interior_orient)
                    trackedFeature_undist_px = photogrF.metric_to_pixel(trackedFeature_undist, interior_orient.resolution_x, interior_orient.resolution_y, 
                                                                        interior_orient.sensor_size_x, interior_orient.sensor_size_y)    
                    trackedFeaturesOutput_undist.append([img_list[img_nbr_tracking+TrackEveryNthFrame], int(featureToTrack[0]), 
                                                         trackedFeature_undist_px[0,0], trackedFeature_undist_px[0,1]])
                    
                except Exception as e:
                    print('stopped tracking feature ' + str(featureToTrack[0]) + ' after frame ' 
                          + img_list[img_nbr_tracking] + '\n')  
    
            trackedFeatures = np.asarray(trackedFeatures)
                
        print('nbr of tracked features: ' + str(trackedFeatures.shape[0]) + '\n')
    
        #for visualization of tracked features in gif
        featuers_end, featuers_start, _ = drawF.assignPtsBasedOnID(trackedFeatures, featuresToTrack)    
        arrowsImg = drawF.drawArrowsOntoImg(templateImg, featuers_start, featuers_end)                                
        
        if save_gif:
            arrowsImg.savefig(directoryOutput + 'temppFT.jpg', dpi=150, pad_inches=0)
            imagesForGif.append(cv2.imread(directoryOutput + 'temppFT.jpg')) 
        else:
            arrowsImg.savefig(directoryOutput + 'temppFT' + str(frameCount) + '.jpg', dpi=150, pad_inches=0)           
        arrowsImg.close()
        del arrowsImg
                            
        featuresToTrack = trackedFeatures
    
        img_nbr_tracking = img_nbr_tracking + TrackEveryNthFrame
        
    return trackedFeaturesOutput_undist, imagesForGif


def FilterTracks(trackedFeaturesOutput_undist, dir_imgs, img_list, directoryOutput,
                 minDistance_px, maxDistance_px, minimumTrackedFeatures, 
                 threshAngleSteadiness, threshAngleRange,
                 binNbrMainflowdirection, MainFlowAngleBuffer):
    #filter tracks considering several filter parameters
    trackedFeaturesOutput_undist = np.asarray(trackedFeaturesOutput_undist)
    trackedFeaturesOutput_undist = np.asarray(trackedFeaturesOutput_undist[:,1:4], dtype=np.float)
    featureIDs_fromTracking = np.unique(trackedFeaturesOutput_undist[:,0])
    Features_px = np.empty((1,6))
    
    for feature in featureIDs_fromTracking:
        processFeature = trackedFeaturesOutput_undist[trackedFeaturesOutput_undist[:,0] == feature, 1:3]
        
        #get distance between tracked features across subsequent frames in image space
        xy_start_tr = processFeature[:-1,:]
        xy_tr = processFeature[1:,:]        
        dist = np.sqrt(np.square(xy_start_tr[:,0] - xy_tr[:,0]) + (np.square(xy_start_tr[:,1] - xy_tr[:,1])))
        
        feature_px = np.hstack((np.ones((xy_start_tr.shape[0],1)) * feature, xy_start_tr))
        feature_px = np.hstack((feature_px, xy_tr))
        feature_px = np.hstack((feature_px, dist.reshape(dist.shape[0],1)))
        
        Features_px = np.vstack((Features_px, feature_px))
    
    Features_px = Features_px[1:,:]

    Features_px = pd.DataFrame({'id' : Features_px[:,0], 'x' : Features_px[:,1], 'y' : Features_px[:,2],
                               'x_tr' : Features_px[:,3], 'y_tr' : Features_px[:,4], 'dist' : Features_px[:,5]})
    
    #draw all tracks from frame to frame into image
    image = cv2.imread(dir_imgs + img_list[0], 0)
    drawF.draw_tracks(Features_px, image, directoryOutput, 'TracksRaw_px.jpg', 'dist', False)   
    print('nbr features prior filtering: ' + str(np.unique(Features_px.id).shape[0]) + '\n')
    nbr_features_raw = np.unique(Features_px.id).shape[0]
       
    #minimum tracking distance 
    filteredFeatures_id = Features_px[Features_px.dist < minDistance_px]
    filteredFeatures_id = filteredFeatures_id.id.unique()
    filteredFeatures = Features_px[~Features_px.id.isin(filteredFeatures_id)]
    filteredFeatures = filteredFeatures.reset_index(drop=True)
    drawF.draw_tracks(filteredFeatures, image, directoryOutput, 'TracksFilteredMinDist.png', 'dist', True)
    print('nbr features after minimum distance filter: ' + str(np.unique(filteredFeatures.id).shape[0]))
    nbr_features_mindist = np.unique(filteredFeatures.id).shape[0]
        
    #maximum tracking distance
    filteredFeatures_id = Features_px[Features_px.dist > maxDistance_px]
    filteredFeatures_id = filteredFeatures_id.id.unique()
    filteredFeatures = filteredFeatures[~filteredFeatures.id.isin(filteredFeatures_id)]
    filteredFeatures = filteredFeatures.reset_index(drop=True)
    drawF.draw_tracks(filteredFeatures, image, directoryOutput, 'TracksFilteredMaxDist.png', 'dist', True)
    print('nbr features after maximum distance filter: ' + str(np.unique(filteredFeatures.id).shape[0]))
    nbr_features_maxdist = np.unique(filteredFeatures.id).shape[0]
              
    #minimum tracking counts
    try:
        filteredFeatures = filterF.TrackFilterMinCount(filteredFeatures, minimumTrackedFeatures)
        drawF.draw_tracks(filteredFeatures, image, directoryOutput, 'TracksFilteredMinCount.png', 'dist', True)
        print('nbr features after minimum count filter: ' + str(np.unique(filteredFeatures.id).shape[0]))
    except:
        print('error during filtering minimum count')
    nbr_features_mincount = np.unique(filteredFeatures.id).shape[0]
             
    #get track vector for each tracked feature
    y_track = filteredFeatures.y_tr.values - filteredFeatures.y.values
    x_track = filteredFeatures.x_tr.values - filteredFeatures.x.values
    track = np.hstack((x_track.reshape(filteredFeatures.shape[0],1), y_track.reshape(filteredFeatures.shape[0],1)))
    #get angle for each track
    angle = filterF.angleBetweenVecAndXaxis(track)    
    filteredFeatures['angle'] = pd.Series(angle, index=filteredFeatures.index)
        
    #directional steadiness
    filteredFeatures, steady_angle = filterF.TrackFilterSteadiness(filteredFeatures, threshAngleSteadiness)
    drawF.draw_tracks(filteredFeatures, image, directoryOutput, 'TracksFilteredSteady.png', 'dist', True)
    print('nbr features after steadiness filter: ' + str(np.unique(filteredFeatures.id).shape[0]))
    nbr_features_steady = np.unique(filteredFeatures.id).shape[0]
        
    #range of directions per track
    filteredFeatures, range_angle = filterF.TrackFilterAngleRange(filteredFeatures, threshAngleRange)
    drawF.draw_tracks(filteredFeatures, image, directoryOutput, 'TracksFilteredRangeAngle.png', 'dist', True)
    print('nbr features after range angle filter: ' + str(np.unique(filteredFeatures.id).shape[0]))
    nbr_features_rangeangle = np.unique(filteredFeatures.id).shape[0]
             
    #filter tracks outside main flow direction
    filteredFeatures, flowdir_angle = filterF.TrackFilterMainflowdirection(filteredFeatures, binNbrMainflowdirection, MainFlowAngleBuffer)
    drawF.draw_tracks(filteredFeatures, image, directoryOutput, 'TracksFilteredFlowDir.png', 'dist', True)
    print('nbr features after flow directions filter: ' + str(np.unique(filteredFeatures.id).shape[0]))
    nbr_features_mainflowdir = np.unique(filteredFeatures.id).shape[0]

    #save filter results
    filteredFeatures.to_csv(directoryOutput + 'TracksFiltered_px.txt', sep='\t', index=False)
    
    return filteredFeatures, [nbr_features_raw, nbr_features_mindist, nbr_features_maxdist, nbr_features_mincount,
                              steady_angle, nbr_features_steady, range_angle, nbr_features_rangeangle, flowdir_angle, nbr_features_mainflowdir]


def TracksPx_to_TracksMetric(filteredFeatures, minimumTrackedFeatures, interior_orient, eor_mat, unit_gcp,
                             frame_rate_cam, TrackEveryNthFrame, waterlevel_pt, directoryOutput, dir_imgs, img_list):
    #scale tracks in image space to tracks in object space to get flow velocity in m/s
    waterlevel = waterlevel_pt

    image = cv2.imread(dir_imgs + img_list[0], 0)

    #get first and last position in image space of each tracked feature
    filteredFeatures_1st = filteredFeatures.groupby('id', as_index=False).head(1)   
    filteredFeatures_last = filteredFeatures.groupby('id', as_index=False).tail(1)  
    filteredFeatures_count = np.asarray(filteredFeatures.groupby('id', as_index=False).count())[:,2]  

    xy_start_tr = np.asarray(filteredFeatures_1st[['x', 'y']])
    xy_tr = np.asarray(filteredFeatures_last[['x', 'y']])

    #intersect first and last position with waterlevel
    XY_start_tr = refF.LinePlaneIntersect(xy_start_tr, waterlevel, interior_orient, eor_mat, unit_gcp) / unit_gcp
    XY_tr = refF.LinePlaneIntersect(xy_tr, waterlevel, interior_orient, eor_mat, unit_gcp) / unit_gcp

    filteredFeatures_1st.loc[:,'X'] = pd.Series(XY_start_tr[:,0], index=filteredFeatures_1st.index)
    filteredFeatures_1st.loc[:,'Y'] = pd.Series(XY_start_tr[:,1], index=filteredFeatures_1st.index)
    filteredFeatures_1st.loc[:,'Z'] = pd.Series(XY_start_tr[:,2], index=filteredFeatures_1st.index)
    filteredFeatures_1st.loc[:,'X_tr'] = pd.Series(XY_tr[:,0], index=filteredFeatures_1st.index)
    filteredFeatures_1st.loc[:,'Y_tr'] = pd.Series(XY_tr[:,1], index=filteredFeatures_1st.index)

    #get corresponding distance in object space
    dist_metric = np.sqrt(np.square(XY_start_tr[:,0] - XY_tr[:,0]) + (np.square(XY_start_tr[:,1] - XY_tr[:,1]))) 
    filteredFeatures_1st.loc[:,'dist_metric'] = pd.Series(dist_metric, index=filteredFeatures_1st.index)
    
    filteredFeatures_1st.loc[:,'count'] = pd.Series(filteredFeatures_count, index=filteredFeatures_1st.index)

    #get corresponding temporal observation span
    frame_rate_cam = np.ones((filteredFeatures_count.shape[0],1), dtype=np.float) * frame_rate_cam
    nbrTrackedFrames = TrackEveryNthFrame * filteredFeatures_count
    trackingDuration = nbrTrackedFrames.reshape(frame_rate_cam.shape[0],1) / frame_rate_cam 

    #get velocity
    velo = dist_metric.reshape(trackingDuration.shape[0],1) / trackingDuration
    filteredFeatures_1st.loc[:,'velo'] = pd.Series(velo.flatten(), index=filteredFeatures_1st.index)

    filteredFeatures_1st.loc[:,'x_tr'] = pd.Series(xy_tr[:,0], index=filteredFeatures_1st.index)
    filteredFeatures_1st.loc[:,'y_tr'] = pd.Series(xy_tr[:,1], index=filteredFeatures_1st.index)
    filteredFeatures = filteredFeatures_1st

    #write referenced tracking results to file
    print('nbr of tracked features: ' + str(filteredFeatures.shape[0]) + '\n')
    filteredFeatures.to_csv(directoryOutput + 'TracksReferenced_raw.txt', sep='\t', index=False)
    drawF.draw_tracks(filteredFeatures, image, directoryOutput, 'TracksReferenced_raw.jpg', 'velo', True)

    #filter for outlier velocities
    MeanVeloAll = filteredFeatures.velo.mean()
    StdVeloAll = filteredFeatures.velo.std()
    threshVelo_Pos = MeanVeloAll + 1.5 * StdVeloAll
    threshVelo_Neg = MeanVeloAll - 1.5 * StdVeloAll

    filteredFeatures = filteredFeatures[filteredFeatures.velo < threshVelo_Pos]
    filteredFeatures = filteredFeatures[filteredFeatures.velo > threshVelo_Neg]

    #write filtered tracking results to file and draw final tracking results to image  
    print('nbr of final tracked features: ' + str(filteredFeatures.shape[0]) + '\n')
    filteredFeatures.to_csv(directoryOutput + 'TracksFiltered.txt', sep='\t', index=False)
    drawF.draw_tracks(filteredFeatures, image, directoryOutput, 'TracksFiltered.jpg', 'velo', True)
