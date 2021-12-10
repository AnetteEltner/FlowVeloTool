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
                     directoryOutput, unitAngles='radians'):
    try:
        #read object coordinates of GCP (including point ID)
        gcpObjPts_table = np.asarray(pd.read_table(gcpCoo_file, header=None))   #, delimiter='\t'
    except:
        print('failed reading GCP file (object space)')
            
    try:
        #read pixel coordinates of image points of GCPs (including ID)
        gcpImgPts_table = np.asarray(pd.read_table(imgCoo_GCP_file, header=None))   #, delimiter='\t'
    except:
        print('failed reading GCP file (imgage space)')

    try:
        gcpPts_ids = gcpImgPts_table[:,0]
        gcpPts_ids = gcpPts_ids.reshape(gcpPts_ids.shape[0],1)
        gcpImgPts_to_undist = gcpImgPts_table[:,1:3]

        #undistort image measurements of GCP
        gcpImgPts_undist = photogrF.undistort_img_coos(gcpImgPts_to_undist, interior_orient, False)
        gcpImgPts_undist = np.hstack((gcpPts_ids, gcpImgPts_undist))
    except:
        print('failed undistorting GCP image measurements')
            
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
            rot_mat = photogrF.rot_Matrix(angles_eor[0], angles_eor[1], angles_eor[2], unitAngles).T
            rot_mat = rot_mat * np.array([[-1,-1,-1],[1,1,1],[-1,-1,-1]])
            
            eor_mat = np.hstack((rot_mat.T, pos_eor)) #if rotation matrix received from opencv transpose rot_mat
            eor_mat = np.vstack((eor_mat, [0,0,0,1]))
            print(eor_mat)
                 
        eor_mat[0:3,3] = eor_mat[0:3,3] * unit_gcp
        
    except Exception as e:        
        print(e)
        print('Referencing image failed\n')
        
    return eor_mat

def getWaterlevelFromContour(contour3D, interior_orient, eor_mat, unit_gcp, directoryOutput):
    contour3D = contour3D * unit_gcp
    contour3DImgVisible = detectF.defineFeatureSearchArea(contour3D, interior_orient, eor_mat, False,
                                                            True, directoryOutput, None, True)
    contour3DImgVisible = contour3DImgVisible / unit_gcp
    waterlevel = np.average(contour3DImgVisible[:,2])

    contour3DImgVisible = pd.DataFrame(contour3DImgVisible)
    contour3DImgVisible.columns = ['X','Y','Z']
    contour3DImgVisible.to_csv(directoryOutput + 'ContourInImgView.txt', sep='\t', index=False)

    print('waterlevel is ' + str(waterlevel))

    return waterlevel

def searchMask(waterlevel_pt, waterlevel_buffer, AoI_file, ptCloud, unit_gcp, interior_orient,
               eor_mat, savePlotData, directoryOutput, img_list, preDefAoI=False, imgContourDraw=None):
    waterlevel = waterlevel_pt - waterlevel_buffer  #read waterlevel
    #use search mask from file
    if preDefAoI:
        try:
            searchMask = detectF.readMaskImg(AoI_file, directoryOutput, imgContourDraw)
        except:
            print('reading search mask file failed')
        
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
    outputFileFD = open(os.path.join(directoryOutput, 'FD_every_' + str(FD_everyIthFrame) + '_' + img_list[frameCount][:-4]) + '.txt', 'w')
    writer = csv.writer(outputFileFD, delimiter="\t")
    writer.writerow(['id','x', 'y'])
    writer.writerows(featuresToTrack)
    outputFileFD.flush()
    outputFileFD.close()
    del writer                
                
    return featuresToTrack, first_loop, feature_ID_max


def FeatureDetectionLSPIV(dir_imgs, img_list, frameCount, pointDistX, pointDistY, searchMask, FD_everyIthFrame,
                          savePlotData, directoryOutput, first_loop, feature_ID_max=None):
    #extract features in search area with pre-defined grid
    featuresToTrack = detectF.LSPIV_features(dir_imgs, img_list[frameCount], searchMask, pointDistX, pointDistY, savePlotData,
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
    outputFileFD = open(os.path.join(directoryOutput, 'FD_every_' + str(FD_everyIthFrame) + '_' + img_list[frameCount][:-4]) + '.txt', 'w')
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
                    lk, initialEstimatesLK, maxDistBackForward_px=1):
    #prepare function input
    template_size = np.asarray([template_width, template_height])
    search_area = np.asarray([search_area_x_CC, search_area_y_CC])
    shiftSearchArea = np.asarray([shiftSearchFromCenter_x, shiftSearchFromCenter_y])
    
    #save initial pixel position of features
    trackedFeatures0_undist = photogrF.undistort_img_coos(featuresToTrack[:,1:3], interior_orient)
    trackedFeatures0_undist_px = photogrF.metric_to_pixel(trackedFeatures0_undist, interior_orient.resolution_x, interior_orient.resolution_y, 
                                                          interior_orient.sensor_size_x, interior_orient.sensor_size_y)        
    frame_name0 = np.asarray([img_list[frameCount] for x in range(featuresToTrack.shape[0])])
    trackedFeaturesOutput_undist0 = np.hstack((frame_name0, featuresToTrack[:,0]))
    trackedFeaturesOutput_undist0 = np.hstack((trackedFeaturesOutput_undist0, trackedFeatures0_undist_px[:,0]))
    trackedFeaturesOutput_undist0 = np.hstack((trackedFeaturesOutput_undist0, trackedFeatures0_undist_px[:,1]))
    trackedFeaturesOutput_undist0 = trackedFeaturesOutput_undist0.reshape(4, frame_name0.shape[0]).T
    trackedFeaturesOutput_undist.extend(trackedFeaturesOutput_undist0) 
    
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
                                                                            template_width, template_height, maxDistBackForward_px)
                
                featuresId = featuresToTrack[:,0]
                trackedFeaturesLKFiltered = np.hstack((featuresId.reshape(featuresId.shape[0],1), trackedFeaturesLK))
                trackedFeaturesLKFiltered = np.hstack((trackedFeaturesLKFiltered, status))
                
                #remove points with erroneous LK tracking (ccheck column 3)
                trackedFeaturesLK_px = trackedFeaturesLKFiltered[~np.all(trackedFeaturesLKFiltered == 0, axis=1)]
                
                #drop rows with nan values (which are features that failed back-forward tracking test)
                trackedFeaturesLK_pxDF = pd.DataFrame(trackedFeaturesLK_px)
                trackedFeaturesLK_pxDF = trackedFeaturesLK_pxDF.dropna()
                trackedFeaturesLK_px = np.asarray(trackedFeaturesLK_pxDF)
                
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
                    
                    #check backwards
                    trackedFeature_pxCheck = trackF.performFeatureTracking(template_size, search_area, trackedFeature_px, searchImg, templateImg, 
                                                                           -1*shiftSearchArea, performLSM, lsmBuffer, threshLSM, subpixel, False)                    
                    #set points that fail backward forward tracking test to nan
                    distBetweenBackForward = abs(featureToTrack[1:]-trackedFeature_pxCheck).reshape(-1, 2).max(-1)
                    if distBetweenBackForward > maxDistBackForward_px:
                        print('feature ' + str(featureToTrack[0]) + ' failed backward test.')
                        x = 1/0
                    
                    #join tracked feature and id of feature
                    trackedFeatures.append([featureToTrack[0], trackedFeature_px[0], trackedFeature_px[1]])
                    
                    #undistort tracked feature measurement
                    trackedFeature_undist = photogrF.undistort_img_coos(trackedFeature_px.reshape(1,2), interior_orient)
                    trackedFeature_undist_px = photogrF.metric_to_pixel(trackedFeature_undist, interior_orient.resolution_x, interior_orient.resolution_y, 
                                                                        interior_orient.sensor_size_x, interior_orient.sensor_size_y)    
                    trackedFeaturesOutput_undist.append([img_list[img_nbr_tracking+TrackEveryNthFrame], int(featureToTrack[0]), 
                                                         trackedFeature_undist_px[0,0], trackedFeature_undist_px[0,1]])
                        
                except:
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


def FilterTracks(trackedFeaturesOutput_undist, img_name, directoryOutput,
                 minDistance_px, maxDistance_px, minimumTrackedFeatures, 
                 threshAngleSteadiness, threshAngleRange,
                 binNbrMainflowdirection, MainFlowAngleBuffer, lspiv):
    '''filter tracks considering several filter parameters'''
    #transform dataframe to numpy array and get feature ids
    trackedFeaturesOutput_undist = np.asarray(trackedFeaturesOutput_undist)
    trackedFeaturesOutput_undist = np.asarray(trackedFeaturesOutput_undist[:,1:4], dtype=np.float)
    featureIDs_fromTracking = np.unique(trackedFeaturesOutput_undist[:,0])
    Features_px = np.empty((1,6))
    
    for feature in featureIDs_fromTracking:
        processFeature = trackedFeaturesOutput_undist[trackedFeaturesOutput_undist[:,0] == feature, 1:3]
        
        #get distance between tracked features across subsequent frames in image space
        if lspiv:
            xy_start_tr = np.ones((processFeature.shape[0]-1,processFeature.shape[1])) * processFeature[0,:]
        else:
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
    image = cv2.imread(img_name, 0)
    drawF.draw_tracks(Features_px, image, directoryOutput, 'TracksRaw_px.jpg', 'dist', False)   
    print('nbr features prior filtering: ' + str(np.unique(Features_px.id).shape[0]) + '\n')
    nbr_features_raw = np.unique(Features_px.id).shape[0]
       
    #minimum tracking distance
    if lspiv:
        filteredFeatures = Features_px[Features_px.dist > minDistance_px]
    else:
        filteredFeatures_id = Features_px[Features_px.dist < minDistance_px]
        filteredFeatures_id = filteredFeatures_id.id.unique()
        filteredFeatures = Features_px[~Features_px.id.isin(filteredFeatures_id)]
    filteredFeatures = filteredFeatures.reset_index(drop=True)
    drawF.draw_tracks(filteredFeatures, image, directoryOutput, 'TracksFilteredMinDist.jpg', 'dist', True)
    print('nbr features after minimum distance filter: ' + str(np.unique(filteredFeatures.id).shape[0]))
    nbr_features_mindist = np.unique(filteredFeatures.id).shape[0]
        
    #maximum tracking distance
    if lspiv:
        filteredFeatures = filteredFeatures[filteredFeatures.dist < maxDistance_px]
    else:    
        filteredFeatures_id = Features_px[Features_px.dist > maxDistance_px]
        filteredFeatures_id = filteredFeatures_id.id.unique()
        filteredFeatures = filteredFeatures[~filteredFeatures.id.isin(filteredFeatures_id)]
    filteredFeatures = filteredFeatures.reset_index(drop=True)
    drawF.draw_tracks(filteredFeatures, image, directoryOutput, 'TracksFilteredMaxDist.jpg', 'dist', True)
    print('nbr features after maximum distance filter: ' + str(np.unique(filteredFeatures.id).shape[0]))
    nbr_features_maxdist = np.unique(filteredFeatures.id).shape[0]
              
    #minimum tracking counts
    try:
        filteredFeatures = filterF.TrackFilterMinCount(filteredFeatures, minimumTrackedFeatures)
        drawF.draw_tracks(filteredFeatures, image, directoryOutput, 'TracksFilteredMinCount.jpg', 'dist', True)
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
    drawF.draw_tracks(filteredFeatures, image, directoryOutput, 'TracksFilteredSteady.jpg', 'dist', True)
    print('nbr features after steadiness filter: ' + str(np.unique(filteredFeatures.id).shape[0]))
    nbr_features_steady = np.unique(filteredFeatures.id).shape[0]
        
    #range of directions per track
    filteredFeatures, range_angle = filterF.TrackFilterAngleRange(filteredFeatures, threshAngleRange)
    drawF.draw_tracks(filteredFeatures, image, directoryOutput, 'TracksFilteredRangeAngle.jpg', 'dist', True)
    print('nbr features after range angle filter: ' + str(np.unique(filteredFeatures.id).shape[0]))
    nbr_features_rangeangle = np.unique(filteredFeatures.id).shape[0]
             
    #filter tracks outside main flow direction
    filteredFeatures, flowdir_angle = filterF.TrackFilterMainflowdirection(filteredFeatures, binNbrMainflowdirection, MainFlowAngleBuffer)
    drawF.draw_tracks(filteredFeatures, image, directoryOutput, 'TracksFilteredFlowDir.jpg', 'dist', True)
    print('nbr features after flow directions filter: ' + str(np.unique(filteredFeatures.id).shape[0]))
    nbr_features_mainflowdir = np.unique(filteredFeatures.id).shape[0]        

    #save filter results
    filteredFeatures.to_csv(directoryOutput + 'TracksFiltered_px.txt', sep='\t', index=False)
    
    return filteredFeatures, [nbr_features_raw, nbr_features_mindist, nbr_features_maxdist, nbr_features_mincount,
                              steady_angle, nbr_features_steady, range_angle, nbr_features_rangeangle, flowdir_angle, nbr_features_mainflowdir]


def TracksPx_to_TracksMetric(filteredFeatures, interior_orient, eor_mat, unit_gcp,
                             frame_rate_cam, TrackEveryNthFrame, waterlevel_pt, directoryOutput, img_name, 
                             veloStdThresh, lspiv, cellsizeFilter, searchMask, NN_nbr=10, scaleInfo=None):
    #scale tracks in image space to tracks in object space to get flow velocity in m/s
    if cellsizeFilter == 0:
        cellsizeFilter = 1
        print('reset cell size of filter to 1 because 0 not possible')

    waterlevel = waterlevel_pt

    image = cv2.imread(img_name, 0)

    if lspiv:
        xy_start_tr = np.asarray(filteredFeatures[['x','y']])
        xy_tr = np.asarray(filteredFeatures[['x_tr', 'y_tr']])
        id_features = np.asarray(filteredFeatures['id'])        
        
    else:
        #get first and last position in image space of each tracked feature
        filteredFeatures_1st = filteredFeatures.groupby('id', as_index=False).head(1)   
        filteredFeatures_last = filteredFeatures.groupby('id', as_index=False).tail(1)  
        filteredFeatures_count = np.asarray(filteredFeatures.groupby('id', as_index=False).count())[:,2]  

        xy_start_tr = np.asarray(filteredFeatures_1st[['x', 'y']])
        xy_tr = np.asarray(filteredFeatures_last[['x_tr', 'y_tr']])

    if scaleInfo:
        #get scaled distance
        dist_metric = np.sqrt(np.square(xy_start_tr[:, 0] - xy_tr[:, 0]) + (np.square(xy_start_tr[:, 1] - xy_tr[:, 1]))) * scaleInfo
        XY_start_tr = np.zeros((xy_start_tr.shape[0],3))
        XY_tr = np.zeros(xy_tr.shape)
    else:
        #intersect first and last position with waterlevel
        XY_start_tr = refF.LinePlaneIntersect(xy_start_tr, waterlevel, interior_orient, eor_mat, unit_gcp) / unit_gcp
        XY_tr = refF.LinePlaneIntersect(xy_tr, waterlevel, interior_orient, eor_mat, unit_gcp) / unit_gcp

        #get corresponding distance in object space
        dist_metric = np.sqrt(np.square(XY_start_tr[:,0] - XY_tr[:,0]) + (np.square(XY_start_tr[:,1] - XY_tr[:,1])))

    #get corresponding temporal observation span
    if lspiv:
        trackingDuration = np.ones((id_features.shape[0],1), dtype=np.float) * TrackEveryNthFrame / np.float(frame_rate_cam)
    else:
        frame_rate_cam = np.ones((filteredFeatures_count.shape[0],1), dtype=np.float) * np.float(frame_rate_cam)
        nbrTrackedFrames = TrackEveryNthFrame * (filteredFeatures_count+1)
        trackingDuration = nbrTrackedFrames.reshape(frame_rate_cam.shape[0],1) / frame_rate_cam

    #get velocity
    velo = dist_metric.reshape(trackingDuration.shape[0],1) / trackingDuration
    
    if lspiv:
        filteredFeaturesPIV = pd.DataFrame(id_features, columns=['id'])        
        filteredFeaturesPIV = filterFeatureOrganise(filteredFeaturesPIV, XY_start_tr, XY_tr, xy_tr, dist_metric, velo, 
                                                    False, None, filteredFeatures[['x','y']])
        drawF.draw_tracks(filteredFeaturesPIV.groupby('id', as_index=False).mean(), image, directoryOutput, 'TracksReferenced_raw_PIV.jpg', 'velo', True)
        filteredFeaturesPIVRawOut = filteredFeaturesPIV[['X','Y','Z','velo','dist_metric']]
        filteredFeaturesPIVRawOut.columns = ['X','Y','Z','velo','dist']
        filteredFeaturesPIVRawOut.rename(columns={"dist_metric": "dist"})
        filteredFeaturesPIVRawOut.to_csv(directoryOutput + 'TracksReferenced_raw_PIV.txt', sep='\t', index=False)
        del filteredFeaturesPIVRawOut
               
        #write referenced tracking results to file
        print('nbr of tracked features: ' + str(filteredFeatures.shape[0]) + '\n')
        
        #filter outliers considering mean and std dev for each grid cell
        filteredFeaturesPIV.loc[:,'veloMean'] = pd.Series(np.empty((len(filteredFeaturesPIV))))        
        filteredFeaturesPIV.loc[:,'veloStd'] = pd.Series(np.empty((len(filteredFeaturesPIV))))
        filteredFeatureMean = filteredFeaturesPIV.groupby('id', as_index=False).velo.mean()
        filteredFeaturesStd = filteredFeaturesPIV.groupby('id', as_index=False).velo.std()
        filteredFeaturesPIV["id_copy"] = filteredFeaturesPIV.id
        filteredFeaturesId = filteredFeaturesPIV.groupby('id_copy', as_index=False).id.mean()
        filteredFeaturesId = filteredFeaturesId.id
        
        featureCount = 0
        while featureCount < len(filteredFeaturesId)-1:
            filteredFeaturesPIV.loc[filteredFeaturesPIV.id == filteredFeaturesId.loc[featureCount], 'veloMean'] = filteredFeatureMean.loc[featureCount,'velo']
            filteredFeaturesPIV.loc[filteredFeaturesPIV.id == filteredFeaturesId.loc[featureCount], 'veloStd'] = filteredFeaturesStd.loc[featureCount,'velo']
            featureCount = featureCount + 1
 
        filteredFeaturesPIV.loc[:,'threshPos'] = filteredFeaturesPIV.veloMean + veloStdThresh * filteredFeaturesPIV.veloStd
        filteredFeaturesPIV.loc[:,'threshNeg'] = filteredFeaturesPIV.veloMean - veloStdThresh * filteredFeaturesPIV.veloStd
        filteredFeaturesPIV = filteredFeaturesPIV[filteredFeaturesPIV.velo < filteredFeaturesPIV.threshPos]
        filteredFeaturesPIV = filteredFeaturesPIV[filteredFeaturesPIV.velo > filteredFeaturesPIV.threshNeg]     

        filteredFeaturesPIV_grouped = filteredFeaturesPIV.groupby('id', as_index=False).mean()
        filteredFeaturesCount = filteredFeaturesPIV.groupby('id', as_index=False).count()
        filteredFeaturesPIV_grouped.loc[:,'count'] = filteredFeaturesCount.loc[:,'velo']
        filteredFeaturesPIV_grouped = filteredFeaturesPIV_grouped.drop(columns=['veloMean','veloStd','threshPos','threshNeg'])
        
        filteredFeatures = filteredFeaturesPIV_grouped
        
        drawF.draw_tracks(filteredFeatures, image, directoryOutput, 'TracksFiltered_PIV.jpg', 'velo', True)
        filteredFeaturesPIVOut = filteredFeatures[['X','Y','Z','velo','dist_metric']]
        filteredFeaturesPIVOut.columns = ['X','Y','Z','velo','dist']
        filteredFeaturesPIVOut.rename(columns={"dist_metric": "dist"})
        filteredFeaturesPIVOut.to_csv(directoryOutput + 'TracksFiltered_PIV.txt', sep='\t', index=False)
        del filteredFeaturesPIVOut
        
    else:
        filteredFeatures_1st = filterFeatureOrganise(filteredFeatures_1st, XY_start_tr, XY_tr, xy_tr, dist_metric, velo, 
                                                     True, filteredFeatures_count)
        filteredFeatures = filteredFeatures_1st.copy()
        filteredFeatures = filteredFeatures.reset_index(drop=True)
        filteredFeaturesRawPTVOut = filteredFeatures[['X','Y','Z','velo','dist_metric','count']]
        filteredFeaturesRawPTVOut.columns = ['X','Y','Z','velo','dist','count']
        filteredFeaturesRawPTVOut.to_csv(directoryOutput + 'TracksReferenced_raw_PTV.txt', sep='\t', index=False)
        del filteredFeaturesRawPTVOut
        drawF.draw_tracks(filteredFeatures, image, directoryOutput, 'TracksReferenced_raw_PTV.jpg', 'velo', True)
        
        #write referenced tracking results to file
        print('nbr of tracked features: ' + str(filteredFeatures.shape[0]) + '\n')        

        ''''''
        #filter outliers area-based with regular raster
        xy_cell = filterF.DefineRFeatures_forRasterbasedFilter(image, searchMask, cellsizeFilter*4)
        XYZ_cell = refF.LinePlaneIntersect(xy_cell, waterlevel, interior_orient, eor_mat, unit_gcp) / unit_gcp
        XYZxy = pd.DataFrame(np.hstack((XYZ_cell, xy_cell)))
        XYZxy.columns = ['X','Y','Z','x','y']
        XYZxy['id'] = range(1, len(XYZxy) + 1)

        filteredFeaturesRaster = filterF.NN_filter(filteredFeatures, XYZxy, cellsizeFilter*4, False)
        filteredFeaturesRaster = filteredFeaturesRaster.reset_index(drop=True)
        del XYZxy

        drawF.draw_tracks(filteredFeaturesRaster, image, directoryOutput, 'filteredTracks_raster_PTV.jpg', 'velo', True)
        filteredFeaturesRaster = filteredFeaturesRaster[['X', 'Y', 'Z', 'velo', 'dist', 'count']]
        filteredFeaturesRaster.to_csv(directoryOutput + 'tracksFiltered_rasterBased_PTV.txt', sep='\t', index=False)
        del filteredFeaturesRaster

        #filter outliers area-based with features in proximity
        filteredFeaturesLocally = filterF.NN_filter(filteredFeatures, filteredFeatures, cellsizeFilter, False)
        drawF.draw_tracks(filteredFeaturesLocally, image, directoryOutput, 'filteredTracks_locally_PTV.jpg', 'velo', True)
        filteredFeaturesLocally = filteredFeaturesLocally[['X','Y','Z','velo','dist','count']]
        filteredFeaturesLocally.to_csv(directoryOutput + 'tracksFiltered_locally_PTV.txt', sep='\t', index=False)

        #filter gaussian
        filterGaussian = filterF.filterGaussian(filteredFeatures, searchRadius=cellsizeFilter, NN_nbr=NN_nbr)
        drawF.draw_tracks(filterGaussian, image, directoryOutput, 'filteredTracks_Gaussian_PTV.jpg', 'velo', True)
        filterGaussian = filterGaussian[['X', 'Y', 'Z', 'velo', 'dist', 'count']]
        filterGaussian.to_csv(directoryOutput + 'tracksFiltered_Gaussian_PTV.txt', sep='\t', index=False)

    #filter for outlier velocities using velocity threshold
    MeanVeloAll = filteredFeatures.velo.mean()
    StdVeloAll = filteredFeatures.velo.std()
    threshVelo_Pos = MeanVeloAll + veloStdThresh * StdVeloAll
    threshVelo_Neg = MeanVeloAll - veloStdThresh * StdVeloAll

    filteredFeatures = filteredFeatures[filteredFeatures.velo < threshVelo_Pos]
    filteredFeatures = filteredFeatures[filteredFeatures.velo > threshVelo_Neg]

    drawF.draw_tracks(filteredFeatures, image, directoryOutput, 'TracksFiltered_PTV_VeloThresh.jpg', 'velo', True)
    filteredFeatures = filteredFeatures[['X','Y','Z','velo','dist_metric','count']]
    filteredFeatures.columns = ['X', 'Y', 'Z', 'velo', 'dist', 'count']
    filteredFeatures.rename(columns={"dist_metric": "dist"})
    filteredFeatures.to_csv(directoryOutput + 'TracksFiltered_PTV_VeloThresh.txt', sep='\t', index=False)

    #write filtered tracking results to file and draw final tracking results to image  
    print('nbr of final tracked features: ' + str(filteredFeatures.shape[0]) + '\n')


def filterFeatureOrganise(organizeDataframe, XYZ, XtrYtr, xytr, distMetric, Velo, 
                          ptv=True, count=None, xy=None):    
    organizeDataframe.loc[:,'X'] = pd.Series(XYZ[:,0], index=organizeDataframe.index)
    organizeDataframe.loc[:,'Y'] = pd.Series(XYZ[:,1], index=organizeDataframe.index)
    organizeDataframe.loc[:,'Z'] = pd.Series(XYZ[:,2], index=organizeDataframe.index)
    organizeDataframe.loc[:,'X_tr'] = pd.Series(XtrYtr[:,0], index=organizeDataframe.index)
    organizeDataframe.loc[:,'Y_tr'] = pd.Series(XtrYtr[:,1], index=organizeDataframe.index)      

    organizeDataframe.loc[:,'velo'] = pd.Series(Velo.flatten(), index=organizeDataframe.index)
    organizeDataframe.loc[:,'dist_metric'] = pd.Series(distMetric, index=organizeDataframe.index)
    
    if ptv:
        organizeDataframe.loc[:,'count'] = pd.Series(count, index=organizeDataframe.index)
    else:
        organizeDataframe.loc[:,'x'] = xy.loc[:,'x']
        organizeDataframe.loc[:,'y'] = xy.loc[:,'y']
 
    organizeDataframe.loc[:,'x_tr'] = pd.Series(xytr[:,0], index=organizeDataframe.index)
    organizeDataframe.loc[:,'y_tr'] = pd.Series(xytr[:,1], index=organizeDataframe.index)
    
    return organizeDataframe
