import os, sys, csv
import numpy as np
import pandas as pd
import cv2
import imageio


# dir_library_extr = '/home/eltner/workspace/eclipse/project_eclipse/python_linux/extruso'
# dir_library_general = '/home/eltner/workspace/eclipse/project_eclipse/python_linux/python_scripts/'



import featureTracking_functions as trackF
import photogrammetry_functions as photogrF
import featureDetect_functions as detectF
import coregistration_functions as coregF
import draw_functions as drawF
import featureFilter_functions as filterF
import featureReference_functions as refF
import input_output_functions as ioF

print "OpenCV version :  {0}".format(cv2.__version__)


'''-------set parameters-------'''
#params input data
main_dir = '/media/eltner/Volume/EXTRUSO/Fliessgeschwindigkeit/steady_cams/Wesenitz/video_2017_03-31_lastVid/1200D1/'
log_file_dir = main_dir

dir_imgs = main_dir + 'framesBig/'

# templateCoo_init_file = '/media/eltner/Volume/EXTRUSO/Fliessgeschwindigkeit/raspberry/Wesenitz/video_2017-12-15_14-00_1200D2-2166MOV/test_lsm.txt'

waterlevel_file = main_dir + 'waterlevel.txt'

imgCoo_GCP_file = main_dir + 'gcp_img_coo.txt'
gcpCoo_file = main_dir + 'gcp_obj_coo.txt'

orientFromFile = False
exteriorFromFileTxt = main_dir + "AccuracyCamOrient.txt"

ior_same_for_frame = False
ior_file = [main_dir, 'elbers_310317_d1200_1_16-9.ior'] #_16-9
ior_file_frame = [main_dir, 'elbers_310317_d1200_1_16-9.ior'] #_16-9
if ior_same_for_frame:
    ior_file_frame = ior_file

ptCloud_file = main_dir + 'uav_shore_underwater_310317_5cm.txt'


#params output data
directoryOutput = main_dir + 'results/'


test_run = True
filter_only = False

#params exterior orientation estimation
estimate_exterior = True
if orientFromFile:
    estimate_exterior = False
ransacApprox = True

pos_eor = [197092, 196402,  204942] 
angles_eor = [-1.11312825569, 0.56344209224, -0.31629768702] #in radians

unit_gcp = 1000     #mm
max_orientation_deviation = 1
if ransacApprox:
    exteriorApprox=np.asarray([0,0,0,0,0,0]).reshape(6,1)
else:
    exteriorApprox=np.asarray([6.95180241e+03, -5.37640890e+03, 6.61533628e+03,
                               1.06712046e+00, -4.47911502e-01, 2.99475921e+00]).reshape(6,1)  


#params 3D point cloud
ptCloud_separator = ','

#params search area definition
waterlevel_buffer = 0 #0.3

#params feature detection
minimumThreshBrightness = 100   #135
neighborSearchRadius_FD = 50    #50
maximumNeighbors_FD = 10
maxFtNbr_FD = 3000
sensitiveFD = 0.002 #0.02
maxNN_FD = 0   #in pixel

#params tracking
threshLSM = 0.01 #0.02 #0.001  #for adjustment
lsmBuffer = 1 #increases lsm search compared to patch
template_width = 6 #has to be even    10
template_height = 6    #10
search_area_x_CC = np.int(2 * template_width)  #3    1.5
search_area_y_CC = np.int(2 * template_height)  #3    1.5
shiftSearchFromCenter_x = -2
shiftSearchFromCenter_y = 2

subpixel=True
performLSM = False
plotData = False
savePlotData = True

#params iterations
FD_everyIthFrame = 15
FT_forNthNberFrames = 20    #30
TrackEveryNthFrame = 1 #2  #3
minimumTrackedFeatures = np.int(FT_forNthNberFrames*0.667/TrackEveryNthFrame) #at least 60% trackable
# RefTrackEveryNth = 5

#params referencing
frame_rate_cam = 30     #Falcon 25
 
#params filter tracks
threshAngleSteadiness = 20 #60 #40  #30
binNbrMainflowdirection = 0
threshAngleRange = 90
MainFlowAngleBuffer =  10 #10 #upstreamElber[40,140]    raspElber: [90,180]
veloStdThresh = 1.5
minDistance_px = 1  #2   #in pixel
maxDistance_px = 10


LK = False
initialEstimatesLK = True
LK_dense = False
lspiv = False

'''-------read data and prepare some for following processing-------'''
# #initial template positions
# templateCoo_init = pd.read_table(templateCoo_init_file, header=None, delimiter='\t')
# templateCoo_init = templateCoo_init.as_matrix().reshape(2,1)


#read waterlevel
waterlevel_table = pd.read_table(waterlevel_file, header=None, delimiter=',')
waterlevel_pts = np.asarray(waterlevel_table)
waterlevel = np.float(waterlevel_pts[0,1]) - waterlevel_buffer


#read interior orientation from file (aicon)
interior_orient = photogrF.read_aicon_ior(ior_file[0], ior_file[1])
interior_orient_frame = photogrF.read_aicon_ior(ior_file_frame[0], ior_file_frame[1])

#read point cloud
pt_cloud_table = pd.read_table(ptCloud_file, header=None, delimiter=ptCloud_separator)
ptCloud = np.asarray(pt_cloud_table)
del pt_cloud_table


#read pixel coordinates of image points of GCPs (including ID)
gcpImgPts_table = pd.read_table(imgCoo_GCP_file, header=None)
gcpImgPts_table = np.asarray(gcpImgPts_table)
gcpPts_ids = gcpImgPts_table[:,0]
gcpPts_ids = gcpPts_ids.reshape(gcpPts_ids.shape[0],1)
gcpImgPts_to_undist = gcpImgPts_table[:,1:3]

 
#undistort image measurements of GCP
gcpImgPts_undist = photogrF.undistort_img_coos(gcpImgPts_to_undist, interior_orient, False)  
gcpImgPts_undist = np.hstack((gcpPts_ids, gcpImgPts_undist))

#read object coordinates of GCP (including point ID)
gcpObjPts_table = pd.read_table(gcpCoo_file, header=None)
gcpObjPts_table = np.asarray(gcpObjPts_table)

#read image names in folder
img_list = []
for img_file in os.listdir(dir_imgs):
    if '.png' in img_file:
        img_list.append(img_file)
img_list = sorted(img_list)


#prepare output
if not os.path.exists(directoryOutput):
    os.system('mkdir ' + directoryOutput)
    
print('all input data read')
print('------------------------------------------')
    
    
'''-------get exterior camera geometry-------'''
#minimise feature search area with point cloud
try:
    if estimate_exterior:
        eor_mat = photogrF.getExteriorCameraGeometry(gcpImgPts_undist, gcpObjPts_table, interior_orient, unit_gcp, max_orientation_deviation, 
                                                     ransacApprox, exteriorApprox, True, directoryOutput)
        eor_mat[0:3,3] = eor_mat[0:3,3] * unit_gcp
    
    elif orientFromFile:
        #read file
        calib_resultsFile = pd.read_csv(exteriorFromFileTxt, sep='\t')
        calib_resultsCam = np.asarray([calib_resultsFile.X[0], calib_resultsFile.Y[0], calib_resultsFile.Z[0],
                                       calib_resultsFile.omega[0], calib_resultsFile.phi[0], calib_resultsFile.kappa[0]])
        eor_mat = photogrF.exteriorFromFile(calib_resultsCam.reshape(calib_resultsCam.shape[0],1))
        
    else:   
        rot_mat = photogrF.rot_Matrix(angles_eor[0], angles_eor[1], angles_eor[2], 'radians').T
        rot_mat = rot_mat * np.array([[-1,-1,-1],[1,1,1],[-1,-1,-1]])
        eor_mat = np.hstack((rot_mat.T, np.array([[pos_eor[0]], [pos_eor[1]], [pos_eor[2]]]))) #if rotation matrix received from opencv transpose rot_mat
        eor_mat = np.vstack((eor_mat, [0,0,0,1]))   
        eor_mat[0:3,3] = eor_mat[0:3,3] #* unit_gcp
        print(eor_mat)
    

    
except Exception as e:
    print(e)
    print('Referencing image failed')
#    continue


'''-------detect and track features-------'''
if not filter_only:
    #select points only below water level to extract river area to search for features
    pointsBelowWater = ptCloud[ptCloud[:,2] < waterlevel] * unit_gcp
    searchMask = detectF.defineFeatureSearchArea(pointsBelowWater, interior_orient_frame, eor_mat, False, savePlotData, directoryOutput, img_list[1])  #xy
    searchMask = np.asarray(searchMask)
    print('search mask with ' + str(searchMask.shape[0]) + ' points defined')
    
    
    frameCount = 0
    imagesForGif = []
    trackedFeaturesOutput_undist = []
    first_loop = True
    
    if test_run:
        lenLoop = 30
    else:
        lenLoop = len(img_list)-FT_forNthNberFrames-1
        
    firstLoopDense = True
    while frameCount < lenLoop:
        
        if frameCount % FD_everyIthFrame == 0:
        
            '''-------perform feature detection-------'''
            if lspiv:
                featuresToTrack = detectF.LSPIV_features(dir_imgs, img_list[frameCount], searchMask, template_width, template_height, savePlotData,
                                                         directoryOutput)
                
            else:
                featuresToTrack = detectF.featureDetection(dir_imgs, img_list[frameCount], searchMask, minimumThreshBrightness, neighborSearchRadius_FD, 
                                                           maximumNeighbors_FD, maxFtNbr_FD, sensitiveFD, savePlotData, directoryOutput, plotData)
            
            feature_ID = np.array(range(featuresToTrack.shape[0]))        
                   
            if first_loop:
                feature_ID_max = featuresToTrack.shape[0] + 1 
                first_loop = False             
    #             #check if feature already existed
    #             featuresToTrack_old = featuresToTrack
            else: 
                feature_ID = feature_ID_max + feature_ID
                feature_ID_max = feature_ID_max + featuresToTrack.shape[0] + 1
    #             featureNN, targetNN = ptv.NN_pts_FD(featuresToTrack_old, featuresToTrack, maxNN_FD)
    
            featuresToTrack_id = np.hstack((feature_ID.reshape(feature_ID.shape[0],1), featuresToTrack[:,1].reshape(featuresToTrack.shape[0],1)))
            featuresToTrack = np.hstack((featuresToTrack_id, featuresToTrack[:,0].reshape(featuresToTrack.shape[0],1)))
    
            
            print('nbr features detected: ' + str(featuresToTrack.shape[0]))
            
            #write detected feature to file
            outputFileFD = open(os.path.join(directoryOutput, 'FD_every_' + str(FD_everyIthFrame) + '_' + img_list[frameCount][:-4]) + '.txt', 'wb')
            writer = csv.writer(outputFileFD, delimiter="\t")
            writer.writerow(['id','x', 'y'])
            writer.writerows(featuresToTrack)
            outputFileFD.flush()
            outputFileFD.close()
            del writer
            
            
            print('features detected')
            print('------------------------------------------')
            
            
            '''-------perform feature tracking-------'''
            #prepare function input
            template_size = np.asarray([template_width, template_height])
            search_area = np.asarray([search_area_x_CC, search_area_y_CC])
            shiftSearchArea = np.asarray([shiftSearchFromCenter_x, shiftSearchFromCenter_y])
            
            #loop through images
            img_nbr_tracking = frameCount
            while img_nbr_tracking < frameCount+FT_forNthNberFrames:
                #read images
                templateImg = cv2.imread(dir_imgs + img_list[img_nbr_tracking], 0)  #loading image in grey scale
                searchImg = cv2.imread(dir_imgs + img_list[img_nbr_tracking+TrackEveryNthFrame], 0) #loading image in grey scale
                
                print('template image: ' + img_list[img_nbr_tracking] + ', search image: ' + img_list[img_nbr_tracking+TrackEveryNthFrame])
                
                #track features per image sequence
                trackedFeatures = []
                
                if LK_dense:
                    
                    flow = trackF.performDenseFeatureTracking(templateImg, searchImg)
                    flow = np.asarray(flow, dtype=np.int)
                    if firstLoopDense:
                        flowMag = flow[:,:,0] #* 10
                        firstLoopDense = False
                    else: 
                        flowMag = flowMag + flow[:,:,0]
                    
                    cv2.imwrite(directoryOutput + 'flow.png', flowMag)
                    
                    img_nbr_tracking = img_nbr_tracking + TrackEveryNthFrame
                    continue
                
                elif LK:
                    try:
                        if initialEstimatesLK == True:
                            featureEstimatesNextFrame = featuresToTrack[:,1:]
                            x_initialGuess, y_initialGuess = featureEstimatesNextFrame[:,0], featureEstimatesNextFrame[:,1]
                            x_initialGuess = x_initialGuess.reshape(x_initialGuess.shape[0],1) + np.ones((featureEstimatesNextFrame.shape[0],1)) * shiftSearchFromCenter_x
                            y_initialGuess = y_initialGuess.reshape(y_initialGuess.shape[0],1) + np.ones((featureEstimatesNextFrame.shape[0],1)) * shiftSearchFromCenter_y
                            featureEstimatesNextFrame = np.hstack((x_initialGuess, y_initialGuess))
                            
                            trackedFeaturesLK, status = trackF.performFeatureTrackingLK(templateImg, searchImg, featuresToTrack[:,1:],
                                                                                        initialEstimatesLK, featureEstimatesNextFrame)
                        else:
                            trackedFeaturesLK, status = trackF.performFeatureTrackingLK(templateImg, searchImg, featuresToTrack[:,1:])
                        
                        featuresId = featuresToTrack[:,0]
                        trackedFeaturesLKFiltered = np.hstack((featuresId.reshape(featuresId.shape[0],1), trackedFeaturesLK))
                        trackedFeaturesLKFiltered = np.hstack((trackedFeaturesLKFiltered, status))
                        trackedFeaturesLK_px = trackedFeaturesLKFiltered[~np.all(trackedFeaturesLKFiltered == 0, axis=1)]
                        
                        trackedFeatures = trackedFeaturesLK_px[:,0:3]
                        
                        #undistort tracked feature measurement
                        trackedFeature_undist = photogrF.undistort_img_coos(trackedFeaturesLK_px[:,1:3], interior_orient_frame)
                        trackedFeature_undist_px = photogrF.metric_to_pixel(trackedFeature_undist, interior_orient_frame.resolution_x, interior_orient_frame.resolution_y, 
                                                                            interior_orient_frame.sensor_size_x, interior_orient_frame.sensor_size_y)    

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
                    for featureToTrack in featuresToTrack:
                        
                        try:
                            trackedFeature_px = trackF.performFeatureTracking(template_size, search_area, featureToTrack[1:], templateImg, searchImg, 
                                                                              shiftSearchArea, performLSM, lsmBuffer, threshLSM, subpixel, False)
                            trackedFeatures.append([featureToTrack[0], trackedFeature_px[0], trackedFeature_px[1]])
                            
                            #undistort tracked feature measurement
                            trackedFeature_undist = photogrF.undistort_img_coos(trackedFeature_px.reshape(1,2), interior_orient_frame)
                            trackedFeature_undist_px = photogrF.metric_to_pixel(trackedFeature_undist, interior_orient_frame.resolution_x, interior_orient_frame.resolution_y, 
                                                                                interior_orient_frame.sensor_size_x, interior_orient_frame.sensor_size_y)    
                            trackedFeaturesOutput_undist.append([img_list[img_nbr_tracking+TrackEveryNthFrame], int(featureToTrack[0]), 
                                                                 trackedFeature_undist_px[0,0], trackedFeature_undist_px[0,1]])
                            
                        except Exception as e:
                            print(e)
                            print('stopped tracking feature ' + str(featureToTrack[0]) + ' after frame ' + img_list[img_nbr_tracking])  
                
                    trackedFeatures = np.asarray(trackedFeatures)
                    
                print('nbr of tracked features: ' + str(trackedFeatures.shape[0]))
            
                #for visualization of tracked features in gif
                featuers_end, featuers_start, _ = drawF.assignPtsBasedOnID(trackedFeatures, featuresToTrack)    
                arrowsImg = drawF.drawArrowsOntoImg(templateImg, featuers_start, featuers_end)              
                arrowsImg.savefig(directoryOutput + 'temppFT.png', dpi=150, pad_inches=0)
    #             arrowsImg.close()
    #             del arrowsImg
                
                imagesForGif.append(cv2.imread(directoryOutput + 'temppFT.png'))
            
                featuresToTrack = trackedFeatures
            
                img_nbr_tracking = img_nbr_tracking + TrackEveryNthFrame
            
        frameCount = frameCount + 1
    
    
    #write tracked features to file
    print('save tracking result to gif')
    outputFileFT = open(os.path.join(directoryOutput, 'Tracking_FT_nbrFrames_' + str(FT_forNthNberFrames) + '_FD_nbrFrames_' + str(FD_everyIthFrame)) + '.txt', 'wb')
    writer = csv.writer(outputFileFT, delimiter="\t")
    writer.writerow(['frame', 'id','x', 'y'])
    writer.writerows(trackedFeaturesOutput_undist)
    outputFileFT.flush()
    outputFileFT.close()
    del writer
    
    #save gif
    imageio.mimsave(directoryOutput + 'trackedFeatures.gif', imagesForGif)
    del imageio
    
    
    print('feature matching done')
    print('------------------------------------------')



'''-------filter tracking results in image space-------'''
if filter_only:
    trackedFeaturesOutput_undist = pd.read_table(directoryOutput + 'Tracking_FT_nbrFrames_' + str(FT_forNthNberFrames) + '_FD_nbrFrames_' + str(FD_everyIthFrame) + '.txt')
    print('file read for filtering')
    
trackedFeaturesOutput_undist = np.asarray(trackedFeaturesOutput_undist)
trackedFeaturesOutput_undist = np.asarray(trackedFeaturesOutput_undist[:,1:4], dtype=np.float)
featureIDs_fromTracking = np.unique(trackedFeaturesOutput_undist[:,0])
Features_px = np.empty((1,6))
for feature in featureIDs_fromTracking:
    processFeature = trackedFeaturesOutput_undist[trackedFeaturesOutput_undist[:,0] == feature, 1:3]
    
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
image = cv2.imread(dir_imgs + img_list[0], 0)
drawF.draw_tracks(Features_px, image, directoryOutput, 'TracksRaw_px.png', 'dist', False)   
print('nbr features prior filtering: ' + str(np.unique(Features_px.id).shape[0]))
nbr_features_raw = np.unique(Features_px.id).shape[0]


#minimum tracking distance 
# filteredFeatures = Features_px[Features_px.dist > minDistance_px]
filteredFeatures_id = Features_px[Features_px.dist <= minDistance_px]
filteredFeatures_id = filteredFeatures_id.id.unique()
filteredFeatures = Features_px[~Features_px.id.isin(filteredFeatures_id)]
filteredFeatures = filteredFeatures.reset_index(drop=True)
try:
    drawF.draw_tracks(filteredFeatures, image, directoryOutput, 'TracksFilteredMinDist.png', 'dist', True)
except:
    print ('drawing minimum distance filtered tracks failed')
print('nbr features after minimum distance filter: ' + str(np.unique(filteredFeatures.id).shape[0]))
nbr_features_mindist = np.unique(filteredFeatures.id).shape[0]


#maximum tracking distance
# filteredFeatures = filteredFeatures[filteredFeatures.dist < maxDistance_px]
filteredFeatures_id = Features_px[Features_px.dist > maxDistance_px]
filteredFeatures_id = filteredFeatures_id.id.unique()
filteredFeatures = filteredFeatures[~filteredFeatures.id.isin(filteredFeatures_id)]
filteredFeatures = filteredFeatures.reset_index(drop=True)
try:
    drawF.draw_tracks(filteredFeatures, image, directoryOutput, 'TracksFilteredMaxDist.png', 'dist', True)
except:
    print ('drawing maximum distance filtered tracks failed')
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


#directional direction steadiness
filteredFeatures, steady_angle = filterF.TrackFilterSteadiness(filteredFeatures, threshAngleSteadiness)
try:
    drawF.draw_tracks(filteredFeatures, image, directoryOutput, 'TracksFilteredSteady.png', 'dist', True)
except:
    print ('drawing direction steadiness filtered tracks failed')
print('nbr features after steadiness filter: ' + str(np.unique(filteredFeatures.id).shape[0]))
nbr_features_steady = np.unique(filteredFeatures.id).shape[0]


#range of directions per track
filteredFeatures, range_angle = filterF.TrackFilterAngleRange(filteredFeatures, threshAngleRange)
try:
    drawF.draw_tracks(filteredFeatures, image, directoryOutput, 'TracksFilteredRangeAngle.png', 'dist', True)
except:
    print ('drawing range directions filtered tracks failed')
print('nbr features after range angle filter: ' + str(np.unique(filteredFeatures.id).shape[0]))
nbr_features_rangeangle = np.unique(filteredFeatures.id).shape[0]

     
#filter tracks outside main flow direction
filteredFeatures, flowdir_angle = filterF.TrackFilterMainflowdirection(filteredFeatures, binNbrMainflowdirection, MainFlowAngleBuffer)
try:
    drawF.draw_tracks(filteredFeatures, image, directoryOutput, 'TracksFilteredFlowDir.png', 'dist', True)
except:
    print ('drawing main flow direction filtered tracks failed')    
print('nbr features after flow directions filter: ' + str(np.unique(filteredFeatures.id).shape[0]))
nbr_features_mainflowdir = np.unique(filteredFeatures.id).shape[0]


#save filter results
filteredFeatures.to_csv(directoryOutput + 'TracksFiltered_px.txt', sep='\t', index=False)


print('filtering tracks done')
print('------------------------------------------')



'''-------transform img measurements into object space-------'''
waterlevel = waterlevel + waterlevel_buffer

# XYZfromSearchMask, searchMask_undist = ptv.getWaterborderXYZ(searchMask, ptCloud * unit_gcp, eor_mat, interior_orient)
# XYZxy = np.hstack((XYZfromSearchMask, searchMask_undist))
# 
# #keep only points close to water level
# XYZxy = XYZxy[XYZxy[:,2] >= waterlevel]
# XY = np.asarray(XYZxy[0:300,0:2], dtype=float)
# xy = np.asarray(XYZxy[0:300,3:5], dtype=float)
# transform_mat = ptv.getTransformationMat(XY, xy)


#reference tracked features
referencedFeatures = np.empty((1,13))
featureIDs_fromTracking = filteredFeatures.id.unique()

# featureRef = 1
# while featureRef < (FT_forNthNberFrames-minimumTrackedFeatures):
    
filteredFeatures_1st = filteredFeatures.groupby('id', as_index=False).head(1)   #featureRef
# filteredFeatures_last = filteredFeatures.groupby('id', as_index=False).nth(minimumTrackedFeatures)  #+featureRef
filteredFeatures_last = filteredFeatures.groupby('id', as_index=False).tail(1)  #+featureRef
filteredFeatures_count = np.asarray(filteredFeatures.groupby('id', as_index=False).count())[:,2]  #+featureRef

xy_start_tr = np.asarray(filteredFeatures_1st[['x', 'y']])
xy_tr = np.asarray(filteredFeatures_last[['x', 'y']])

XY_start_tr = refF.LinePlaneIntersect(xy_start_tr, waterlevel, interior_orient_frame, eor_mat, unit_gcp) / unit_gcp
XY_tr = refF.LinePlaneIntersect(xy_tr, waterlevel, interior_orient_frame, eor_mat, unit_gcp) / unit_gcp

filteredFeatures_1st['X'] = pd.Series(XY_start_tr[:,0], index=filteredFeatures_1st.index)
filteredFeatures_1st['Y'] = pd.Series(XY_start_tr[:,1], index=filteredFeatures_1st.index)
filteredFeatures_1st['Z'] = pd.Series(XY_start_tr[:,2], index=filteredFeatures_1st.index)
filteredFeatures_1st['X_tr'] = pd.Series(XY_tr[:,0], index=filteredFeatures_1st.index)
filteredFeatures_1st['Y_tr'] = pd.Series(XY_tr[:,1], index=filteredFeatures_1st.index)

dist_metric = np.sqrt(np.square(XY_start_tr[:,0] - XY_tr[:,0]) + (np.square(XY_start_tr[:,1] - XY_tr[:,1]))) 
filteredFeatures_1st['dist_metric'] = pd.Series(dist_metric, index=filteredFeatures_1st.index)

filteredFeatures_1st['count'] = pd.Series(filteredFeatures_count, index=filteredFeatures_1st.index)

# frame_rate = frame_rate_cam / np.float((TrackEveryNthFrame * minimumTrackedFeatures))
frame_rate_cam = np.ones((filteredFeatures_count.shape[0],1), dtype=np.float) * frame_rate_cam
counts = TrackEveryNthFrame * filteredFeatures_count
frame_rate = frame_rate_cam / counts.reshape(frame_rate_cam.shape[0],1)
velo = dist_metric.reshape(frame_rate.shape[0],1) / (1/frame_rate)

# velo = dist_metric/(1/np.float(frame_rate))
filteredFeatures_1st['velo'] = pd.Series(velo.flatten(), index=filteredFeatures_1st.index)

filteredFeatures_1st.x_tr = pd.Series(xy_tr[:,0], index=filteredFeatures_1st.index)
filteredFeatures_1st.y_tr = pd.Series(xy_tr[:,1], index=filteredFeatures_1st.index)

#     if featureRef == 1:
#         filteredFeatures_out = filteredFeatures_1st
#     else:
#         filteredFeatures_out = filteredFeatures_out.concat(filteredFeatures_1st)
#     
#     featureRef = RefTrackEveryNth + featureRef

filteredFeatures = filteredFeatures_1st

#write referenced tracking results to file
print('nbr of tracked features: ' + str(filteredFeatures.shape[0]))
filteredFeatures.to_csv(directoryOutput + 'TracksReferenced_raw.txt', sep='\t', index=False)
drawF.draw_tracks(filteredFeatures, image, directoryOutput, 'TracksReferenced_raw.png', 'velo', True)



#filter for outlier velocities (due it twice)
MeanVeloAll = filteredFeatures.velo.mean()
StdVeloAll = filteredFeatures.velo.std()
threshVelo_Pos = MeanVeloAll + 1.5 * StdVeloAll
threshVelo_Neg = MeanVeloAll - 1.5 * StdVeloAll

filteredFeatures = filteredFeatures[filteredFeatures.velo < threshVelo_Pos]
filteredFeatures = filteredFeatures[filteredFeatures.velo > threshVelo_Neg]

MeanVeloAll = filteredFeatures.velo.mean()
StdVeloAll = filteredFeatures.velo.std()
threshVelo_Pos = MeanVeloAll + veloStdThresh * StdVeloAll
threshVelo_Neg = MeanVeloAll - veloStdThresh * StdVeloAll

filteredFeatures = filteredFeatures[filteredFeatures.velo < threshVelo_Pos]
filteredFeatures = filteredFeatures[filteredFeatures.velo > threshVelo_Neg]


# filteredFeatures = ptv.FilteredTracksGroupPerID(filteredFeatures)
# 
# #consider all velocities
# filteredFeatures = filteredFeatures[filteredFeatures.velo_mean < threshVelo_Pos]
# filteredFeatures = filteredFeatures[filteredFeatures.velo_mean > threshVelo_Neg]
# 
# #consider all velocities per feature
# filteredFeatures = filteredFeatures[filteredFeatures.velo_std < veloStdThresh]


#write filtered tracking results to file  
print('nbr of final tracked features: ' + str(filteredFeatures.shape[0]))
filteredFeatures.to_csv(directoryOutput + 'TracksFiltered.txt', sep='\t', index=False)
# filteredFeatures.rename(columns={'velo_mean': 'velo'}, inplace=True)
drawF.draw_tracks(filteredFeatures, image, directoryOutput, 'TracksFiltered.png', 'velo', True)


'''-------logfile-------'''
log_file_wirter, logfile = ioF.logfile_writer(log_file_dir + 'logfile.txt')
log_file_wirter.writerow([['test run: ', test_run],['exterior angles: ', angles_eor],['exterior position: ', pos_eor],
                         ['unit_gcp: ', unit_gcp],['use ransacApprox: ', ransacApprox],
                         ['waterlevel: ',waterlevel],['waterlevel_buffer: ',waterlevel_buffer],
                         ['minimumThreshBrightness: ',minimumThreshBrightness],['neighborSearchRadius_FD: ',neighborSearchRadius_FD],
                         ['maximumNeighbors_FD :',maximumNeighbors_FD],['maxFtNbr_FD :',maxFtNbr_FD],['sensitiveFD: ',sensitiveFD],['maxNN_FD: ',maxNN_FD],
                         ['template_width: ',template_width],['template_height: ',template_height],['search_area_x_CC: ',search_area_x_CC],['search_area_y_CC: ',search_area_y_CC], 
                         ['shiftSearchFromCenter_x: ',shiftSearchFromCenter_x],['shiftSearchFromCenter_y: ',shiftSearchFromCenter_y],
                         ['subpixel: ',subpixel],['performLSM: ',performLSM],['FD_everyIthFrame: ',FD_everyIthFrame],['FT_forNthNberFrames: ',FT_forNthNberFrames],
                         ['TrackEveryNthFrame: ',TrackEveryNthFrame],['frame_rate_cam: ',np.median(frame_rate_cam)],
                         ['minDistance_px: ',minDistance_px],['nbr features min dist: ',nbr_features_mindist],
                         ['maxDistance_px: ',maxDistance_px],['nbr features max dist: ',nbr_features_maxdist],
                         ['minimumTrackedFeatures: ',minimumTrackedFeatures],['nbr features min count: ',nbr_features_mincount],
                         ['threshAngleSteadiness: ',threshAngleSteadiness],['nbr features steadyness: ', nbr_features_steady],['average angle steadiness: ', steady_angle],
                         ['threshAngleRange: ',threshAngleRange],['nbr features angle range: ',nbr_features_rangeangle],['average range angle: ', range_angle],
                         ['binNbrMainflowdirection: ',binNbrMainflowdirection],['MainFlowAngleBuffer: ',MainFlowAngleBuffer],
                         ['nbr features main flow direction: ', nbr_features_mainflowdir],['median angle flow direction: ', flowdir_angle],
                         ['veloStdThresh: ',veloStdThresh],['nbr filtered features: ',filteredFeatures.shape[0]],['nbr raw features: ',nbr_features_raw]])
logfile.flush()
logfile.close()



print('finished')
print('------------------------------------------')





#do track referencing it for each feature separately (in case of outliers)
# for feature in featureIDs_fromTracking:
#     first_iter = True
#     try:
#         processFeature_1st = filteredFeatures_1st[filteredFeatures_1st.id == feature]
#         processFeature_last = filteredFeatures_last[filteredFeatures_last.id == feature]
#         
#         xy_start_tr = [processFeature_1st.x, processFeature_1st.y]
#         xy_tr = np.asarray([processFeature_last.x_tr, processFeature_last.y_tr])
#         
#         XY_start_tr = ptv.LinePlaneIntersect(xy_start_tr, waterlevel, interior_orient, eor_mat, unit_gcp) / unit_gcp
#         XY_tr = ptv.LinePlaneIntersect(xy_tr, waterlevel, interior_orient, eor_mat, unit_gcp) / unit_gcp
#         
#         dist_metric = np.sqrt(np.square(XY_start_tr[:,0] - XY_tr[:,0]) + (np.square(XY_start_tr[:,1] - XY_tr[:,1]))) 
#         processFeature_1st['dist_metric'] = pd.Series(dist_metric, index=processFeature_1st.index)
#         
#         velo = dist_metric/(1/np.float(frame_rate))
#         processFeature_1st['velo'] = pd.Series(velo, index=processFeature_1st.index)
#         
#         if first_iter:
#             first_iter = False
#             process_feature = processFeature_1st
#             continue
#         else:
#             process_feature = pd.concat(processFeature_1st)
# 
#     
#     except Exception as e:
#         print(e)
#         print('feature ' + str(feature) + ' failed to rectify')

