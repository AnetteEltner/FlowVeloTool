#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os, sys
import numpy as np
import pandas as pd
import cv2
import imageio

import coregistration_functions as coregF
import photogrammetry_functions as photogrF
import input_output_functions as ioF
import PTV_functions as ptv

import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from ttk import *


# global imageio

class FlowVeloTool:

    '''input data'''
    def __init__(self):
        print("")

    def EstimateVelocity(self):
        '''-------set parameters-------'''
        # ---> here are the needed input values!
        fileDir = ".../example/"
        directoryOutputMain = ".../YourOutputFolderChoice/"
        ior_file = ".../example/sensorFrame.txt"
        eor_File = ".../example/ExteriorFromMetashape_Kappa(Yaw)Rotated180deg.csv"
        ptCloud_file = ".../example/waterMaskFromSfMPointCloud.txt"
        contour3D = ".../example/contour3D.txt"

        lenFileAdd = 6  #to depict filename part, which infront of actual frame number
        lenFileAddEnd = -23 #to depict filename part for pose retreival, which starts after frame number
        lenFileAddEnd_tailing = -13  # to depict filename part of individual frames for tracking, which starts after frame number
        # <--- that are all new parameters, unless you need to change the parameters


        # params referencing
        frame_rate_cam = 50
        waterlevel_buffer = 0
        unit_gcp = 1000

        # parameters feature detection
        lspiv = False
        lk = True
        initialLK = False
        minimumThreshBrightness = 200
        neighborSearchRadius_FD = 30
        maximumNeighbors_FD = 20
        maxFtNbr_FD = 6000
        sensitiveFD = 0.001
        pointDistX = 200
        pointDistY = 200

        # parameters tracking
        threshLSM = 0.001  # for adjustment
        lsmBuffer = 3  # increases lsm search area compared to patch
        template_width = 10  # has to be odd
        template_height = 10  # has to be odd
        search_area_x_CC = 30
        search_area_y_CC = 30
        shiftSearchFromCenter_x = 0
        shiftSearchFromCenter_y = 8
        subpixel = False

        performLSM = False
        savePlotData = True
        save_gif = True

        # parameters iterations
        FD_everyIthFrame = 6
        FT_forNthNberFrames = 7
        TrackEveryNthFrame = 2

        # params filter tracks
        threshAngleSteadiness = 25
        threshAngleRange = 45
        binNbrMainflowdirection = 0
        MainFlowAngleBuffer = 45
        veloStdThresh = 2
        minDistance_px = 2
        maxDistance_px = 55
        minTrackedFeatures = 66 # values in percentage (%)
        minTrackedFeatures = int(FT_forNthNberFrames * (minTrackedFeatures / 100) / TrackEveryNthFrame)
        veloFilterSize = 50

        gcpCoo_file = None
        imgCoo_GCP_file = None
        AoI_file = None

        '''-------read data and prepare for following processing-------'''
        # read parameters from directories
        try:
            ptCloud = np.asarray(pd.read_csv(ptCloud_file))  # read point cloud
        except:
            print('failed reading point cloud file')

        try:
            interior_orient = photogrF.read_aicon_ior(ior_file)  # read interior orientation from file (aicon)
        except:
            print('failed reading interior orientation file')

        try:
            contour3D = np.asarray(pd.read_csv(contour3D))
        except:
            print('failed reading 3D contour file')

        eor_Table = pd.read_csv(eor_File)

        #go through folders to get images in each folder
        if os.path.isdir(fileDir):
            for dirpath, dirsubpaths, dirfiles in os.walk(fileDir):
                if len(dirsubpaths) >= 1:
                    dir_imgsList = dirsubpaths
                    break
                else:
                    print('empty directory: ' + dirpath)
                    sys.exit()
        else:
            print('directory ' + fileDir + ' not found')
            sys.exit()

        print('all input data read\n')
        print('------------------------------------------')

        for index, frame in eor_Table.iterrows():
            print(frame.id[lenFileAdd:lenFileAddEnd])
            # if frame.id[0:lenFileAdd] == veloFramesSkip:
            #     continue
            # if index < 41:
            #     continue

            for dir_imgs in dir_imgsList:
                if dir_imgs[0:-11] == frame.id[lenFileAdd:lenFileAddEnd]:
                    try:
                        dir_imgs = dirpath + dir_imgs + '\\'

                        try:
                            img_list = ioF.read_imgs_folder(dir_imgs)  # read image names in folder

                            # select only frames to corresponding oriented image
                            img_listUpdated = []
                            for imgfile in img_list:
                                if frame.id[0:lenFileAdd] in imgfile[0:lenFileAdd]:
                                    img_listUpdated.append(imgfile)
                            img_list = sorted(img_listUpdated, key=lambda x: int(x[lenFileAdd:lenFileAddEnd_tailing])) #sort based on frame number

                        except Exception as e:
                            print(e)
                            print('failed reading images from folder')
                            continue

                        # read first image name (including folder) in folder for later visualization
                        img_name = dir_imgs + img_list[0]

                        # prepare output
                        directoryOutput = directoryOutputMain + frame.id[0:lenFileAddEnd] + "_velocities/"
                        if not os.path.exists(directoryOutput):
                            os.makedirs(directoryOutput)

                        '''-------get exterior camera geometry-------'''
                        # parameters exterior orientation estimation
                        angles_eor = np.asarray([frame.pitch, frame.roll, frame.yaw], dtype=np.float64)
                        angles_eor = angles_eor.reshape(3, 1)

                        pos_eor = np.asarray([frame.x, frame.y, frame.z], dtype=np.float64)
                        pos_eor = pos_eor.reshape(3, 1)

                        eor_mat = ptv.EstimateExterior(gcpCoo_file, imgCoo_GCP_file, interior_orient, False,
                                                       unit_gcp, None, False, angles_eor, pos_eor,
                                                       directoryOutput, 'grad')

                        '''define waterlevel from contour'''
                        try:
                            waterlevel_pt = ptv.getWaterlevelFromContour(contour3D, interior_orient, eor_mat, unit_gcp, directoryOutput)
                        except:
                            print('failed waterlevel retreival from contour')

                        '''define search area for features'''
                        searchMask = ptv.searchMask(waterlevel_pt, waterlevel_buffer, AoI_file, ptCloud, unit_gcp,
                                                    interior_orient,
                                                    eor_mat, savePlotData, directoryOutput, img_list, False)


                        '''-------perform feature detection-------'''
                        frameCount = 0
                        imagesForGif = []
                        trackedFeaturesOutput_undist = []
                        first_loop = True

                        lenLoop = len(img_list) - FT_forNthNberFrames - 1

                        if lspiv:
                            featuresToTrack, first_loop, feature_ID_max = ptv.FeatureDetectionLSPIV(dir_imgs, img_list,
                                                                                                    frameCount,
                                                                                                    pointDistX, pointDistY,
                                                                                                    searchMask,
                                                                                                    FD_everyIthFrame,
                                                                                                    savePlotData,
                                                                                                    directoryOutput,
                                                                                                    first_loop,
                                                                                                    None)

                        while frameCount < lenLoop:

                            if frameCount % FD_everyIthFrame == 0:

                                if first_loop:
                                    feature_ID_max = None

                                if ptv:
                                    featuresToTrack, first_loop, feature_ID_max = ptv.FeatureDetectionPTV(dir_imgs,
                                                                                                          img_list,
                                                                                                          frameCount,
                                                                                                          minimumThreshBrightness,
                                                                                                          neighborSearchRadius_FD,
                                                                                                          searchMask,
                                                                                                          maximumNeighbors_FD,
                                                                                                          maxFtNbr_FD,
                                                                                                          sensitiveFD,
                                                                                                          savePlotData,
                                                                                                          directoryOutput,
                                                                                                          FD_everyIthFrame,
                                                                                                          first_loop,
                                                                                                          feature_ID_max)

                                print('features detected for folder ' + str(index) + ' out of ' + str(
                                    len(eor_Table.index)) + ' folders \n')
                                print('------------------------------------------')

                                '''-------perform feature tracking-------'''
                                trackedFeaturesOutput_undist, imagesForGif = ptv.FeatureTracking(template_width,
                                                                                                 template_height,
                                                                                                 search_area_x_CC,
                                                                                                 search_area_y_CC,
                                                                                                 shiftSearchFromCenter_x,
                                                                                                 shiftSearchFromCenter_y,
                                                                                                 frameCount,
                                                                                                 FT_forNthNberFrames,
                                                                                                 TrackEveryNthFrame,
                                                                                                 dir_imgs,
                                                                                                 img_list, featuresToTrack,
                                                                                                 interior_orient,
                                                                                                 performLSM, lsmBuffer,
                                                                                                 threshLSM,
                                                                                                 subpixel,
                                                                                                 trackedFeaturesOutput_undist,
                                                                                                 save_gif, imagesForGif,
                                                                                                 directoryOutput,
                                                                                                 lk, initialLK)

                            frameCount = frameCount + 1

                        # write tracked features to file
                        ioF.writeOutput(trackedFeaturesOutput_undist, FT_forNthNberFrames, FD_everyIthFrame,
                                        directoryOutput)

                        # save gif
                        if save_gif:
                            print('save tracking result to gif\n')
                            # global imageio
                            imageio.mimsave(directoryOutput + 'trackedFeatures.gif', imagesForGif)
                            # del imageio
                        print('feature tracking done\n')
                        print('------------------------------------------')

                        '''-------filter tracking results in image space-------'''
                        filteredFeatures, [nbr_features_raw, nbr_features_mindist,
                                           nbr_features_maxdist, minimumTrackedFeatures, steady_angle,
                                           nbr_features_steady, range_angle, nbr_features_rangeangle,
                                           flowdir_angle, nbr_features_mainflowdir] = ptv.FilterTracks(trackedFeaturesOutput_undist,
                                                                                                        img_name, directoryOutput,
                                                                                                        minDistance_px, maxDistance_px,
                                                                                                        minTrackedFeatures,
                                                                                                        threshAngleSteadiness,
                                                                                                        threshAngleRange,
                                                                                                        binNbrMainflowdirection,
                                                                                                        MainFlowAngleBuffer,
                                                                                                        lspiv)

                        print('filtering tracks done for folder ' + str(index) + ' out of ' + str(
                            len(eor_Table.index)) + ' folders \n')
                        print('------------------------------------------')

                        '''-------transform img measurements into object space-------'''
                        ptv.TracksPx_to_TracksMetric(filteredFeatures, interior_orient, eor_mat,
                                                     unit_gcp,
                                                     frame_rate_cam, TrackEveryNthFrame, waterlevel_pt, directoryOutput,
                                                     img_name,
                                                     veloStdThresh, lspiv, veloFilterSize, searchMask)

                        '''-------logfile-------'''
                        log_file_writer, logfile = ioF.logfile_writer(directoryOutput + 'logfile.txt')
                        log_file_writer.writerow(
                            [['waterlevel: ', waterlevel_pt], ['waterlevel_buffer: ', waterlevel_buffer],
                             ['minimumThreshBrightness: ', minimumThreshBrightness],
                             ['neighborSearchRadius_FD: ', neighborSearchRadius_FD],
                             ['maximumNeighbors_FD :', maximumNeighbors_FD], ['maxFtNbr_FD :', maxFtNbr_FD],
                             ['sensitiveFD: ', sensitiveFD],
                             ['template_width: ', template_width], ['template_height: ', template_height],
                             ['search_area_x_CC: ', search_area_x_CC], ['search_area_y_CC: ', search_area_y_CC],
                             ['shiftSearchFromCenter_x: ', shiftSearchFromCenter_x],
                             ['shiftSearchFromCenter_y: ', shiftSearchFromCenter_y],
                             ['subpixel: ', subpixel], ['performLSM: ', performLSM],
                             ['FD_everyIthFrame: ', FD_everyIthFrame],
                             ['FT_forNthNberFrames: ', FT_forNthNberFrames],
                             ['TrackEveryNthFrame: ', TrackEveryNthFrame], ['frame_rate_cam: ', frame_rate_cam],
                             ['minDistance_px: ', minDistance_px], ['nbr features min dist: ', nbr_features_mindist],
                             ['maxDistance_px: ', maxDistance_px], ['nbr features max dist: ', nbr_features_maxdist],
                             ['minTrackedFeatures: ', minTrackedFeatures],['minimumTrackedFeatures: ', minimumTrackedFeatures],
                             ['threshAngleSteadiness: ', threshAngleSteadiness],
                             ['nbr features steadyness: ', nbr_features_steady],
                             ['average angle steadiness: ', steady_angle],
                             ['threshAngleRange: ', threshAngleRange],
                             ['nbr features angle range: ', nbr_features_rangeangle],
                             ['average range angle: ', range_angle],
                             ['binNbrMainflowdirection: ', binNbrMainflowdirection],
                             ['MainFlowAngleBuffer: ', MainFlowAngleBuffer],
                             ['nbr features main flow direction: ', nbr_features_mainflowdir],
                             ['median angle flow direction: ', flowdir_angle],
                             ['veloStdThresh: ', veloStdThresh], ['nbr filtered features: ', filteredFeatures.shape[0]],
                             ['nbr raw features: ', nbr_features_raw]])
                        logfile.flush()
                        logfile.close()
                    except Exception as e:
                        print(e)
                        print('folder ' + dir_imgs + ' failed')

                    print('finished for ' + str(index) + ' out of ' + str(len(eor_Table.index)) + ' folders')

                    print('------------------------------------------')


def main():

    app = FlowVeloTool()
    app.EstimateVelocity()


if __name__ == "__main__":
    main()
