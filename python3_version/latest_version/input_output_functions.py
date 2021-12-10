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


import csv, os, cv2, numpy as np, pandas as pd
     
     
def logfile_writer(outputFile):
    #log file
    logfile = open(outputFile, 'w')
    writer = csv.writer(logfile, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
    logfile.flush()
    
    return writer, logfile


def read_imgs_folder(dir_imgs):
    #read image names in folder
    img_list = []
    for img_file in os.listdir(dir_imgs):
        if '.jpg' or '.png' in img_file:
            img_list.append(img_file)
    img_list = sorted(img_list)
    
    return img_list


def writeOutput(trackedFeaturesOutput_undist, FT_forNthNberFrames, FD_everyIthFrame, directoryOutput):
    outputFileFT = open(os.path.join(directoryOutput, 'Tracking_FT_nbrFrames_' + str(FT_forNthNberFrames) + '_FD_nbrFrames_' + str(FD_everyIthFrame)) + '.txt', 'w')
    writer = csv.writer(outputFileFT, delimiter="\t")
    writer.writerow(['frame', 'id','x', 'y'])
    writer.writerows(trackedFeaturesOutput_undist)
    outputFileFT.flush()
    outputFileFT.close()
    del writer


def click_event(event, x, y, flags, params):

    if event == cv2.EVENT_LBUTTONDOWN:
        outf = open(outfile, 'a')
        outf.write('%i' % gcpName)
        ## displaying the coordinates
        # print(gcpName)
        # print(x, ' ', y)

        print('point coos of ' + str(gcpName) + ': ' + str(x), ' ', str(y))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'GCP ' + str(gcpName) + ' ' + str(x) + ',' + str(y), (x, y), font, 1, (255, 0, 0), 2)
        cv2.drawMarker(img, (x, y), (255, 0, 0), markerType=cv2.MARKER_SQUARE, markerSize=10)
        cv2.imshow(' ', img)
        outf.write('\t%i\t%i\t\n' % (x, y))
        outf.close()

    elif event == cv2.EVENT_RBUTTONDOWN:
        # removing point from table
        outf = open(outfile, 'r')
        lines = outf.readlines()
        del lines[-1]
        outf.close()

        outf = open(outfile, 'w')
        for line in lines:
            outf.write(line)
        outf.write('deleteLine')
        outf.close()
        print('deleted previous point')


def draw_text(img, text,
          font=cv2.FONT_HERSHEY_SIMPLEX,
          pos=(0, 0),
          font_scale=1,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, (x-5, y-5), (x + text_w+5, y + text_h+5), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size

def clickImgCoo(image, gcpList, directoryOutput):
    # reading the image
    global img
    img = cv2.imread(image)
    imagePathName = 'image for GCP measurement'
    gcpObjPts_table = np.asarray(pd.read_table(gcpList, header=None))

    global gcpName
    global pointsNbr
    global outfile
    outfile = directoryOutput + "gcpPts_img_manualMeasured.txt"

    outf = open(outfile, 'w')
    #outf.write("id x y\n")
    outf.close()

    pointsNbr = 0
    while pointsNbr < gcpObjPts_table.shape[0]:
        gcpName = gcpObjPts_table[pointsNbr, 0]
        print('measure point ' + str(gcpName))

        # resizing display window
        cv2.namedWindow(imagePathName, cv2.WINDOW_NORMAL)
        cv2.moveWindow(imagePathName, 10, 10)
        cv2.resizeWindow(imagePathName, 1600, 900)

        instr_left = "Left click:  Define the GCP in image (only click once)"
        instr_right = "Right click: Delete the previously clicked point"
        instr_key = "After each mouse action please click any key to continue with the next point"

        draw_text(img, instr_left, pos=(25, 25), text_color=(255, 0, 0))
        draw_text(img, instr_right, pos=(25, 60), text_color=(255, 255, 0))
        draw_text(img, instr_key, pos=(25, 95), text_color=(255, 255, 255))

        # displaying the image
        cv2.imshow(imagePathName, img)

        # setting mouse handler for the image
        cv2.setMouseCallback(imagePathName, click_event)

        # wait for a key to be pressed to exit
        cv2.waitKey(0)
        # close the window
        cv2.destroyAllWindows()

        outf = open(outfile, 'r')
        lines = outf.readlines()
        outf.close()
        deletedPoint = False
        for line in lines:
            if line == "deleteLine":
                deletedPoint = True
                break

        if deletedPoint:
            outf = open(outfile, 'w')
            del lines[-1]
            for line in lines:
                outf.write(line)
            outf.close()
            pointsNbr = pointsNbr - 1
            print('removed point ' + str(gcpObjPts_table[pointsNbr, 0]) + ' and has to be measured again')
        else:
            pointsNbr = pointsNbr + 1

    outf.close()


#!/usr/bin/env python
# -- coding: utf-8 --

# """
# Hannah weiser, September 2021
# h.weiser@stud.uni-heidelberg.de
# Function to manually track objects (branches) in the video frames.
# Execution (example)
# image_coords_by_click.py --img "I:\UAV-photo\Befliegung_2020\for_velocity\frames_30fps_coreg" --outdir "I:\UAV-photo\test" --ior "I:\UAV-photo\Befliegung_2020\for_velocity\sensorInteriorOrientation.txt" --gcpcoo "I:\UAV-photo\Befliegung_2020\for_velocity\GCPsinObjectSpace.txt" --gcpimg "I:\UAV-photo\Befliegung_2020\for_velocity\GCPsInImage.txt" --fps 30 --trackevery 30 --waterlevel 137.0
# To view the help, run:
# image_coords_by_click.py --help
# or
# image_coords_by_click.py -h
# """

# import cv2
# from pathlib import Path, PurePath
# import pandas as pd
# import PTV_functions as ptv
# import numpy as np
# import featureReference_functions as refF
# import draw_functions as drawF
# import featureFilter_functions as filterF
# import photogrammetry_functions as photogrF
# import matplotlib.pyplot as plt
# import matplotlib
# import argparse
#
#
# def readUserArguments():
#     # Generate help and user argument parsing
#     parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#                                      description="(c) Hannah Weiser (2021) - Heidelberg University")
#     parser.add_argument("--img", dest='images', type=str,
#                         help="Directory containing (co-registered) frames of the video",
#                         required=True)
#     parser.add_argument("--outdir", dest='outdir', type=str,
#                         help="Output directory of the FlowVelo tool, containing the results of feature tracking, filtering and velocity computation",
#                         required=True)
#     parser.add_argument("--ior", dest='ior_file', type=str,
#                         help="Output directory of the FlowVelo tool, containing the results of feature tracking, filtering and velocity computation",
#                         required=True)
#     parser.add_argument("--gcpcoo", dest='gcpCoo_file', type=str,
#                         help="Output directory of the FlowVelo tool, containing the results of feature tracking, filtering and velocity computation",
#                         required=True)
#     parser.add_argument("--gcpimg", dest='imgCoo_GCP_file', type=str,
#                         help="Output directory of the FlowVelo tool, containing the results of feature tracking, filtering and velocity computation",
#                         required=True)
#     parser.add_argument("--fps", dest='framerate', type=int,
#                         help="Frame rate (frames per second) of the camera",
#                         required=True)
#     parser.add_argument("--trackevery", dest='trackeverynth', type=int,
#                         help="Number of frames to skip, i.e., track every nth frame",
#                         required=True)
#     parser.add_argument("--waterlevel", dest='water_level', type=float,
#                         help="Water level in m",
#                         required=True)
#
#     opts = parser.parse_args()
#     return opts
#
# def TracksPx_to_TracksMetric(filteredFeatures, interior_orient, eor_mat, unit_gcp,
#                              frame_rate_cam, TrackEveryNthFrame, waterlevel_pt,
#                              directoryOutput, img_name, every_xth=1):
#     # scale tracks in image space to tracks in object space to get flow velocity in m/s
#     waterlevel = waterlevel_pt
#
#     filteredFeatures_list = []
#     if every_xth > 1:
#         for group, coords in filteredFeatures.groupby('id', as_index=False):
#             for i, j in enumerate(range(0, coords.shape[0]-every_xth+1, every_xth)):
#                 filteredFeatures = coords.iloc[j:j + every_xth + 1]
#                 if group == 1:
#                     filteredFeatures_list.append(filteredFeatures)
#                 else:
#                     df = filteredFeatures_list[i]
#                     df = df.append(filteredFeatures)
#                     filteredFeatures_list[i] = df
#     else:
#         filteredFeatures_list = [filteredFeatures]
#
#     subtracks = len(filteredFeatures_list)
#
#     for k, filteredFeatures in enumerate(filteredFeatures_list):
#         image = cv2.imread(img_name, 0)
#
#         # get first and last position in image space of each tracked feature
#         filteredFeatures_1st = filteredFeatures.groupby('id', as_index=False).head(1)
#         filteredFeatures_last = filteredFeatures.groupby('id', as_index=False).tail(1)
#         filteredFeatures_count = np.asarray(filteredFeatures.groupby('id', as_index=False).count())[:, 2]
#         print("count", filteredFeatures_count)
#
#         xy_start_tr = np.asarray(filteredFeatures_1st[['x', 'y']])
#         xy_tr = np.asarray(filteredFeatures_last[['x', 'y']])
#
#         # intersect first and last position with waterlevel
#         XY_start_tr = refF.LinePlaneIntersect(xy_start_tr, waterlevel, interior_orient, eor_mat, unit_gcp) / unit_gcp
#         XY_tr = refF.LinePlaneIntersect(xy_tr, waterlevel, interior_orient, eor_mat, unit_gcp) / unit_gcp
#
#         # get angle of track
#         x_track = xy_tr[:, 0] - xy_start_tr[:, 0]
#         y_track = xy_tr[:, 1] - xy_start_tr[:, 1]
#         track = np.hstack((x_track.reshape(x_track.shape[0], 1), y_track.reshape(y_track.shape[0], 1)))
#         angle = np.degrees(filterF.angleBetweenVecAndXaxis(track))
#
#         # get corresponding distance in object space
#         dist_metric = np.sqrt(np.square(XY_start_tr[:, 0] - XY_tr[:, 0]) + (np.square(XY_start_tr[:, 1] - XY_tr[:, 1])))
#
#         # get corresponding temporal observation span
#         frame_rate = np.ones((filteredFeatures_count.shape[0], 1), dtype=np.float) * np.float(frame_rate_cam)
#         nbrTrackedFrames = TrackEveryNthFrame * (filteredFeatures_count-1)
#         trackingDuration = nbrTrackedFrames.reshape(frame_rate.shape[0], 1) / frame_rate
#         print(trackingDuration)
#
#         # get velocity
#         velo = dist_metric.reshape(trackingDuration.shape[0], 1) / trackingDuration
#         filteredFeatures_1st = ptv.filterFeatureOrganise(filteredFeatures_1st, XY_start_tr, XY_tr, xy_tr, dist_metric,
#                                                          velo, True, filteredFeatures_count-1)
#         filteredFeatures = filteredFeatures_1st.copy()
#         filteredFeatures = filteredFeatures.reset_index(drop=True)
#         filteredFeaturesRawPTVOut = filteredFeatures[['X', 'Y', 'Z', 'velo', 'dist_metric', 'count']]
#         filteredFeaturesRawPTVOut.columns = ['X', 'Y', 'Z', 'velo', 'dist', 'count']
#         filteredFeaturesRawPTVOut['angle'] = angle.values
#         filteredFeaturesRawPTVOut['duration'] = filteredFeaturesRawPTVOut['count'] * TrackEveryNthFrame
#         if subtracks == 1:
#             suffix = ''
#         else:
#             suffix = '_sub%s' % k
#         filteredFeaturesRawPTVOut.to_csv(directoryOutput + 'TracksReferenced_raw_PTV' + suffix + '.txt', sep='\t',
#                                          index=False)
#         del filteredFeaturesRawPTVOut
#         draw_tracks(filteredFeatures, image, directoryOutput, 'TracksReferenced_raw_PTV' + suffix + '.jpg',
#                           'velo', colors=["red", "deepskyblue"], label_data="True", variableToLabel='velo')
#
#         print('nbr of tracked features: ' + str(filteredFeatures.shape[0]) + '\n')
#
# def draw_tracks(Final_Vals, image, dir_out, outputImgName, variableToDraw, colors,
#                 label_data=False, variableToLabel=None):
#     try:
#         '''visualize'''
#         # sort after flow velocity
#         image_points = Final_Vals.sort_values(variableToDraw)
#         image_points = image_points.reset_index(drop=True)
#
#         # set font size
#         fontProperties_text = {'size' : 12,
#                                'family' : 'serif'}
#         matplotlib.rc('font', **fontProperties_text)
#
#         # draw figure
#         fig = plt.figure(frameon=False)
#         ax = plt.Axes(fig, [0., 0., 1., 1.])
#         ax.set_axis_off()
#         ax.axis('equal')
#         fig.add_axes(ax)
#
#         image_points = image_points.sort_values('id')
#         image_points = image_points.reset_index(drop=True)
#
#         # add arrows
#         point_n = 0
#         label_criteria = 0
#
#         while point_n <= image_points.shape[0]:
#             try:
#                 if label_data:
#                     id, label, xl, yl, arr_x, arr_y = image_points['id'][point_n], image_points[variableToLabel][point_n], image_points['x'][point_n], image_points['y'][point_n], image_points['x_tr'][point_n], image_points['y_tr'][point_n]
#                 else:
#                     xl, yl, arr_x, arr_y = image_points['x'][point_n], image_points['y'][point_n], image_points['x_tr'][point_n], image_points['y_tr'][point_n]
#                 ax.arrow(xl, yl, arr_x-xl, arr_y-yl, color=colors[id-1],
#                          head_width=5, head_length=5, width=1.5)
#
#                 if label_data:
#                     if id == 1:
#                         ax.annotate(str("{0:.2f}".format(label)), xy=(xl+25, yl+25), color=colors[id-1],
#                                     **fontProperties_text)
#                     else:
#                         ax.annotate(str("{0:.2f}".format(label)), xy=(xl+35, yl-25), color=colors[id-1],
#                                     **fontProperties_text)
#                 point_n += 1
#
#             except Exception as e:
#                 point_n += 1
#
#         ax.imshow(image, cmap='gray')
#
#         # save figure
#         plt.savefig(str(Path(dir_out) / outputImgName),  dpi=600)
#         plt.close('all')
#         plt.clf()
#
#     except Exception as e:
#         print(e)
#
#
# def click_event(event, x, y, flags, params):
#     if event == cv2.EVENT_LBUTTONDOWN:
#
#         # writing filename to output file
#         outf.write(str(img_path) + " 1")
#
#         # displaying the coordinates
#         # a) on the Shell
#         print(x, ' ', y)
#
#         # b) on the image window
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         cv2.putText(img, str(x) + ',' + str(y), (x, y), font, 1, (255, 0, 0), 2)
#         cv2.drawMarker(img, (x, y), (255, 0, 0), markerType=cv2.MARKER_SQUARE, markerSize=10)
#         cv2.imshow(img_path.name, img)
#
#         # write image coordinates to file
#         outf.write(' %i %i\n' % (x, y))
#
#     elif event == cv2.EVENT_RBUTTONDOWN:
#
#         # writing filename to output file
#         outf.write(str(img_path) + " 2")
#
#         # displaying the coordinates
#         # a) on the Shell
#         print(x, ' ', y)
#
#         # b) on the image window
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         cv2.putText(img, str(x) + ',' + str(y), (x, y), font, 1, (255, 255, 0), 2)
#         cv2.drawMarker(img, (x, y), (255, 255, 0), markerType=cv2.MARKER_SQUARE, markerSize=10)
#         cv2.imshow(img_path.name, img)
#
#         # write image coordinates to file
#         outf.write(' %i %i\n' % (x, y))
#
#
# def draw_text(img, text,
#           font=cv2.FONT_HERSHEY_SIMPLEX,
#           pos=(0, 0),
#           font_scale=1,
#           font_thickness=2,
#           text_color=(0, 255, 0),
#           text_color_bg=(0, 0, 0)
#           ):
#
#     x, y = pos
#     text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
#     text_w, text_h = text_size
#     cv2.rectangle(img, (x-5, y-5), (x + text_w+5, y + text_h+5), text_color_bg, -1)
#     cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)
#
#     return text_size
#
# def clickImgCoo():
#     # reading the image
#     img = cv2.imread(str(img_path))
#
#     # resizing display window
#     cv2.namedWindow(img_path.name, cv2.WINDOW_NORMAL)
#     cv2.moveWindow(img_path.name, 10, 10)
#     cv2.resizeWindow(img_path.name, 1600, 900)
#
#     # fontScale
#     fontScale = 1
#
#     # Line thickness of 2 px
#     thickness = 2
#
#     instr_left = "Left click:  Define the one end of the branch"
#     instr_right = "Right click: Define the other end of the branch"
#     instr_key = "Any key:    Press any key to continue with the next image"
#
#     draw_text(img, instr_left, pos=(25, 25), text_color=(255, 0, 0))
#     draw_text(img, instr_right, pos=(25, 60), text_color=(255, 255, 0))
#     draw_text(img, instr_key, pos=(25, 95), text_color=(255, 255, 255))
#
#     # displaying the image
#     cv2.imshow(img_path.name, img)
#
#     # setting mouse handler for the image
#     # and calling the click_event() function
#     cv2.setMouseCallback(img_path.name, click_event)
#
#     # wait for a key to be pressed to exit

#     cv2.waitKey(0)
#
#     # close the window
#     cv2.destroyAllWindows()
#
# # driver function
# if __name__ == "__main__":

    # opts = readUserArguments()
    #
    # images = Path(opts.images).glob('*.jpg')
    # out_dir = opts.outdir
    # try:
    #     Path(out_dir).mkdir(parents=True, exist_ok=False)
    # except FileExistsError:
    #     print("Folder is already there")
    # else:
    #     print("Folder was created")
    #
    # outfile = PurePath(out_dir, 'branch_coords.txt')
    #
    # outf = open(outfile, 'w')
    # outf.write("filename id x y\n")
    #
    # for i, img_path in enumerate(images):
    #     if i % opts.trackeverynth == 0:
    #         # reading the image
    #         img = cv2.imread(str(img_path))
    #
    #         # resizing display window
    #         cv2.namedWindow(img_path.name, cv2.WINDOW_NORMAL)
    #         cv2.moveWindow(img_path.name, 10, 10)
    #         cv2.resizeWindow(img_path.name, 1600, 900)
    #
    #         # fontScale
    #         fontScale = 1
    #
    #         # Line thickness of 2 px
    #         thickness = 2
    #
    #         instr_left = "Left click:  Define the one end of the branch"
    #         instr_right = "Right click: Define the other end of the branch"
    #         instr_key = "Any key:    Press any key to continue with the next image"
    #
    #         draw_text(img, instr_left, pos=(25, 25), text_color=(255, 0, 0))
    #         draw_text(img, instr_right, pos=(25, 60), text_color=(255, 255, 0))
    #         draw_text(img, instr_key, pos=(25, 95), text_color=(255, 255, 255))
    #
    #         # displaying the image
    #         cv2.imshow(img_path.name, img)
    #
    #         # setting mouse handler for the image
    #         # and calling the click_event() function
    #         cv2.setMouseCallback(img_path.name, click_event)
    #
    #         # wait for a key to be pressed to exit
    #         cv2.waitKey(0)
    #
    #         # close the window
    #         cv2.destroyAllWindows()

    # outf.close()
    #
    # images = list(Path(opts.images).glob('*.jpg'))
    # index_lastimg = opts.trackeverynth * np.floor(len(images)/opts.trackeverynth) - 1
    # last_img = str(images[int(index_lastimg)])
    # img = cv2.imread(last_img)
    #
    # cv2.namedWindow('track', cv2.WINDOW_NORMAL)
    # cv2.moveWindow('track', 10, 10)
    # cv2.resizeWindow('track', 1600, 900)
    #
    # # create track and write onto last image
    # df_coords = pd.read_csv(outfile, sep=" ")
    # df_coords_1 = df_coords[df_coords.id == 1]
    # df_coords_2 = df_coords[df_coords.id == 2]
    #
    # for i, vals in enumerate(df_coords_1.values):
    #     if i < 1:
    #         continue
    #     else:
    #         prev_point = df_coords_1.values[i-1]
    #     cv2.line(img, (prev_point[2], prev_point[3]), (vals[2], vals[3]), color=(0, 0, 255), thickness=2)
    #
    # for i, vals in enumerate(df_coords_2.values):
    #     if i < 1:
    #         continue
    #     else:
    #         prev_point = df_coords_2.values[i-1]
    #     cv2.line(img, (prev_point[2], prev_point[3]), (vals[2], vals[3]), color=(255, 191, 0), thickness=2)
    #
    # cv2.imshow('track', img)
    # cv2.imwrite(str(Path(out_dir) / "branch_track.jpg"), img)
    # cv2.waitKey(0)
    #
    # # close the window
    # cv2.destroyAllWindows()
    # interior_orient = photogrF.read_aicon_ior(opts.ior_file)
    # print("Computing velocity")
    # eor_mat = ptv.EstimateExterior(opts.gcpCoo_file, opts.imgCoo_GCP_file, interior_orient, estimate_exterior=True,
    #                                unit_gcp=1000.0, max_orientation_deviation=1, ransacApprox=True, angles_eor=None,
    #                                pos_eor=None, directoryOutput=str(out_dir))
    # TracksPx_to_TracksMetric(df_coords, interior_orient, eor_mat, unit_gcp=1000.0,
    #                          frame_rate_cam=opts.framerate, TrackEveryNthFrame=opts.trackeverynth, waterlevel_pt=opts.water_level,
    #                          directoryOutput=str(out_dir), img_name=last_img, every_xth=1)
    # TracksPx_to_TracksMetric(df_coords, interior_orient, eor_mat, unit_gcp=1000.0,
    #                          frame_rate_cam=opts.framerate, TrackEveryNthFrame=opts.trackeverynth, waterlevel_pt=opts.water_level,
    #                          directoryOutput=str(out_dir), img_name=last_img, every_xth=4)