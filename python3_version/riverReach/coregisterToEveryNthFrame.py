import os  
import cv2
import numpy as np

import coregistration_functions as coregF


def read_imgs_folder(dir_imgs):
    # read image names in folder
    img_list = []
    for img_file in os.listdir(dir_imgs):
        if '.jpg' in img_file:
            img_list.append(dir_imgs + img_file)
        elif '.png' in img_file:
            img_list.append(dir_imgs + img_file)
    img_list = sorted(img_list)

    return img_list

def videoToFrame(videoFile, outputDir):
    vidcap = cv2.VideoCapture(videoFile)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(outputDir + "frame%05d.png" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1

def mergeBands(imgBlue, imgGreen, imgRed):
    mergedImg = np.dstack((imgBlue, imgGreen))
    mergedImg = np.dstack((mergedImg, imgRed))
    mergedImg = np.asarray(mergedImg, dtype=np.uint8)

    return mergedImg


### ---> things you need to declare

'''your input data'''
# output directory of the processed video frames and extracted video frames
dirOut = '.../example/'
# path to and file name of the video file
videoFile = '[YourPath]/[yourVideoFileName.mp4]'

'''set parameters for processing'''
# every how many frames should a new head be selected
everyIthFrame = 50
# how many frames are co-registered to each head
numberFramesToCoregister = 15
# akaze threshold (see FlowVeloTool instruction)
keypointReduce = 0.015

VideoFile = 'dummy'
### <--- that is all


vidcap = cv2.VideoCapture(videoFile)
success, image = vidcap.read()
countFrames = 0
countIth = 0
image_count = 0
imgList_coreg = []
while success:
    #start reading video frames from video file
    success, image = vidcap.read()

    # considers after how many frames a new head frame is chosen to which tailing frames
    # will be co-registered (loop continues without processing anything until nbr of head
    # frame is reached)
    if countIth < everyIthFrame:
        countIth = countIth + 1
        continue

    #considers how many frames will be co-registered (i.e. how many frames will be
    #kept for processing -> result is stack of frames)
    if countFrames < numberFramesToCoregister:
        imgList_coreg.append(image)
        countFrames = countFrames + 1
        continue

    # reset count before start of actual frame processing
    countFrames = 0
    countIth = 0

    # select first frame as raw, original frame representative of stacked frames
    if not os.path.isdir(dirOut + '1st'):
        os.mkdir(dirOut + '1st')
    cv2.imwrite(dirOut + '1st/' + VideoFile + str(image_count * everyIthFrame) + '_nbrCoReg_' + str(numberFramesToCoregister) + '.png',
                imgList_coreg[1])
    print('finished saving 1st images')

    # co-register stacked frames
    imgsCoregistered = coregF.coregistrationListOut(imgList_coreg, keypointReduce, 'akaze', True)

    # process co-registered frames
    img_coreg_count = 0
    for img in imgsCoregistered:

        # processing for each band separately
        imgNew_Blue = img[:, :, 0]
        imgNew_Green = img[:, :, 1]
        imgNew_Red = img[:, :, 2]

        # stack frames of each band
        if img_coreg_count == 0:
            imgForStat_Blue = imgNew_Blue
            imgForStat_Green = imgNew_Green
            imgForStat_Red = imgNew_Red
        else:
            imgForStat_Blue = np.dstack((imgForStat_Blue, imgNew_Blue))
            imgForStat_Green = np.dstack((imgForStat_Green, imgNew_Green))
            imgForStat_Red = np.dstack((imgForStat_Red, imgNew_Red))

        img_coreg_count = img_coreg_count + 1

        # output each frame of stacked frames for later flow velocity processing
        if not os.path.isdir(dirOut + 'forVelos'):
            os.mkdir(dirOut + 'forVelos')
        forVelosSubdir = dirOut + 'forVelos/' + str(image_count * everyIthFrame) + '_velocities/'
        if not os.path.isdir(forVelosSubdir):
            os.mkdir(forVelosSubdir)
        cv2.imwrite(forVelosSubdir + VideoFile + str(image_count * everyIthFrame + img_coreg_count) +
                    '_oriFrame.png', img)

    print('finished stacking images')

    # get median image from stack of frames (per band)
    imgMedian_Blue = np.nanmedian(imgForStat_Blue, axis=2)
    imgMedian_Green = np.nanmedian(imgForStat_Green, axis=2)
    imgMedian_Red = np.nanmedian(imgForStat_Red, axis=2)
    imgMedian = mergeBands(imgMedian_Blue, imgMedian_Green, imgMedian_Red)

    if not os.path.isdir(dirOut + 'median'):
        os.mkdir(dirOut + 'median')
    cv2.imwrite(dirOut + 'median/' + VideoFile + str(image_count * everyIthFrame) + '_nbrCoReg_' + str(
                numberFramesToCoregister) + '_median.png', imgMedian)
    print('finished median images')

    # get minimum image from stack of frames (per band)
    imgMin_Blue = np.nanmin(imgForStat_Blue, axis=2)
    imgMin_Green = np.nanmin(imgForStat_Green, axis=2)
    imgMin_Red = np.nanmin(imgForStat_Red, axis=2)
    imgMin = mergeBands(imgMin_Blue, imgMin_Green, imgMin_Red)

    if not os.path.isdir(dirOut + 'min'):
        os.mkdir(dirOut + 'min')
    cv2.imwrite(
        dirOut + 'min/' + VideoFile + str(image_count * everyIthFrame) + '_nbrCoReg_' + str(numberFramesToCoregister) + '_min.png',
        imgMin)
    print('finished min images')

    #get maximum image from stack of frames (per band)
    imgMax_Blue = np.nanmax(imgForStat_Blue, axis=2)
    imgMax_Green = np.nanmax(imgForStat_Green, axis=2)
    imgMax_Red = np.nanmax(imgForStat_Red, axis=2)
    imgMax = mergeBands(imgMax_Blue, imgMax_Green, imgMax_Red)

    if not os.path.isdir(dirOut + 'max'):
        os.mkdir(dirOut + 'max')
    cv2.imwrite(
        dirOut + 'max/' + VideoFile + str(image_count * everyIthFrame) + '_nbrCoReg_' + str(numberFramesToCoregister) + '_max.png',
        imgMax)
    print('finished max images')

    #get mean image from stack of frames (per band)
    imgMean_Blue = np.nanmean(imgForStat_Blue, axis=2)
    imgMean_Green = np.nanmean(imgForStat_Green, axis=2)
    imgMean_Red = np.nanmean(imgForStat_Red, axis=2)
    imgMean = mergeBands(imgMean_Blue, imgMean_Green, imgMean_Red)

    if not os.path.isdir(dirOut + 'mean'):
        os.mkdir(dirOut + 'mean')
    cv2.imwrite(dirOut + 'mean/' + VideoFile + str(image_count * everyIthFrame) + '_nbrCoReg_' + str(
        numberFramesToCoregister) + '_mean.png', imgMean)
    print('finished mean images')

    print('finished ' + str(countFrames / everyIthFrame) + ' of ' + str(len(imgList_coreg) / everyIthFrame))

    imgList_coreg = []
    image_count = image_count + 1
