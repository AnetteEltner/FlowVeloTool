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


import csv, os
     
     
def logfile_writer(outputFile):
    #log file
    logfile = open(outputFile, 'wb')
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
    outputFileFT = open(os.path.join(directoryOutput, 'Tracking_FT_nbrFrames_' + str(FT_forNthNberFrames) + '_FD_nbrFrames_' + str(FD_everyIthFrame)) + '.txt', 'wb')
    writer = csv.writer(outputFileFT, delimiter="\t")
    writer.writerow(['frame', 'id','x', 'y'])
    writer.writerows(trackedFeaturesOutput_undist)
    outputFileFT.flush()
    outputFileFT.close()
    del writer