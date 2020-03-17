#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Anette Eltner'
__contact__ = 'Anette.Eltner@tu-dresden.de'
__copyright__ = '(c) Anette Eltner 2019'
__license__ = 'MIT'
__date__ = '03 May 2019'
__version__ = '1.0'
__status__ = "initial release"
__url__ = "https://github.com/AnetteEltner/FlowVeloTool"


"""
Name:           FlowVeloTool
Compatibility:  Python 3.6
Description:    This program performs image-based flow velocity estimation. It includes
                camera orientation estimation, feature detection and tracking, 
                and image co-registration.
URL:            https://github.com/AnetteEltner/FlowVeloTool
Requires:       tkinter, pyttk 0.3.2, scipy 1.4.1, scikit-learn 0.22.2.post1, scikit-image 0.16.2, shapely 1.7.0, imageio 2.8.0,
                opencv 4.2.0.32, seaborn 0.10.0, matplotlib 3.2.0
AUTHORS:        Anette Eltner
ORGANIZATION:   TU Dresden
Contact:        Anette.Eltner@tu-dresden.de
Copyright:      (c) Anette Eltner 2019
Licence:        MIT
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in 
the Software without restriction, including without limitation the rights to 
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
of the Software, and to permit persons to whom the Software is furnished to do 
so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
"""


import os
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

#global imageio 

class FlowVeloTool:
    
    def __init__(self, master):
        
        master_frame = Frame(master, name='master_frame')
        master.title('Image-based flow velocity estimation')
        note = Notebook(master_frame, name='note')
        master_frame.grid()
        
        #text box for display output
        self.textbox = ScrolledText(master, height=10, width=20)
        self.textbox.place(x=700, y=50, width=300, height=800)
                                
        
        '''----------------frame flow velocity I-------------------'''
        frame = Frame(note)
        note.add(frame, text="flow velocity I")
        note.grid(row=0, column=0, ipadx=500, ipady=460)
        
        self.xButton = 370
        self.xText = 250
        self.xText2 = 350
        self.yAddText = 10
        self.yAddText2 = 10
        Style().configure("RB.TButton", foreground='blue', font=('helvetica', 10))

        currentDirectory = os.getcwd()

        # save parameter settings to file
        Label(frame, text="Save parameter settings: ", font=("Helvetica", 10)).place(x=10, y=self.yAddText)
        self.ParamSave_Button = Button(frame, text='...', command=lambda: self.safeParemeterSetting())
        self.ParamSave_Button.place(x=self.xText - 50, y=self.yAddText, width=20, height=20)

        # load parameter settings to file
        self.yAddText = self.yAddText + 20
        Label(frame, text="Load parameter settings: ", font=("Helvetica", 10)).place(x=10, y=self.yAddText)
        self.ParamLoad_Button = Button(frame, text='...', command=lambda: self.loadParameterSettings())
        self.ParamLoad_Button.place(x=self.xText - 50, y=self.yAddText, width=20, height=20)

        #test run
        self.test_run = tk.BooleanVar()
        self.test_run.set(False)
        self.test_runBut = tk.Checkbutton(frame, text = "Test run?", font=("Helvetica", 10), variable=self.test_run)
        self.test_runBut.place(x=540, y=5)        
        
        #load files
        self.yAddText = self.yAddText + 50
        Label(frame, text="Data input", font=("Courier", 10)).place(x=10, y=self.yAddText)
        self.yAddText = self.yAddText + 20
        Label(frame, text="Output directory: ", font=("Helvetica", 10)).place(x=10, y=self.yAddText)
        self.directoryOutput = tk.StringVar()
        self.directoryOutput_Param = Entry(frame, textvariable=self.directoryOutput, font=("Helvetica", 10, 'italic'))
        self.directoryOutput_Param.place(x=self.xText+30, y=self.yAddText, width=350, height=20)
        self.directoryOutput_Button = Button(frame, text = '...', command = lambda:self.select_dirOutput())
        self.directoryOutput_Button.place(x=self.xText, y=self.yAddText, width=20, height=20)
        self.directoryOutput.set(currentDirectory + '/tutorial/resultsFlowVelo/')
        
        self.yAddText = self.yAddText + 20
        Label(frame, text="Images directory: ", font=("Helvetica", 10)).place(x=10, y=self.yAddText)
        self.dir_imgs = tk.StringVar()
        self.dir_imgs_Param = Entry(frame, textvariable=self.dir_imgs, font=("Helvetica", 10, 'italic'))
        self.dir_imgs_Param.place(x=self.xText+30, y=self.yAddText, width=350, height=20)
        self.dir_imgs_Button = Button(frame, text = '...', command = lambda:self.select_dirImgs())
        self.dir_imgs_Button.place(x=self.xText, y=self.yAddText, width=20, height=20)
        self.dir_imgs.set(currentDirectory + '/tutorial/frames/')

        self.yAddText = self.yAddText + 20
        Label(frame, text="GCP file (object space): ", font=("Helvetica", 10)).place(x=10, y=self.yAddText)
        self.gcpCoo_file = tk.StringVar()
        self.gcpCoo_file_Param = Entry(frame, textvariable=self.gcpCoo_file, font=("Helvetica", 10, 'italic'))
        self.gcpCoo_file_Param.place(x=self.xText+30, y=self.yAddText, width=350, height=20)
        self.gcpCoo_file_Button = Button(frame, text = '...', command = lambda:self.select_GCPcooFile())
        self.gcpCoo_file_Button.place(x=self.xText, y=self.yAddText, width=20, height=20)
        self.gcpCoo_file.set(currentDirectory + '/tutorial/markersGCPobj.txt')
        
        self.yAddText = self.yAddText + 20
        Label(frame, text="GCP file (image space): ", font=("Helvetica", 10)).place(x=10, y=self.yAddText)
        self.imgCoo_GCP_file = tk.StringVar()
        self.imgCoo_GCP_file_Param = Entry(frame, textvariable=self.imgCoo_GCP_file, font=("Helvetica", 10, 'italic'))
        self.imgCoo_GCP_file_Param.place(x=self.xText+30, y=self.yAddText, width=350, height=20)
        self.imgCoo_GCP_file_Button = Button(frame, text = '...', command = lambda:self.select_GCPimgCooFile())
        self.imgCoo_GCP_file_Button.place(x=self.xText, y=self.yAddText, width=20, height=20)
        self.imgCoo_GCP_file.set(currentDirectory + '/tutorial/markersGCPimg.txt')
        
        self.yAddText = self.yAddText + 20
        Label(frame, text="Interior orientation file: ", font=("Helvetica", 10)).place(x=10, y=self.yAddText)
        self.ior_file = tk.StringVar()
        self.ior_file_Param = Entry(frame, textvariable=self.ior_file, font=("Helvetica", 10, 'italic'))
        self.ior_file_Param.place(x=self.xText+30, y=self.yAddText, width=350, height=20)
        self.ior_file_Button = Button(frame, text = '...', command = lambda:self.select_iorFile())
        self.ior_file_Button.place(x=self.xText, y=self.yAddText, width=20, height=20)    
        self.ior_file.set(currentDirectory + '/tutorial/interiorGeometry.txt')
        
        self.yAddText = self.yAddText + 20
        Label(frame, text="3D point cloud file: ", font=("Helvetica", 10)).place(x=10, y=self.yAddText)
        self.ptCloud_file = tk.StringVar()
        self.ptCloud_file_Param = Entry(frame, textvariable=self.ptCloud_file, font=("Helvetica", 10, 'italic'))
        self.ptCloud_file_Param.place(x=self.xText+30, y=self.yAddText, width=350, height=20)
        self.ptCloud_file_Button = Button(frame, text = '...', command = lambda:self.select_ptClFile())
        self.ptCloud_file_Button.place(x=self.xText, y=self.yAddText, width=20, height=20)
        self.ptCloud_file.set(currentDirectory + '/tutorial/3DmodelPointCloud.txt')
        
        self.yAddText = self.yAddText + 20
        Label(frame, text="Image file (for visualisation): ", font=("Helvetica", 10)).place(x=10, y=self.yAddText)
        self.img_name = tk.StringVar()
        self.img_name_Param = Entry(frame, textvariable=self.img_name, font=("Helvetica", 10, 'italic'))
        self.img_name_Param.place(x=self.xText+30, y=self.yAddText, width=350, height=20)
        self.img_name_Button = Button(frame, text = '...', command = lambda:self.select_imgName())
        self.img_name_Button.place(x=self.xText, y=self.yAddText, width=20, height=20)
        self.img_name.set(currentDirectory + '/tutorial/frames/frame003.png')


        #exterior estimate
        self.yAddText = self.yAddText + 50
        Label(frame, text="Exterior orientation", font=("Courier", 10)).place(x=10, y=self.yAddText)
        
        self.yAddText = self.yAddText + 20
        self.stayImgSpace = tk.BooleanVar()
        self.stayImgSpace.set(False)
        self.stayImgSpaceBut = tk.Checkbutton(frame, text = "Stay in image space", font=("Helvetica", 10), 
                                              variable=self.stayImgSpace, command = lambda:self.checkImgSpace())
        self.stayImgSpaceBut.place(x=0, y=self.yAddText)           
        
        self.yAddText = self.yAddText + 20
        self.estimate_exterior = tk.BooleanVar()
        self.estimate_exterior.set(True)
        self.estimate_exteriorBut = tk.Checkbutton(frame, text = "Estimate exterior orientation", font=("Helvetica", 10), 
                                                   variable=self.estimate_exterior, command = lambda:self.checkExterior())
        self.estimate_exteriorBut.place(x=0, y=self.yAddText)
        self.estimate_exteriorBut.config(font=("Helvetica", 10))    

        self.yAddText = self.yAddText + 20
        self.ransacApprox = tk.BooleanVar()
        self.ransacApprox.set(True)
        self.ransacApproxBut = tk.Checkbutton(frame, text = "Use RANSAC for exterior estimation", variable=self.ransacApprox,
                                              font=("Helvetica", 10), command = lambda:self.checkExterior())
        self.ransacApproxBut.place(x=0, y=self.yAddText)
                       
        self.yAddText = self.yAddText + 20
        Label(frame, text="Approximate position: ", font=("Helvetica", 10)).place(x=10, y=self.yAddText)
        self.pos_eor_Str = tk.StringVar()
        self.pos_eor_Str_Param = Entry(frame, textvariable=self.pos_eor_Str, state='disabled', font=("Helvetica", 10, 'italic'))
        self.pos_eor_Str_Param.place(x=self.xText, y=self.yAddText, width=300, height=20)
        self.pos_eor_Str.set('1.971e+02,1.965e+02,2.0458e+02')                       
        
        self.yAddText = self.yAddText + 20 
        Label(frame, text="Approximate orientation [rad]: ", font=("Helvetica", 10)).place(x=10, y=self.yAddText)
        self.angles_eor_Str = tk.StringVar()
        self.angles_eor_Str_Param = Entry(frame, textvariable=self.angles_eor_Str, state='disabled', font=("Helvetica", 10, 'italic'))
        self.angles_eor_Str_Param.place(x=self.xText, y=self.yAddText, width=300, height=20)
        self.angles_eor_Str.set('-8.901e-01,6.269e-01,-4.321e-01')                
        
        self.yAddText = self.yAddText + 20 
        Label(frame, text="Unit GCPs [mm]: ", font=("Helvetica", 10)).place(x=10, y=self.yAddText)
        self.unit_gcp = tk.DoubleVar()
        self.unit_gcp_Param = Entry(frame, textvariable=self.unit_gcp, font=("Helvetica", 10, 'italic'))
        self.unit_gcp_Param.place(x=self.xText, y=self.yAddText, width=75, height=20)
        self.unit_gcp.set(1000)


        '''----------------frame flow velocity II-------------------'''
        frame1 = Frame(note)
        note.add(frame1, text="flow velocity II")

        self.xButton = 370
        self.xText = 250
        self.xText2 = 350
        self.yAddText = 10
        self.yAddText2 = 10

        #parameters feature detection
        Label(frame1, text="Feature detection", font=("Courier", 10)).place(x=10, y=self.yAddText)
 
        self.yAddText = self.yAddText + 20     
        self.lspiv = tk.BooleanVar()
        self.lspiv.set(False)
        self.lspivBut = tk.Checkbutton(frame1, text = "LSPIV", variable=self.lspiv, font=("Helvetica", 10),
                                       command = lambda:self.checkLSPIV())
        self.lspivBut.place(x=self.xText - 250, y=self.yAddText)
   
        self.ptv = tk.BooleanVar()
        self.ptv.set(True)
        self.ptvBut = tk.Checkbutton(frame1, text = "PTV", variable=self.ptv, font=("Helvetica", 10),
                                     command = lambda:self.checkPTV())
        self.ptvBut.place(x=self.xText - 150, y=self.yAddText)    

        self.yAddText = self.yAddText + 25         
        Label(frame1, text="Maximum number features: ").place(x=10, y=self.yAddText)
        self.maxFtNbr_FD = tk.IntVar()
        self.maxFtNbr_FD_Param = Entry(frame1, textvariable=self.maxFtNbr_FD, font=("Helvetica", 10, 'italic'))
        self.maxFtNbr_FD_Param.place(x=self.xText, y=self.yAddText, width=75, height=20)
        self.maxFtNbr_FD.set(1000)  
                
        self.yAddText = self.yAddText + 20
        Label(frame1, text="Minimum feature brightness: ").place(x=10, y=self.yAddText)
        self.minimumThreshBrightness = tk.DoubleVar()
        self.minimumThreshBrightness_Param = Entry(frame1, textvariable=self.minimumThreshBrightness, font=("Helvetica", 10, 'italic'))
        self.minimumThreshBrightness_Param.place(x=self.xText, y=self.yAddText, width=75, height=20)
        self.minimumThreshBrightness.set(70)                        
         
        self.yAddText = self.yAddText + 20
        Label(frame1, text="Neighbor search radius: ").place(x=10, y=self.yAddText)
        self.neighborSearchRadius_FD = tk.IntVar()
        self.neighborSearchRadius_FD_Param = Entry(frame1, textvariable=self.neighborSearchRadius_FD, font=("Helvetica", 10, 'italic'))
        self.neighborSearchRadius_FD_Param.place(x=self.xText, y=self.yAddText, width=75, height=20)
        self.neighborSearchRadius_FD.set(30)                         
         
        self.yAddText = self.yAddText + 20         
        Label(frame1, text="Maximum number neighbors: ").place(x=10, y=self.yAddText)
        self.maximumNeighbors_FD = tk.IntVar()
        self.maximumNeighbors_FD_Param = Entry(frame1, textvariable=self.maximumNeighbors_FD, font=("Helvetica", 10, 'italic'))
        self.maximumNeighbors_FD_Param.place(x=self.xText, y=self.yAddText, width=75, height=20)
        self.maximumNeighbors_FD.set(5)                              

        self.yAddText = self.yAddText + 20         
        Label(frame1, text="Sensitivity feature detection: ").place(x=10, y=self.yAddText)
        self.sensitiveFD = tk.DoubleVar()
        self.sensitiveFD_Param = Entry(frame1, textvariable=self.sensitiveFD, font=("Helvetica", 10, 'italic'))
        self.sensitiveFD_Param.place(x=self.xText, y=self.yAddText, width=75, height=20)
        self.sensitiveFD.set(0.02)

        self.yAddText = self.yAddText + 20         
        Label(frame1, text="PIV cell width: ").place(x=10, y=self.yAddText)
        self.pointDistX = tk.IntVar()
        self.pointDistX_Param = Entry(frame1, textvariable=self.pointDistX, font=("Helvetica", 10, 'italic'))
        self.pointDistX_Param.place(x=self.xText, y=self.yAddText, width=75, height=20)
        self.pointDistX.set(200)
        self.pointDistX_Param.config(state='disabled')
        
        self.yAddText = self.yAddText + 20         
        Label(frame1, text="PIV cell height: ").place(x=10, y=self.yAddText)
        self.pointDistY = tk.IntVar()
        self.pointDistY_Param = Entry(frame1, textvariable=self.pointDistY, font=("Helvetica", 10, 'italic'))
        self.pointDistY_Param.place(x=self.xText, y=self.yAddText, width=75, height=20)
        self.pointDistY.set(200)        
        self.pointDistY_Param.config(state='disabled')

        self.yAddText = self.yAddText + 20            
 
 
        #parameters feature tracking
        self.yAddText2 = self.yAddText - 185
        Label(frame1, text="Feature tracking", font=("Courier", 10)).place(x=self.xText2, y=self.yAddText2)
        
        self.yAddText2 = self.yAddText2 + 20     
        self.lk = tk.BooleanVar()
        self.lk.set(False)
        self.lkBut = tk.Checkbutton(frame1, text = "LK", variable=self.lk, font=("Helvetica", 10),
                                    command = lambda:self.checkLK())
        self.lkBut.place(x=self.xText2-10, y=self.yAddText2)
        
        self.initialLK = tk.BooleanVar()
        self.initialLK.set(False)
        self.initialLKBut = tk.Checkbutton(frame1, text = "Initial Estimates LK", variable=self.initialLK, font=("Helvetica", 10),
                                           command = lambda:self.checkLK())
        self.initialLKBut.place(x=self.xText2 + 45, y=self.yAddText2)
        self.initialLKBut.config(state='disabled')           
   
        self.ncc = tk.BooleanVar()
        self.ncc.set(True)
        self.nccBut = tk.Checkbutton(frame1, text = "NCC", variable=self.ncc, font=("Helvetica", 10),
                                     command = lambda:self.checkNCC())
        self.nccBut.place(x=self.xText2 + 230, y=self.yAddText2)

        self.yAddText2 = self.yAddText2 + 25
        Label(frame1, text="Template width: ").place(x=self.xText2, y=self.yAddText2)
        self.template_width = tk.IntVar()
        self.template_width_Param = Entry(frame1, textvariable=self.template_width, font=("Helvetica", 10, 'italic'))
        self.template_width_Param.place(x=self.xText2 + 240, y=self.yAddText2, width=75, height=20)
        self.template_width.set(7)                  
        
        self.yAddText2 = self.yAddText2 + 20
        Label(frame1, text="Template height: ").place(x=self.xText2, y=self.yAddText2)
        self.template_height = tk.IntVar()
        self.template_height_Param = Entry(frame1, textvariable=self.template_height, font=("Helvetica", 10, 'italic'))
        self.template_height_Param.place(x=self.xText2 + 240, y=self.yAddText2, width=75, height=20)
        self.template_height.set(7)                  
        
        self.yAddText2 = self.yAddText2 + 20
        Label(frame1, text="Search area size x direction: ").place(x=self.xText2, y=self.yAddText2)
        self.search_area_x_CC = tk.IntVar()
        self.search_area_x_CC_Param = Entry(frame1, textvariable=self.search_area_x_CC, font=("Helvetica", 10, 'italic'))
        self.search_area_x_CC_Param.place(x=self.xText2 + 240, y=self.yAddText2, width=75, height=20)
        self.search_area_x_CC.set(24)                
        
        self.yAddText2 = self.yAddText2 + 20
        Label(frame1, text="Search area size y direction: ").place(x=self.xText2, y=self.yAddText2)
        self.search_area_y_CC = tk.IntVar()
        self.search_area_y_CC_Param = Entry(frame1, textvariable=self.search_area_y_CC, font=("Helvetica", 10, 'italic'))
        self.search_area_y_CC_Param.place(x=self.xText2 + 240, y=self.yAddText2, width=75, height=20)
        self.search_area_y_CC.set(24)                 
        
        self.yAddText2 = self.yAddText2 + 20
        Label(frame1, text="Shift search area in x: ").place(x=self.xText2, y=self.yAddText2)
        self.shiftSearchFromCenter_x = tk.IntVar()
        self.shiftSearchFromCenter_x_Param = Entry(frame1, textvariable=self.shiftSearchFromCenter_x, font=("Helvetica", 10, 'italic'))
        self.shiftSearchFromCenter_x_Param.place(x=self.xText2 + 240, y=self.yAddText2, width=75, height=20)
        self.shiftSearchFromCenter_x.set(0)                 
        
        self.yAddText2 = self.yAddText2 + 20
        Label(frame1, text="Shift search area in y: ").place(x=self.xText2, y=self.yAddText2)
        self.shiftSearchFromCenter_y = tk.IntVar()
        self.shiftSearchFromCenter_y_Param = Entry(frame1, textvariable=self.shiftSearchFromCenter_y, font=("Helvetica", 10, 'italic'))
        self.shiftSearchFromCenter_y_Param.place(x=self.xText2 + 240, y=self.yAddText2, width=75, height=20)
        self.shiftSearchFromCenter_y.set(8)                    
        
        self.yAddText2 = self.yAddText2 + 20
        self.subpixel = tk.BooleanVar()
        self.subpixel.set(True)
        self.subpixelBut = tk.Checkbutton(frame1, text = "Subpix", variable=self.subpixel, font=("Helvetica", 10))
        self.subpixelBut.place(x=self.xText2-10, y=self.yAddText2)

        self.performLSM = tk.BooleanVar()
        self.performLSM.set(False)
        self.performLSMBut = tk.Checkbutton(frame1, text = "LSM", variable=self.performLSM, font=("Helvetica", 10))
        self.performLSMBut.place(x=self.xText2 + 67, y=self.yAddText2)

        self.savePlotData = tk.BooleanVar()
        self.savePlotData.set(True)
        self.savePlotDataBut = tk.Checkbutton(frame1, text = "Plot results", variable=self.savePlotData, font=("Helvetica", 10))
        self.savePlotDataBut.place(x=self.xText2 + 126, y=self.yAddText2)
        
        self.yAddText2 = self.yAddText2# + 20
        self.saveGif = tk.BooleanVar()
        self.saveGif.set(True)
        self.saveGifBut = tk.Checkbutton(frame1, text = "Save gif", variable=self.saveGif, font=("Helvetica", 10))
        self.saveGifBut.place(x=self.xText2 + 231, y=self.yAddText2)   
                
         
        #parameters iterations
        self.yAddText = self.yAddText2 + 50
        Label(frame1, text="Iterations", font=("Courier", 10)).place(x=10, y=self.yAddText)
        self.yAddText = self.yAddText + 20
        Label(frame1, text="FD every nth frame: ").place(x=10, y=self.yAddText)
        self.FD_everyIthFrame = tk.IntVar()
        self.FD_everyIthFrame_Param = Entry(frame1, textvariable=self.FD_everyIthFrame, font=("Helvetica", 10, 'italic'))
        self.FD_everyIthFrame_Param.place(x=self.xText, y=self.yAddText, width=75, height=20)
        self.FD_everyIthFrame.set(10)                  
         
        self.yAddText = self.yAddText + 20 
        Label(frame1, text="Track for n frames: ").place(x=10, y=self.yAddText)
        self.FT_forNthNberFrames = tk.IntVar()
        self.FT_forNthNberFrames_Param = Entry(frame1, textvariable=self.FT_forNthNberFrames, font=("Helvetica", 10, 'italic'))
        self.FT_forNthNberFrames_Param.place(x=self.xText, y=self.yAddText, width=75, height=20)
        self.FT_forNthNberFrames.set(20)                 
        
        self.yAddText = self.yAddText + 20
        Label(frame1, text="Track every nth frame: ").place(x=10, y=self.yAddText)
        self.TrackEveryNthFrame = tk.IntVar()
        self.TrackEveryNthFrame_Param = Entry(frame1, textvariable=self.TrackEveryNthFrame, font=("Helvetica", 10, 'italic'))
        self.TrackEveryNthFrame_Param.place(x=self.xText, y=self.yAddText, width=75, height=20)
        self.TrackEveryNthFrame.set(2)

        '''----------------frame flow velocity III-------------------'''
        frame2 = Frame(note)
        note.add(frame2, text="flow velocity III")

        self.xButton = 370
        self.xText = 250
        self.yAddText = 10

        #only filter tracks
        self.filterOnly = tk.BooleanVar()
        self.filterOnly.set(False)
        self.filterOnlyBut = tk.Checkbutton(frame2, text = "Filter only tracks", variable=self.filterOnly, font=("Helvetica", 10),
                                            command = lambda:self.checkFilter())
        self.filterOnlyBut.place(x=self.xText - 250, y=self.yAddText)
        self.filterOnlyBut.config(font=("Helvetica", 10))
         
        #parameters filtering tracks    
        self.yAddText = self.yAddText + 50
        Label(frame2, text="Filtering tracks", font=("Courier", 10)).place(x=10, y=self.yAddText)
        
        self.yAddText = self.yAddText + 20
        Label(frame2, text="Minimum count features [%]: ").place(x=10, y=self.yAddText)
        self.minimumTrackedFeatures = tk.DoubleVar()
        self.minimumTrackedFeatures_Param = Entry(frame2, textvariable=self.minimumTrackedFeatures, font=("Helvetica", 10, 'italic'))
        self.minimumTrackedFeatures_Param.place(x=self.xText, y=self.yAddText, width=75, height=20)
        self.minimumTrackedFeatures.set(66)   
        
        self.yAddText = self.yAddText + 20
        Label(frame2, text="Minimum track distance [px]: ").place(x=10, y=self.yAddText)
        self.minDistance_px = tk.DoubleVar()
        self.minDistance_px_Param = Entry(frame2, textvariable=self.minDistance_px, font=("Helvetica", 10, 'italic'))
        self.minDistance_px_Param.place(x=self.xText, y=self.yAddText, width=75, height=20)
        self.minDistance_px.set(2)                
        
        self.yAddText = self.yAddText + 20
        Label(frame2, text="Maximum track distance [px]: ").place(x=10, y=self.yAddText)
        self.maxDistance_px = tk.DoubleVar()
        self.maxDistance_px_Param = Entry(frame2, textvariable=self.maxDistance_px, font=("Helvetica", 10, 'italic'))
        self.maxDistance_px_Param.place(x=self.xText, y=self.yAddText, width=75, height=20)
        self.maxDistance_px.set(50)            
        
        self.yAddText = self.yAddText + 20
        Label(frame2, text="Steadiness [deg]: ").place(x=10, y=self.yAddText)
        self.threshAngleSteadiness = tk.DoubleVar()
        self.threshAngleSteadiness_Param = Entry(frame2, textvariable=self.threshAngleSteadiness, font=("Helvetica", 10, 'italic'))
        self.threshAngleSteadiness_Param.place(x=self.xText, y=self.yAddText, width=75, height=20)
        self.threshAngleSteadiness.set(25)  
        
        self.yAddText = self.yAddText + 20
        Label(frame2, text="Range track directions [deg]: ").place(x=10, y=self.yAddText)
        self.threshAngleRange = tk.DoubleVar()
        self.threshAngleRange_Param = Entry(frame2, textvariable=self.threshAngleRange, font=("Helvetica", 10, 'italic'))
        self.threshAngleRange_Param.place(x=self.xText, y=self.yAddText, width=75, height=20)
        self.threshAngleRange.set(90)                        
        
        self.yAddText = self.yAddText + 20
        Label(frame2, text="Bin nbr main flow direction: ").place(x=10, y=self.yAddText)
        self.binNbrMainflowdirection = tk.IntVar()
        self.binNbrMainflowdirection_Param = Entry(frame2, textvariable=self.binNbrMainflowdirection, font=("Helvetica", 10, 'italic'))
        self.binNbrMainflowdirection_Param.place(x=self.xText, y=self.yAddText, width=75, height=20)
        self.binNbrMainflowdirection.set(0)                
        
        self.yAddText = self.yAddText + 20
        Label(frame2, text="Buffer main flow [deg]: ").place(x=10, y=self.yAddText)
        self.MainFlowAngleBuffer = tk.DoubleVar()
        self.MainFlowAngleBuffer_Param = Entry(frame2, textvariable=self.MainFlowAngleBuffer, font=("Helvetica", 10, 'italic'))
        self.MainFlowAngleBuffer_Param.place(x=self.xText, y=self.yAddText, width=75, height=20)
        self.MainFlowAngleBuffer.set(10)                                    
        
        self.yAddText = self.yAddText + 20
        Label(frame2, text="setValue velocity threshold: ").place(x=10, y=self.yAddText)
        self.veloStdThresh = tk.DoubleVar()
        self.veloStdThresh_Param = Entry(frame2, textvariable=self.veloStdThresh, font=("Helvetica", 10, 'italic'))
        self.veloStdThresh_Param.place(x=self.xText, y=self.yAddText, width=75, height=20)
        self.veloStdThresh.set(1.5)      
  
        self.yAddText = self.yAddText + 20
        Label(frame2, text="setValue filter radius [pix]: ").place(x=10, y=self.yAddText)
        self.veloFilterSize = tk.DoubleVar()
        self.veloFilterSize_Param = Entry(frame2, textvariable=self.veloFilterSize, font=("Helvetica", 10, 'italic'))
        self.veloFilterSize_Param.place(x=self.xText, y=self.yAddText, width=75, height=20)
        self.veloFilterSize.set(0)     


        '''----------------frame flow velocity IV-------------------'''
        frame3 = Frame(note)
        note.add(frame3, text="flow velocity IV")

        self.xButton = 370
        self.xText = 250
        self.xText2 = 350
        self.yAddText = 10
        self.yAddText2 = 10

        #referencing
        Label(frame3, text="Scaling", font=("Courier", 10)).place(x=10, y=self.yAddText)
        self.yAddText = self.yAddText + 20
        Label(frame3, text="Frame rate: ", font=("Helvetica", 10)).place(x=10, y=self.yAddText)
        self.frame_rate_cam = tk.IntVar()
        self.frame_rate_cam_Param = Entry(frame3, textvariable=self.frame_rate_cam, font=("Helvetica", 10, 'italic'))
        self.frame_rate_cam_Param.place(x=110, y=self.yAddText, width=70, height=20)
        self.frame_rate_cam.set(30)

        #water level
        Label(frame3, text="Define feature search area and set water level", font=("Courier", 10)).place(x=220, y=self.yAddText2)
        self.yAddText2 = self.yAddText2 + 20
        Label(frame3, text="Water level [m]: ", font=("Helvetica", 10)).place(x=220, y=self.yAddText2)
        self.waterlevel_pt = tk.DoubleVar()
        self.waterlevel_pt_Param = Entry(frame3, textvariable=self.waterlevel_pt, font=("Helvetica", 10, 'italic'))    #, state=DISABLED
        self.waterlevel_pt_Param.place(x=340, y=self.yAddText2, width=75, height=20)
        self.waterlevel_pt.set(94.6)
        Label(frame3, text="Buffer [m]: ", font=("Helvetica", 10)).place(x=430, y=self.yAddText2)
        self.waterlevel_buffer = tk.DoubleVar()
        self.waterlevel_buffer_Param = Entry(frame3, textvariable=self.waterlevel_buffer, font=("Helvetica", 10, 'italic'))
        self.waterlevel_buffer_Param.place(x=510, y=self.yAddText2, width=75, height=20)
        self.waterlevel_buffer.set(0.3)    

        self.yAddText2 = self.yAddText2 + 25
        self.importAoIextent = tk.BooleanVar()
        self.importAoIextent.set(False)
        self.importAoIextentBut = tk.Checkbutton(frame3, text = "Import search area file", variable=self.importAoIextent,
                                                 font=("Helvetica", 10), command = lambda:self.checkSearchArea())
        self.importAoIextentBut.place(x=230, y=self.yAddText2)
        self.AoI_file = tk.StringVar()
        self.AoI_file_Param = Entry(frame3, textvariable=self.AoI_file, font=("Helvetica", 10, 'italic'), state='disabled')
        self.AoI_file_Param.place(x=430, y=self.yAddText2, width=200, height=20)
        self.AoI_file_Button = Button(frame3, text = '...', command = lambda:self.select_AoIFile())
        self.AoI_file_Button.place(x=405, y=self.yAddText2, width=20, height=20)

           
        #starting flow velocity estimation
        self.yAddText = self.yAddText + 70
        self.waterlineDetection = Button(frame3, text="Estimate Flow Velocity", style="RB.TButton", command=self.EstimateVelocity)
        self.waterlineDetection.place(x=250, y=self.yAddText+30)
        

        '''----------------frame co-registration-------------------'''
        frame4 = Frame(note)
        note.add(frame4, text="co-registration")
        
        self.xButton = 370
        self.xText = 250
        self.xText2 = 350
        self.yAddText = 10
        self.yAddText2 = 10

        #set parameters for co-registration
        Label(frame4, text="Perform co-registration of frames", font=("Courier", 10)).place(x=10, y=self.yAddText)
        self.yAddText = self.yAddText + 20
        Label(frame4, text="Maximum number of keypoints: ", font=("Helvetica", 10)).place(x=10, y=self.yAddText)
        self.coregist_kpnbr = tk.IntVar()
        self.coregist_kpnbr_Param = Entry(frame4, textvariable=self.coregist_kpnbr, font=("Helvetica", 10, 'italic'))
        self.coregist_kpnbr_Param.place(x=self.xText, y=self.yAddText, width=75, height=20)
        self.coregist_kpnbr.set(5000)
        
        self.yAddText = self.yAddText + 20
        Label(frame4, text="Number of good matches: ", font=("Helvetica", 10)).place(x=10, y=self.yAddText)
        self.nbr_good_matches = tk.IntVar()
        self.nbr_good_matches_Param = Entry(frame4, textvariable=self.nbr_good_matches, font=("Helvetica", 10, 'italic'))
        self.nbr_good_matches_Param.place(x=self.xText, y=self.yAddText, width=75, height=20)
        self.nbr_good_matches.set(10)
       
        self.yAddText = self.yAddText + 20
        self.orb = tk.BooleanVar()
        self.orb.set(True)
        self.orbBut = tk.Checkbutton(frame4, text = "Matching with ORB", variable=self.orb, font=("Helvetica", 10),
                                     command = lambda:self.checkDescriptorORB())
        self.orbBut.place(x=0, y=self.yAddText)

        self.feature_match_twosided = tk.BooleanVar()
        self.feature_match_twosided.set(True)
        self.feature_match_twosidedBut = tk.Checkbutton(frame4, text = "Feature matching 2sided", font=("Helvetica", 10),
                                                        variable=self.feature_match_twosided)
        self.feature_match_twosidedBut.place(x=185, y=self.yAddText)

        self.yAddText = self.yAddText + 20
        self.akaze = tk.BooleanVar()
        self.akaze.set(False)
        self.akazeBut = tk.Checkbutton(frame4, text = "Matching with AKAZE", variable=self.akaze, font=("Helvetica", 10),
                                       command = lambda:self.checkDescriptorAKAZE())
        self.akazeBut.place(x=0, y=self.yAddText)

        self.master_0 = tk.BooleanVar()
        self.master_0.set(True)
        self.master_0But = tk.Checkbutton(frame4, text = "Register to first frame", variable=self.master_0, font=("Helvetica", 10))
        self.master_0But.place(x=185, y=self.yAddText)


        #starting co-registration
        self.yAddText = self.yAddText + 30
        self.coregister = Button(frame4, text="Co-register frames", style="RB.TButton", command=self.coregistration)
        self.coregister.place(x=10, y=self.yAddText)   
        
        
        self.yAddText = self.yAddText + 70             
        Label(frame4, text="Accuracy co-registration", font=("Courier", 10)).place(x=10, y=self.yAddText)
        self.yAddText = self.yAddText + 20
        Label(frame4, text="Template size for co-registration accuracy: ", font=("Helvetica", 10)).place(x=10, y=self.yAddText)
        self.template_size_coregAcc = tk.IntVar()
        self.template_size_coregAcc_Param = Entry(frame4, textvariable=self.template_size_coregAcc, font=("Helvetica", 10, 'italic'))
        self.template_size_coregAcc_Param.place(x=self.xText+70, y=self.yAddText, width=75, height=20)
        self.template_size_coregAcc.set(30)        

        #starting accuracy assessment co-registration
        self.yAddText = self.yAddText + 30
        self.coregisteracc = Button(frame4, text="Accuracy co-registration", style="RB.TButton", command=self.accuracy_coregistration)
        self.coregisteracc.place(x=10, y=self.yAddText) 


    def checkImgSpace(self):
        if self.stayImgSpace.get() == True:
            self.pos_eor_Str_Param.config(state = 'disabled')
            self.angles_eor_Str_Param.config(state = 'disabled')
            self.ransacApproxBut.config(stat = 'disabled')
            self.estimate_exteriorBut.config(stat = 'disabled')
            self.unit_gcp_Param.config(stat = 'disabled')
            self.importAoIextentBut.config(state = 'normal')
            self.ptCloud_file_Param.config(state = 'disabled')
            self.waterlevel_buffer_Param.config(state = 'disabled')
            self.waterlevel_pt_Param.config(state = 'disabled')
            self.AoI_file_Param.config(state = 'normal')
            self.gcpCoo_file_Param.config(state = 'disabled')
            self.imgCoo_GCP_file_Param.config(state = 'disabled')
        else:
            self.checkExterior()
            self.estimate_exteriorBut.config(stat = 'normal')
            self.unit_gcp_Param.config(stat = 'normal')
            self.checkSearchArea()
            self.waterlevel_pt_Param.config(state = 'normal')
            self.gcpCoo_file_Param.config(state = 'normal')
            self.imgCoo_GCP_file_Param.config(state = 'normal')          
    
    def checkExterior(self):
        if self.estimate_exterior.get() == True:
            self.pos_eor_Str_Param.config(state = 'disabled')
            self.angles_eor_Str_Param.config(state = 'disabled')
            self.ransacApproxBut.config(stat = 'normal')           
        else:
            self.pos_eor_Str_Param.config(state = 'normal')
            self.angles_eor_Str_Param.config(state = 'normal')
            self.ransacApproxBut.config(stat = 'disabled')
            
        if self.ransacApprox.get() == True and self.estimate_exterior.get() == True:
            self.pos_eor_Str_Param.config(state = 'disabled')
            self.angles_eor_Str_Param.config(state = 'disabled')   
        elif self.ransacApprox.get() == False and self.estimate_exterior.get() == True:
            self.pos_eor_Str_Param.config(state = 'normal')
            self.angles_eor_Str_Param.config(state = 'normal')       

    def checkLSPIV(self):
        if self.lspiv.get() == True:
            self.maxFtNbr_FD_Param.config(state = 'disabled')
            self.minimumThreshBrightness_Param.config(state = 'disabled')
            self.neighborSearchRadius_FD_Param.config(stat = 'disabled')        
            self.maximumNeighbors_FD_Param.config(state = 'disabled')
            self.sensitiveFD_Param.config(stat = 'disabled')
            self.FT_forNthNberFrames_Param.config(stat = 'disabled')
            self.FT_forNthNberFrames.set(1)
            self.pointDistX_Param.config(state = 'normal')
            self.pointDistY_Param.config(state = 'normal')
            self.minimumTrackedFeatures.set(0)
            self.minimumTrackedFeatures_Param.config(stat = 'disabled') 
            self.ptv.set(False)            
        else:
            self.maxFtNbr_FD_Param.config(state = 'normal')
            self.minimumThreshBrightness_Param.config(state = 'normal')
            self.neighborSearchRadius_FD_Param.config(stat = 'normal')
            self.maximumNeighbors_FD_Param.config(state = 'normal')
            self.sensitiveFD_Param.config(stat = 'normal')
            self.FT_forNthNberFrames_Param.config(stat = 'normal')
            self.pointDistX_Param.config(state = 'disabled')
            self.pointDistY_Param.config(state = 'disabled') 
            self.minimumTrackedFeatures_Param.config(stat = 'normal')
            self.minimumTrackedFeatures.set(66)
            self.ptv.set(True)

    def checkPTV(self):
        if self.ptv.get() == False:
            self.maxFtNbr_FD_Param.config(state = 'disabled')
            self.minimumThreshBrightness_Param.config(state = 'disabled')
            self.neighborSearchRadius_FD_Param.config(stat = 'disabled')        
            self.maximumNeighbors_FD_Param.config(state = 'disabled')
            self.sensitiveFD_Param.config(stat = 'disabled')
            self.FT_forNthNberFrames_Param.config(stat = 'disabled')
            self.FT_forNthNberFrames.set(1)
            self.pointDistX_Param.config(state = 'normal')
            self.pointDistY_Param.config(state = 'normal')
            self.minimumTrackedFeatures.set(0)
            self.minimumTrackedFeatures_Param.config(stat = 'disabled')             
            self.lspiv.set(True)        
        else:
            self.maxFtNbr_FD_Param.config(state = 'normal')
            self.minimumThreshBrightness_Param.config(state = 'normal')
            self.neighborSearchRadius_FD_Param.config(stat = 'normal')
            self.maximumNeighbors_FD_Param.config(state = 'normal')
            self.sensitiveFD_Param.config(stat = 'normal')
            self.FT_forNthNberFrames_Param.config(stat = 'normal')
            self.FT_forNthNberFrames.set(20)
            self.pointDistX_Param.config(state = 'disabled')
            self.pointDistY_Param.config(state = 'disabled')
            self.minimumTrackedFeatures_Param.config(stat = 'normal')
            self.minimumTrackedFeatures.set(66)
            self.lspiv.set(False)                     

    def checkLK(self):
        if self.lk.get() == False:
            self.ncc.set(True)
            self.subpixelBut.config(state='normal')
            self.performLSMBut.config(state='normal')
            self.initialLKBut.config(state='disabled')
            self.search_area_y_CC_Param.config(state='normal')
            self.search_area_x_CC_Param.config(state='normal')
            self.shiftSearchFromCenter_x_Param.config(state = 'normal')  
            self.shiftSearchFromCenter_y_Param.config(state = 'normal')             
        
        elif self.lk.get() == True and self.initialLK.get() == True:
            self.ncc.set(False)
            self.subpixelBut.config(state='disabled')
            self.performLSMBut.config(state='disabled')
            self.initialLKBut.config(state='normal')
            self.search_area_y_CC_Param.config(state='disabled')
            self.search_area_x_CC_Param.config(state='disabled') 
            self.shiftSearchFromCenter_x_Param.config(state = 'normal')  
            self.shiftSearchFromCenter_y_Param.config(state = 'normal') 
            
        elif self.lk.get() == True and self.initialLK.get() == False:
            self.ncc.set(False)
            self.subpixelBut.config(state='disabled')
            self.performLSMBut.config(state='disabled')
            self.initialLKBut.config(state='normal')
            self.search_area_y_CC_Param.config(state='disabled')
            self.search_area_x_CC_Param.config(state='disabled') 
            self.shiftSearchFromCenter_x_Param.config(state = 'disabled')  
            self.shiftSearchFromCenter_y_Param.config(state = 'disabled')        
            
    def checkNCC(self):
        if self.ncc.get() == False:
            self.lk.set(True)
            self.subpixelBut.config(state='disabled')
            self.performLSMBut.config(state='disabled')
            self.initialLKBut.config(state='normal')
            self.search_area_y_CC_Param.config(state='disabled')
            self.search_area_x_CC_Param.config(state='disabled')
            self.shiftSearchFromCenter_x_Param.config(state = 'disabled')  
            self.shiftSearchFromCenter_y_Param.config(state = 'disabled')                    
        else:
            self.lk.set(False)
            self.subpixelBut.config(state='normal')
            self.performLSMBut.config(state='normal')
            self.initialLKBut.config(state='disabled')
            self.search_area_y_CC_Param.config(state='normal')
            self.search_area_x_CC_Param.config(state='normal')
            self.shiftSearchFromCenter_x_Param.config(state = 'normal')  
            self.shiftSearchFromCenter_y_Param.config(state = 'normal')

    def checkDescriptorORB(self):
        if self.orb.get() == True:
            self.akaze.set(False)
        if self.orb.get() == False:
            self.akaze.set(True)

    def checkDescriptorAKAZE(self):
        if self.akaze.get() == True:
            self.orb.set(False)
        if self.akaze.get() == False:
            self.orb.set(True)

    def checkSearchArea(self):
        if self.importAoIextent.get() == True:
            self.AoI_file_Param.config(state = 'normal')
            self.ptCloud_file_Param.config(state = 'disabled')
            self.waterlevel_buffer_Param.config(state = 'disabled')   
        else:
            self.AoI_file_Param.config(state = 'disabled')
            self.ptCloud_file_Param.config(state = 'normal')
            self.waterlevel_buffer_Param.config(state = 'normal')    
            
    def checkFilter(self):
        if self.filterOnly.get() == True:
            self.maxFtNbr_FD_Param.config(state='disabled')
            self.minimumThreshBrightness_Param.config(state='disabled')
            self.neighborSearchRadius_FD_Param.config(state='disabled')
            self.sensitiveFD_Param.config(state='disabled')
            self.maximumNeighbors_FD_Param.config(state='disabled')
            self.template_width_Param.config(state='disabled')
            self.template_height_Param.config(state='disabled')
            self.search_area_x_CC_Param.config(state='disabled')
            self.search_area_y_CC_Param.config(state='disabled')
            self.shiftSearchFromCenter_x_Param.config(state='disabled')
            self.shiftSearchFromCenter_y_Param.config(state='disabled')
            self.subpixelBut.config(state='disabled')
            self.performLSMBut.config(state='disabled')
            self.savePlotDataBut.config(state='disabled')
            self.saveGifBut.config(state='disabled')
            self.FD_everyIthFrame_Param.config(state='disabled')
            self.FT_forNthNberFrames_Param.config(state='disabled')
            self.TrackEveryNthFrame_Param.config(state='disabled')
            self.initialLKBut.config(state='disabled')
            self.lspivBut.config(state='disabled')
            self.ptvBut.config(state='disabled')
            self.lkBut.config(state='disabled')
            self.nccBut.config(state='disabled')
            self.pointDistX_Param.config(state = 'disabled')
            self.pointDistY_Param.config(state = 'disabled')                   
        else:
            self.checkNCC()
            self.checkImgSpace()
            self.checkExterior()
            self.checkLK()
            self.checkLSPIV()
            self.checkPTV()
            self.checkSearchArea()            
            self.template_width_Param.config(state='normal')
            self.template_height_Param.config(state='normal')
            self.savePlotDataBut.config(state='normal')
            self.saveGifBut.config(state='normal')
            self.FD_everyIthFrame_Param.config(state='normal')
            self.TrackEveryNthFrame_Param.config(state='normal')
            self.lspivBut.config(state='normal')
            self.ptvBut.config(state='normal')
            self.lkBut.config(state='normal')
            self.nccBut.config(state='normal')
                     
        

    '''functions to load input data'''
    def select_dirOutput(self):
        outputDir = tk.filedialog.askdirectory(title = 'Select output directory')
        if not outputDir:
            self.directoryOutput.set("")
        else:
            self.directoryOutput.set(outputDir  + '/')
            
    def select_dirImgs(self):
        imgsDir = tk.filedialog.askdirectory(title = 'Select directory of frames')
        if not imgsDir:
            self.dir_imgs.set("")
        else:
            self.dir_imgs.set(imgsDir  + '/')
                                         
    def select_imgName(self):
       imgName = tk.filedialog.askopenfilename(title='Image to draw velocity tracks for visualisation',
                                              initialdir=os.getcwd())
       if not imgName:
           self.img_name.set("")
       else:
           self.img_name.set(imgName)
           
    def select_GCPcooFile(self):
       gcpCooFile = tk.filedialog.askopenfilename(title='Set file with GCP coordinates (object space)',
                                                 filetypes=[('Text file (*.txt)', '*.txt')],initialdir=os.getcwd())
       if not gcpCooFile:
           self.gcpCoo_file.set("")
       else:
           self.gcpCoo_file.set(gcpCooFile)    
           
    def select_GCPimgCooFile(self):
       gcpImgCooFile = tk.filedialog.askopenfilename(title='Set file with GCP coordinates (image space)',
                                                    filetypes=[('Text file (*.txt)', '*.txt')],initialdir=os.getcwd())
       if not gcpImgCooFile:
           self.imgCoo_GCP_file.set("")
       else:
           self.imgCoo_GCP_file.set(gcpImgCooFile)                    

    def select_iorFile(self):
       iorFile = tk.filedialog.askopenfilename(title='Set file with interior camera orientation parameters',
                                              filetypes=[('Text file (*.txt)', '*.txt')],initialdir=os.getcwd())
       if not iorFile:
           self.ior_file.set("")
       else:
           self.ior_file.set(iorFile)
           
    def select_ptClFile(self):
       ptClFile = tk.filedialog.askopenfilename(title='Set file with point cloud of river topography/bathymetry',
                                               filetypes=[('Text file (*.txt)', '*.txt')],initialdir=os.getcwd())
       if not ptClFile:
           self.ptCloud_file.set("")
       else:
           self.ptCloud_file.set(ptClFile)    
           
    def select_AoIFile(self):
       aoiFile = tk.filedialog.askopenfilename(title='Set file with AoI extent coordinates (xy, image space)',
                                              filetypes=[('Text file (*.txt)', '*.txt')],initialdir=os.getcwd())
       if not aoiFile:
           self.AoI_file.set("")
       else:
           self.AoI_file.set(aoiFile)                            


    def printTxt(self, txt):
        self.textbox.insert(END, txt)
        return   

    def safeParemeterSetting(self):
        '''-------logfile-------'''
        listParams = ['outputDir ' + self.directoryOutput.get(),
                       'imagesDir ' + self.dir_imgs.get(),
                       'gcpObjSpace ' + self.gcpCoo_file.get(),
                       'gcpImgSpace ' + self.imgCoo_GCP_file.get(),
                       'interior ' + self.ior_file.get(),
                       '3DPtCl ' + self.ptCloud_file.get(),
                       'ImgFileIllustr ' + self.img_name.get(),
                       'EstimateExt ' + str(self.estimate_exterior.get()),
                       'UseRansac ' + str(self.ransacApprox.get()),
                       'ApproxPos ' + self.pos_eor_Str.get(),
                       'ApproxOri ' + self.angles_eor_Str.get(),
                       'unitGCP ' + str(self.unit_gcp.get()),
                       'maxFeatNbr ' + str(self.maxFtNbr_FD.get()),
                       'minFeatBright ' + str(self.minimumThreshBrightness.get()),
                       'NeighSearchRad ' + str(self.neighborSearchRadius_FD.get()),
                       'maxNN ' + str(self.maximumNeighbors_FD.get()),
                       'sensFeatDetect ' + str(self.sensitiveFD.get()),
                       'TemplWidth ' + str(self.template_width.get()),
                       'TemplHeight ' + str(self.template_height.get()),
                       'SearchAreaX ' + str(self.search_area_x_CC.get()),
                       'SearchAreaY ' + str(self.search_area_y_CC.get()),
                       'ShiftX ' + str(self.shiftSearchFromCenter_x.get()),
                       'ShiftY ' + str(self.shiftSearchFromCenter_y.get()),
                       'Subpix ' + str(self.subpixel.get()),
                       'LSM ' + str(self.performLSM.get()),
                       'PlotResults ' + str(self.savePlotData.get()),
                       'SaveGif ' + str(self.saveGif.get()),
                       'FDnthFrame ' + str(self.FD_everyIthFrame.get()),
                       'FTnFrames ' + str(self.FT_forNthNberFrames.get()),
                       'FTevernFrames ' + str(self.TrackEveryNthFrame.get()),
                       'minCountFeat ' + str(self.minimumTrackedFeatures.get()),
                       'minTrackDist ' + str(self.minDistance_px.get()),
                       'maxTrackDist ' + str(self.maxDistance_px.get()),
                       'Steady ' + str(self.threshAngleSteadiness.get()),
                       'RangeTrackDir ' + str(self.threshAngleRange.get()),
                       'binNbrMainFlow ' + str(self.binNbrMainflowdirection.get()),
                       'BufferMainFlow ' + str(self.MainFlowAngleBuffer.get()),
                       'setValVelo ' + str(self.veloStdThresh.get()),
                       'FrameRate ' + str(self.frame_rate_cam.get()),
                       'Waterlevel ' + str(self.waterlevel_pt.get()),
                       'Buffer ' + str(self.waterlevel_buffer.get()),
                       'ImportAoI ' + str(self.importAoIextent.get()),
                       'AoIFile ' + self.AoI_file.get(),
                       'LK ' + str(self.lk.get()),
                       'NCC ' + str(self.ncc.get()),
                       'initalEstimLK ' + str(self.initialLK.get()),
                       'LSPIV ' + str(self.lspiv.get()),
                       'PTV ' + str(self.ptv.get()),
                       'PIVpointDistX ' + str(self.pointDistX.get()),
                       'PIVpointDistY ' + str(self.pointDistY.get()),
                       'stayImgSpace ' + str(self.stayImgSpace.get())]
        
        listParams = pd.DataFrame(listParams)
        listParams.to_csv(self.directoryOutput.get() + 'parameterSettings.txt', index=False, header=None)
        print('parameters saved')   
        self.printTxt('parameters saved')
        
    def loadParameterSettings(self):
        paramFile = tk.filedialog.askopenfilename(title='Load parameter file',
                                                 filetypes=[('Text file (*.txt)', '*.txt')],initialdir=os.getcwd())
        listParams = pd.read_csv(paramFile, sep=' ', header=None)
        
        self.directoryOutput.set(listParams.iloc[0,1])
        self.dir_imgs.set(listParams.iloc[1,1])
        self.gcpCoo_file.set(listParams.iloc[2,1])
        self.imgCoo_GCP_file.set(listParams.iloc[3,1])
        self.ior_file.set(listParams.iloc[4,1])
        self.ptCloud_file.set(listParams.iloc[5,1])
        self.img_name.set(listParams.iloc[6,1])
        self.estimate_exterior.set(listParams.iloc[7,1])
        self.ransacApprox.set(listParams.iloc[8,1])
        posEor = listParams.iloc[9,0]
        posEor = posEor.split(' ')
        self.pos_eor_Str.set(posEor[1])
        anglesEor = listParams.iloc[10,0]
        anglesEor = anglesEor.split(' ')
        self.angles_eor_Str.set(anglesEor[1])     
        self.unit_gcp.set(listParams.iloc[11,1])         
        self.maxFtNbr_FD.set(listParams.iloc[12,1])
        self.minimumThreshBrightness.set(listParams.iloc[13,1])
        self.neighborSearchRadius_FD.set(listParams.iloc[14,1])
        self.maximumNeighbors_FD.set(listParams.iloc[15,1])
        self.sensitiveFD.set(listParams.iloc[16,1])
        self.template_width.set(listParams.iloc[17,1])
        self.template_height.set(listParams.iloc[18,1])
        self.search_area_x_CC.set(listParams.iloc[19,1])
        self.search_area_y_CC.set(listParams.iloc[20,1])
        self.shiftSearchFromCenter_x.set(listParams.iloc[21,1])
        self.shiftSearchFromCenter_y.set(listParams.iloc[22,1])
        self.subpixel.set(listParams.iloc[23,1])
        self.performLSM.set(listParams.iloc[24,1])
        self.savePlotData.set(listParams.iloc[25,1])
        self.saveGif.set(listParams.iloc[26,1])
        self.FD_everyIthFrame.set(listParams.iloc[27,1])
        self.FT_forNthNberFrames.set(listParams.iloc[28,1])
        self.TrackEveryNthFrame.set(listParams.iloc[29,1])
        self.minimumTrackedFeatures.set(listParams.iloc[30,1])
        self.minDistance_px.set(listParams.iloc[31,1])
        self.maxDistance_px.set(listParams.iloc[32,1])
        self.threshAngleSteadiness.set(listParams.iloc[33,1])
        self.threshAngleRange.set(listParams.iloc[34,1])
        self.binNbrMainflowdirection.set(listParams.iloc[35,1])
        self.MainFlowAngleBuffer.set(listParams.iloc[36,1])
        self.veloStdThresh.set(listParams.iloc[37,1])
        self.frame_rate_cam.set(listParams.iloc[38,1])
        self.waterlevel_pt.set(listParams.iloc[39,1])
        self.waterlevel_buffer.set(listParams.iloc[40,1])
        self.importAoIextent.set(listParams.iloc[41,1])
        self.AoI_file.set(listParams.iloc[42,1])
        self.lk.set(listParams.iloc[43,1])
        self.ncc.set(listParams.iloc[44,1])
        self.initialLK.set(listParams.iloc[45,1])
        self.lspiv.set(listParams.iloc[46,1])
        self.ptv.set(listParams.iloc[47,1])
        self.pointDistX.set(listParams.iloc[48,1])
        self.pointDistY.set(listParams.iloc[49,1])
        self.stayImgSpace.set(listParams.iloc[50,1])

        print('parameters loaded')
        self.printTxt('parameters loaded')


    def printTxt(self, txt):
        self.textbox.insert(tk.END, txt)
        return
    

    '''functions for data processing'''
    def accuracy_coregistration(self):
        #read parameters from directories
        failing = True
        while failing:
            try:
                directoryOutput_coreg_acc = tk.filedialog.askdirectory(title='Output directory accuracy results') + '/'
                
                image_list_coreg = tk.filedialog.askopenfilenames(title='Open co-registered frames')
                
                check_points_forAccCoreg =  tk.filedialog.askopenfilename(title='File with CP coordinates (image space)',
                                                                         filetypes=[('Text file (*.txt)', '*.txt')],initialdir=os.getcwd())
                         
                failing = False
            
            except Exception as e:
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print('failed reading data, please try again\n')
            
        image_list_coreg = sorted(image_list_coreg, key=lambda image: (image[0], image[1]))
        try:
            img_check_pts = np.asarray(pd.read_table(check_points_forAccCoreg), dtype=np.float32)
        except:
            print('failed reading file with check points in image space')
        
        coregF.accuracy_coregistration(image_list_coreg, img_check_pts, self.template_size_coregAcc.get(), directoryOutput_coreg_acc)        
        
        print('Accuracy assessment co-registration finished.')               
        
    
    def coregistration(self):        
        #read parameters from directories
        print(cv2.__version__)
        failing = True
        while failing:
            try:
                directoryOutput_coreg = tk.filedialog.askdirectory(title='Output directory co-registration') + '/'
                
                image_list = tk.filedialog.askopenfilenames(title='Open frames for co-registration')
                                         
                failing = False
            
            except Exception as e:
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print('failed reading data, please try again\n')
        

        image_list = sorted(image_list, key=lambda image: (image[0], image[1]))

        if self.orb.get() == True:
            descrVers = "orb"
        else:
            descrVers = "akaze"

        coregF.coregistration(image_list, directoryOutput_coreg, self.coregist_kpnbr.get(), descrVers,
                              self.feature_match_twosided.get(), self.nbr_good_matches.get())      
                
        print('Co-registration finished.')
        self.printTxt('------------------------------------------\n'
                      'finished co-registration\n')
        

    def EstimateVelocity(self):
        '''-------set parameters-------'''
        test_run = self.test_run.get()
        
        if not self.stayImgSpace.get():
            #parameters exterior orientation estimation
            estimate_exterior = self.estimate_exterior.get()
            if (estimate_exterior == False) or (estimate_exterior == True and self.ransacApprox.get() == False):
                angles_eor_Str = self.angles_eor_Str.get()
                angles_eor = angles_eor_Str.split(',')
                angles_eor = np.asarray([float(x) for x in angles_eor]).reshape(3,1) 
                
                pos_eor_Str = self.pos_eor_Str.get()
                pos_eor = pos_eor_Str.split(',')
                pos_eor = np.asarray([float(x) for x in pos_eor]).reshape(3,1)
    
                self.ransacApprox.set(False)
    
            else:
                angles_eor = np.zeros((3,1))
                pos_eor = np.zeros((3,1))       
            
            unit_gcp = self.unit_gcp.get()
            max_orientation_deviation = 1
            ransacApprox = self.ransacApprox.get()

        #parameters feature detection
        minimumThreshBrightness = self.minimumThreshBrightness.get()
        neighborSearchRadius_FD = self.neighborSearchRadius_FD.get()
        maximumNeighbors_FD = self.maximumNeighbors_FD.get()
        maxFtNbr_FD = self.maxFtNbr_FD.get()
        sensitiveFD = self.sensitiveFD.get()
        pointDistX = self.pointDistX.get()
        pointDistY = self.pointDistX.get()

        #parameters tracking
        threshLSM = 0.001  #for adjustment
        lsmBuffer = 3 #increases lsm search area compared to patch
        template_width = self.template_width.get() #has to be odd
        template_height = self.template_height.get() #has to be odd
        search_area_x_CC = self.search_area_x_CC.get()
        search_area_y_CC = self.search_area_y_CC.get()
        shiftSearchFromCenter_x = self.shiftSearchFromCenter_x.get()
        shiftSearchFromCenter_y = self.shiftSearchFromCenter_y.get()
        subpixel = self.subpixel.get()

        performLSM = self.performLSM.get()
        savePlotData = self.savePlotData.get()
        save_gif = self.saveGif.get()

        #parameters iterations
        FD_everyIthFrame = self.FD_everyIthFrame.get()
        FT_forNthNberFrames = self.FT_forNthNberFrames.get()
        TrackEveryNthFrame = self.TrackEveryNthFrame.get()

        #params referencing
        frame_rate_cam = self.frame_rate_cam.get()         

        #params filter tracks
        threshAngleSteadiness = self.threshAngleSteadiness.get()
        threshAngleRange = self.threshAngleRange.get()
        binNbrMainflowdirection = self.binNbrMainflowdirection.get()
        MainFlowAngleBuffer =  self.MainFlowAngleBuffer.get()
        veloStdThresh = self.veloStdThresh.get()
        minDistance_px = self.minDistance_px.get()   #in pixel
        maxDistance_px = self.maxDistance_px.get()
        minimumTrackedFeatures = self.minimumTrackedFeatures.get()
        minimumTrackedFeatures = np.int(FT_forNthNberFrames*(minimumTrackedFeatures/100)/TrackEveryNthFrame)


        '''-------read data and prepare for following processing-------'''        
        #read parameters from directories
        directoryOutput = self.directoryOutput.get()
        dir_imgs = self.dir_imgs.get()
        img_name = self.img_name.get()
        ior_file = self.ior_file.get()
        
        if not self.stayImgSpace.get():
            gcpCoo_file = self.gcpCoo_file.get()
            imgCoo_GCP_file = self.imgCoo_GCP_file.get()
            waterlevel_pt = self.waterlevel_pt.get()  #float
            waterlevel_buffer = self.waterlevel_buffer.get()            

            #parameters search area definition                            
            if not self.importAoIextent.get():
                ptCloud_file = self.ptCloud_file.get()

                AoI_file = None
                
                try:
                    ptCloud = np.asarray(pd.read_table(ptCloud_file, header=None, delimiter=',')) #read point cloud
                except:
                    print('failed reading point cloud file')
                
            else:
                ptCloud = []
                AoI_file = self.AoI_file.get()

        else:
            self.importAoIextent.set(True)
            gcpCoo_file = None
            imgCoo_GCP_file = None
            unit_gcp = None
            ptCloud = []
            waterlevel_pt = np.nan
            waterlevel_buffer = np.nan              
            AoI_file = self.AoI_file.get()

        try:
            interior_orient = photogrF.read_aicon_ior(ior_file) #read interior orientation from file (aicon)   
        except:
            print('failed reading interior orientation file')        

        try:
            img_list = ioF.read_imgs_folder(dir_imgs) #read image names in folder
        except:
            print('failed reading images from folder')
    
        #prepare output
        if not os.path.exists(directoryOutput):
            os.system('mkdir ' + directoryOutput)
            
        print('all input data read\n')
        print('------------------------------------------')
        self.printTxt('------------------------------------------\n'
                      'finished reading input data')
                
                
        '''-------get exterior camera geometry-------'''
        if not self.stayImgSpace.get():
            eor_mat = ptv.EstimateExterior(gcpCoo_file, imgCoo_GCP_file, interior_orient, estimate_exterior,
                                           unit_gcp, max_orientation_deviation, ransacApprox, angles_eor, pos_eor,
                                           directoryOutput)
        else:
            eor_mat = None

        '''define search area for features'''
        searchMask = ptv.searchMask(waterlevel_pt, waterlevel_buffer, AoI_file, ptCloud, unit_gcp, interior_orient,
                                    eor_mat, savePlotData, directoryOutput, img_list, self.importAoIextent.get())
        
        
        '''-------perform feature detection-------'''
        if self.filterOnly.get() == False:
            frameCount = 0
            imagesForGif = []
            trackedFeaturesOutput_undist = []
            first_loop = True
            
            if test_run and len(img_list) > 20:
                lenLoop = 20
            elif len(img_list) > FT_forNthNberFrames+1:
                lenLoop = len(img_list)-FT_forNthNberFrames-1
            else:
                lenLoop = 1

            if self.lspiv.get():
                featuresToTrack, first_loop, feature_ID_max = ptv.FeatureDetectionLSPIV(dir_imgs, img_list, frameCount, pointDistX, pointDistY, searchMask, 
                                                                                        FD_everyIthFrame, savePlotData, directoryOutput, first_loop, None)
                
            while frameCount < lenLoop:
                
                if frameCount % FD_everyIthFrame == 0:
                    
                    if first_loop:
                        feature_ID_max = None
                    
                    if self.ptv.get():
                        featuresToTrack, first_loop, feature_ID_max = ptv.FeatureDetectionPTV(dir_imgs, img_list, frameCount, minimumThreshBrightness, neighborSearchRadius_FD,
                                                                                              searchMask, maximumNeighbors_FD, maxFtNbr_FD, sensitiveFD, savePlotData, directoryOutput,
                                                                                              FD_everyIthFrame, first_loop, feature_ID_max)
                
                    print('features detected\n')
                    print('------------------------------------------')
                    
                    
                    '''-------perform feature tracking-------'''
                    trackedFeaturesOutput_undist, imagesForGif = ptv.FeatureTracking(template_width, template_height, search_area_x_CC, search_area_y_CC, shiftSearchFromCenter_x, shiftSearchFromCenter_y,
                                                                                     frameCount, FT_forNthNberFrames, TrackEveryNthFrame, dir_imgs, img_list, featuresToTrack, interior_orient,
                                                                                     performLSM, lsmBuffer, threshLSM, subpixel, trackedFeaturesOutput_undist, save_gif, imagesForGif, directoryOutput,
                                                                                     self.lk.get(), self.initialLK.get())
    
                frameCount = frameCount + 1
    
            #write tracked features to file
            ioF.writeOutput(trackedFeaturesOutput_undist, FT_forNthNberFrames, FD_everyIthFrame, directoryOutput)
    
            #save gif
            if save_gif:
                print('save tracking result to gif\n')
                #global imageio 
                imageio.mimsave(directoryOutput + 'trackedFeatures.gif', imagesForGif)
                #del imageio
            print('feature tracking done\n')
            print('------------------------------------------')
            self.printTxt('------------------------------------------\n'
                          'finished feature tracking')
        
        else:
            trackFile = tk.filedialog.askopenfilename(title='File with tracks', filetypes=[('Text file (*.txt)', '*.txt')],initialdir=os.getcwd())
            try:
                trackedFeaturesOutput_undist = pd.read_table(trackFile)
            except:
                print('failed reading feature file')


        '''-------filter tracking results in image space-------'''        
        filteredFeatures, [nbr_features_raw, nbr_features_mindist, 
                           nbr_features_maxdist,minimumTrackedFeatures,steady_angle, 
                           nbr_features_steady,range_angle, nbr_features_rangeangle, 
                           flowdir_angle,nbr_features_mainflowdir] = ptv.FilterTracks(trackedFeaturesOutput_undist, img_name, directoryOutput,
                                                                                      minDistance_px, maxDistance_px, minimumTrackedFeatures, 
                                                                                      threshAngleSteadiness, threshAngleRange,
                                                                                      binNbrMainflowdirection, MainFlowAngleBuffer, self.lspiv.get())

        print('filtering tracks done\n')
        print('------------------------------------------')
        self.printTxt('------------------------------------------\n'
                      'finished filtering tracks\n')


        '''-------transform img measurements into object space-------'''
        if not self.stayImgSpace.get():
            ptv.TracksPx_to_TracksMetric(filteredFeatures, minimumTrackedFeatures, interior_orient, eor_mat, unit_gcp,
                                         frame_rate_cam, TrackEveryNthFrame, waterlevel_pt, directoryOutput, img_name, 
                                         veloStdThresh, self.lspiv.get(), self.veloFilterSize.get(), searchMask)
            self.printTxt('------------------------------------------\n'
                          'finished transforming pixel values to velocities')
        
        
        '''-------logfile-------'''
        if self.filterOnly.get() == False and self.stayImgSpace.get() == False:
            log_file_writer, logfile = ioF.logfile_writer(directoryOutput + 'logfile.txt')
            log_file_writer.writerow([['test run: ', test_run],['exterior angles: ', angles_eor],['exterior position: ', pos_eor],
                                     ['unit_gcp: ', unit_gcp],['use ransacApprox: ', ransacApprox],
                                     ['waterlevel: ',waterlevel_pt],['waterlevel_buffer: ',waterlevel_buffer],
                                     ['minimumThreshBrightness: ',minimumThreshBrightness],['neighborSearchRadius_FD: ',neighborSearchRadius_FD],
                                     ['maximumNeighbors_FD :',maximumNeighbors_FD],['maxFtNbr_FD :',maxFtNbr_FD],['sensitiveFD: ',sensitiveFD],
                                     ['template_width: ',template_width],['template_height: ',template_height],['search_area_x_CC: ',search_area_x_CC],['search_area_y_CC: ',search_area_y_CC], 
                                     ['shiftSearchFromCenter_x: ',shiftSearchFromCenter_x],['shiftSearchFromCenter_y: ',shiftSearchFromCenter_y],
                                     ['subpixel: ',subpixel],['performLSM: ',performLSM],['FD_everyIthFrame: ',FD_everyIthFrame],['FT_forNthNberFrames: ',FT_forNthNberFrames],
                                     ['TrackEveryNthFrame: ',TrackEveryNthFrame],['frame_rate_cam: ',frame_rate_cam],
                                     ['minDistance_px: ',minDistance_px],['nbr features min dist: ',nbr_features_mindist],
                                     ['maxDistance_px: ',maxDistance_px],['nbr features max dist: ',nbr_features_maxdist],
                                     ['minimumTrackedFeatures: ',minimumTrackedFeatures],
                                     ['threshAngleSteadiness: ',threshAngleSteadiness],['nbr features steadyness: ', nbr_features_steady],['average angle steadiness: ', steady_angle],
                                     ['threshAngleRange: ',threshAngleRange],['nbr features angle range: ',nbr_features_rangeangle],['average range angle: ', range_angle],
                                     ['binNbrMainflowdirection: ',binNbrMainflowdirection],['MainFlowAngleBuffer: ',MainFlowAngleBuffer],
                                     ['nbr features main flow direction: ', nbr_features_mainflowdir],['median angle flow direction: ', flowdir_angle],
                                     ['veloStdThresh: ',veloStdThresh],['nbr filtered features: ', filteredFeatures.shape[0]],['nbr raw features: ',nbr_features_raw]])
            logfile.flush()
            logfile.close()
        
        elif self.filterOnly.get() == True and self.stayImgSpace.get() == True:
            log_file_writer, logfile = ioF.logfile_writer(directoryOutput + 'logfileFilterImgspace.txt')
            log_file_writer.writerow([['frame_rate_cam: ',frame_rate_cam],
                                     ['minDistance_px: ',minDistance_px],['nbr features min dist: ',nbr_features_mindist],
                                     ['maxDistance_px: ',maxDistance_px],['nbr features max dist: ',nbr_features_maxdist],
                                     ['minimumTrackedFeatures: ',minimumTrackedFeatures],
                                     ['threshAngleSteadiness: ',threshAngleSteadiness],['nbr features steadyness: ', nbr_features_steady],['average angle steadiness: ', steady_angle],
                                     ['threshAngleRange: ',threshAngleRange],['nbr features angle range: ',nbr_features_rangeangle],['average range angle: ', range_angle],
                                     ['binNbrMainflowdirection: ',binNbrMainflowdirection],['MainFlowAngleBuffer: ',MainFlowAngleBuffer],
                                     ['nbr features main flow direction: ', nbr_features_mainflowdir],['median angle flow direction: ', flowdir_angle],
                                     ['nbr filtered features: ', filteredFeatures.shape[0]],['nbr raw features: ',nbr_features_raw]])
            logfile.flush()
            logfile.close()    

        elif self.filterOnly.get() == True and self.stayImgSpace.get() == False:
            log_file_writer, logfile = ioF.logfile_writer(directoryOutput + 'logfileFilter.txt')
            log_file_writer.writerow([['exterior angles: ', angles_eor],['exterior position: ', pos_eor],
                                     ['unit_gcp: ', unit_gcp],['use ransacApprox: ', ransacApprox],
                                     ['waterlevel: ',waterlevel_pt],['waterlevel_buffer: ',waterlevel_buffer],
                                     ['frame_rate_cam: ',frame_rate_cam],
                                     ['minDistance_px: ',minDistance_px],['nbr features min dist: ',nbr_features_mindist],
                                     ['maxDistance_px: ',maxDistance_px],['nbr features max dist: ',nbr_features_maxdist],
                                     ['minimumTrackedFeatures: ',minimumTrackedFeatures],
                                     ['threshAngleSteadiness: ',threshAngleSteadiness],['nbr features steadyness: ', nbr_features_steady],['average angle steadiness: ', steady_angle],
                                     ['threshAngleRange: ',threshAngleRange],['nbr features angle range: ',nbr_features_rangeangle],['average range angle: ', range_angle],
                                     ['binNbrMainflowdirection: ',binNbrMainflowdirection],['MainFlowAngleBuffer: ',MainFlowAngleBuffer],
                                     ['nbr features main flow direction: ', nbr_features_mainflowdir],['median angle flow direction: ', flowdir_angle],
                                     ['veloStdThresh: ',veloStdThresh],['nbr filtered features: ', filteredFeatures.shape[0]],['nbr raw features: ',nbr_features_raw]])
            logfile.flush()
            logfile.close()
            
        if self.filterOnly.get() == False and self.stayImgSpace.get() == True:
            log_file_writer, logfile = ioF.logfile_writer(directoryOutput + 'logfileImgSpace.txt')
            log_file_writer.writerow([['test run: ', test_run],
                                     ['minimumThreshBrightness: ',minimumThreshBrightness],['neighborSearchRadius_FD: ',neighborSearchRadius_FD],
                                     ['maximumNeighbors_FD :',maximumNeighbors_FD],['maxFtNbr_FD :',maxFtNbr_FD],['sensitiveFD: ',sensitiveFD],
                                     ['template_width: ',template_width],['template_height: ',template_height],['search_area_x_CC: ',search_area_x_CC],['search_area_y_CC: ',search_area_y_CC], 
                                     ['shiftSearchFromCenter_x: ',shiftSearchFromCenter_x],['shiftSearchFromCenter_y: ',shiftSearchFromCenter_y],
                                     ['subpixel: ',subpixel],['performLSM: ',performLSM],['FD_everyIthFrame: ',FD_everyIthFrame],['FT_forNthNberFrames: ',FT_forNthNberFrames],
                                     ['TrackEveryNthFrame: ',TrackEveryNthFrame],['frame_rate_cam: ',frame_rate_cam],
                                     ['minDistance_px: ',minDistance_px],['nbr features min dist: ',nbr_features_mindist],
                                     ['maxDistance_px: ',maxDistance_px],['nbr features max dist: ',nbr_features_maxdist],
                                     ['minimumTrackedFeatures: ',minimumTrackedFeatures],
                                     ['threshAngleSteadiness: ',threshAngleSteadiness],['nbr features steadyness: ', nbr_features_steady],['average angle steadiness: ', steady_angle],
                                     ['threshAngleRange: ',threshAngleRange],['nbr features angle range: ',nbr_features_rangeangle],['average range angle: ', range_angle],
                                     ['binNbrMainflowdirection: ',binNbrMainflowdirection],['MainFlowAngleBuffer: ',MainFlowAngleBuffer],
                                     ['nbr features main flow direction: ', nbr_features_mainflowdir],['median angle flow direction: ', flowdir_angle],
                                     ['nbr filtered features: ', filteredFeatures.shape[0]],['nbr raw features: ',nbr_features_raw]])
            logfile.flush()
            logfile.close()


        print('finished\n')
        print('------------------------------------------')
        self.printTxt('------------------------------------------\n'
                      'finished velocity estimations')
        
 
def main():        

    root = tk.Tk()

    size_width = str(root.winfo_screenwidth())
    size_height = str(root.winfo_screenheight())
    root.geometry(size_width + "x" + size_height)
    
    app = FlowVeloTool(root)   
    
    root.mainloop()
    
    #root.destroy() # optional; see description below        


if __name__ == "__main__":
    main()  