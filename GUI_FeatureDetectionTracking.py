#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'Anette Eltner'
__contact__ = 'Anette.Eltner@tu-dresden.de'
__copyright__ = '(c) Anette Eltner 2019'
__license__ = 'MIT'
__date__ = '03 May 2019'
__version__ = '1.0'
__status__ = "initial release"
__url__ = "https://github.???"


"""
Name:           flowvelo_gui.py
Compatibility:  Python 2.7
Description:    This program performs image-based flow velocity estimation. It includes
                camera orientation estimation, feature detection and tracking, 
                and image co-registration.
URL:            https://github.???
Requires:       Tkinter, scipy 1.1.0, scikit-learn 0.18.2, scikit-image 0.13.1, shapely, imageio 2.3.0,
                opencv 3.2.0, seaborn 0.8.1, matplotlib 2.2.2
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


import Tkinter as tk
import tkFileDialog, ScrolledText
from ttk import *

#global imageio 

class FlowVeloTool:
    
    def __init__(self, master):
        
        master_frame = Frame(master, name='master_frame')
        master.title('Image-based flow velocity estimation')
        note = Notebook(master_frame, name='note')
        master_frame.grid()
        
        #text box for display output
        self.textbox = ScrolledText.ScrolledText(master, height=10, width=20)
        self.textbox.place(x=700, y=50, width=300, height=800)
                                
        
        '''----------------frame flow velocity-------------------'''
        frame = Frame(note)  
        note.add(frame, text="flow velocity")
        note.grid(row=0, column=0, ipadx=500, ipady=440)
        
        self.xButton = 370
        self.xText = 250
        self.xText2 = 350
        self.yAddText = 10
        self.yAddText2 = 10
        Style().configure("RB.TButton", foreground='blue', font=('helvetica', 10))

        currentDirectory = os.getcwd()

        #test run
        self.test_run = tk.BooleanVar()
        self.test_run.set(False)
        self.test_runBut = tk.Checkbutton(frame, text = "Test run?", font=("Helvetica", 10), variable=self.test_run)
        self.test_runBut.place(x=540, y=5)        
        
        #load files
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
        self.yAddText = self.yAddText + 40
        Label(frame, text="Exterior orientation", font=("Courier", 10)).place(x=10, y=self.yAddText)
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
         

        #parameter feature detection       
        self.yAddText = self.yAddText + 40
        Label(frame, text="Feature detection", font=("Courier", 10)).place(x=10, y=self.yAddText)
 
        self.yAddText = self.yAddText + 20     
        self.lspiv = tk.BooleanVar()
        self.lspiv.set(False)
        self.lspivBut = tk.Checkbutton(frame, text = "LSPIV", variable=self.lspiv, font=("Helvetica", 10), 
                                       command = lambda:self.checkLSPIV())
        self.lspivBut.place(x=self.xText - 250, y=self.yAddText)
   
        self.ptv = tk.BooleanVar()
        self.ptv.set(True)
        self.ptvBut = tk.Checkbutton(frame, text = "PTV", variable=self.ptv, font=("Helvetica", 10),
                                     command = lambda:self.checkPTV())
        self.ptvBut.place(x=self.xText - 150, y=self.yAddText)    

        self.yAddText = self.yAddText + 25         
        Label(frame, text="Maximum number features: ").place(x=10, y=self.yAddText)
        self.maxFtNbr_FD = tk.IntVar()
        self.maxFtNbr_FD_Param = Entry(frame, textvariable=self.maxFtNbr_FD, font=("Helvetica", 10, 'italic'))
        self.maxFtNbr_FD_Param.place(x=self.xText, y=self.yAddText, width=75, height=20)
        self.maxFtNbr_FD.set(1000)  
                
        self.yAddText = self.yAddText + 20
        Label(frame, text="Minimum feature brightness: ").place(x=10, y=self.yAddText)        
        self.minimumThreshBrightness = tk.DoubleVar()
        self.minimumThreshBrightness_Param = Entry(frame, textvariable=self.minimumThreshBrightness, font=("Helvetica", 10, 'italic'))
        self.minimumThreshBrightness_Param.place(x=self.xText, y=self.yAddText, width=75, height=20)
        self.minimumThreshBrightness.set(70)                        
         
        self.yAddText = self.yAddText + 20
        Label(frame, text="Neighbor search radius: ").place(x=10, y=self.yAddText)
        self.neighborSearchRadius_FD = tk.IntVar()
        self.neighborSearchRadius_FD_Param = Entry(frame, textvariable=self.neighborSearchRadius_FD, font=("Helvetica", 10, 'italic'))
        self.neighborSearchRadius_FD_Param.place(x=self.xText, y=self.yAddText, width=75, height=20)
        self.neighborSearchRadius_FD.set(30)                         
         
        self.yAddText = self.yAddText + 20         
        Label(frame, text="Maximum number neighbors: ").place(x=10, y=self.yAddText)
        self.maximumNeighbors_FD = tk.IntVar()
        self.maximumNeighbors_FD_Param = Entry(frame, textvariable=self.maximumNeighbors_FD, font=("Helvetica", 10, 'italic'))
        self.maximumNeighbors_FD_Param.place(x=self.xText, y=self.yAddText, width=75, height=20)
        self.maximumNeighbors_FD.set(5)                              

        self.yAddText = self.yAddText + 20         
        Label(frame, text="Sensitivity feature detection: ").place(x=10, y=self.yAddText)
        self.sensitiveFD = tk.DoubleVar()
        self.sensitiveFD_Param = Entry(frame, textvariable=self.sensitiveFD, font=("Helvetica", 10, 'italic'))
        self.sensitiveFD_Param.place(x=self.xText, y=self.yAddText, width=75, height=20)
        self.sensitiveFD.set(0.02)                      
 
        self.yAddText = self.yAddText + 20            
 
 
        #parameters feature tracking
        self.yAddText2 = self.yAddText - 145
        Label(frame, text="Feature tracking", font=("Courier", 10)).place(x=self.xText2, y=self.yAddText2)
        
        self.yAddText2 = self.yAddText2 + 20     
        self.lk = tk.BooleanVar()
        self.lk.set(False)
        self.lkBut = tk.Checkbutton(frame, text = "LK", variable=self.lk, font=("Helvetica", 10), 
                                       command = lambda:self.checkLK())
        self.lkBut.place(x=self.xText2-10, y=self.yAddText2)
        
        self.initialLK = tk.BooleanVar()
        self.initialLK.set(False)
        self.initialLKBut = tk.Checkbutton(frame, text = "Initial Estimates LK", variable=self.initialLK, font=("Helvetica", 10),
                                     command = lambda:self.checkLK())
        self.initialLKBut.place(x=self.xText2 + 45, y=self.yAddText2)
        self.initialLKBut.config(state='disabled')           
   
        self.ncc = tk.BooleanVar()
        self.ncc.set(True)
        self.nccBut = tk.Checkbutton(frame, text = "NCC", variable=self.ncc, font=("Helvetica", 10),
                                     command = lambda:self.checkNCC())
        self.nccBut.place(x=self.xText2 + 230, y=self.yAddText2)
         
                
        self.yAddText2 = self.yAddText2 + 25
        Label(frame, text="Template width: ").place(x=self.xText2, y=self.yAddText2)
        self.template_width = tk.IntVar()
        self.template_width_Param = Entry(frame, textvariable=self.template_width, font=("Helvetica", 10, 'italic'))
        self.template_width_Param.place(x=self.xText2 + 240, y=self.yAddText2, width=75, height=20)
        self.template_width.set(7)                  
        
        self.yAddText2 = self.yAddText2 + 20
        Label(frame, text="Template height: ").place(x=self.xText2, y=self.yAddText2)
        self.template_height = tk.IntVar()
        self.template_height_Param = Entry(frame, textvariable=self.template_height, font=("Helvetica", 10, 'italic'))
        self.template_height_Param.place(x=self.xText2 + 240, y=self.yAddText2, width=75, height=20)
        self.template_height.set(7)                  
        
        self.yAddText2 = self.yAddText2 + 20
        Label(frame, text="Search area size x direction: ").place(x=self.xText2, y=self.yAddText2)
        self.search_area_x_CC = tk.IntVar()
        self.search_area_x_CC_Param = Entry(frame, textvariable=self.search_area_x_CC, font=("Helvetica", 10, 'italic'))
        self.search_area_x_CC_Param.place(x=self.xText2 + 240, y=self.yAddText2, width=75, height=20)
        self.search_area_x_CC.set(24)                
        
        self.yAddText2 = self.yAddText2 + 20
        Label(frame, text="Search area size y direction: ").place(x=self.xText2, y=self.yAddText2)
        self.search_area_y_CC = tk.IntVar()
        self.search_area_y_CC_Param = Entry(frame, textvariable=self.search_area_y_CC, font=("Helvetica", 10, 'italic'))
        self.search_area_y_CC_Param.place(x=self.xText2 + 240, y=self.yAddText2, width=75, height=20)
        self.search_area_y_CC.set(24)                 
        
        self.yAddText2 = self.yAddText2 + 20
        Label(frame, text="Shift search area in x: ").place(x=self.xText2, y=self.yAddText2)
        self.shiftSearchFromCenter_x = tk.IntVar()
        self.shiftSearchFromCenter_x_Param = Entry(frame, textvariable=self.shiftSearchFromCenter_x, font=("Helvetica", 10, 'italic'))
        self.shiftSearchFromCenter_x_Param.place(x=self.xText2 + 240, y=self.yAddText2, width=75, height=20)
        self.shiftSearchFromCenter_x.set(0)                 
        
        self.yAddText2 = self.yAddText2 + 20
        Label(frame, text="Shift search area in y: ").place(x=self.xText2, y=self.yAddText2)
        self.shiftSearchFromCenter_y = tk.IntVar()
        self.shiftSearchFromCenter_y_Param = Entry(frame, textvariable=self.shiftSearchFromCenter_y, font=("Helvetica", 10, 'italic'))
        self.shiftSearchFromCenter_y_Param.place(x=self.xText2 + 240, y=self.yAddText2, width=75, height=20)
        self.shiftSearchFromCenter_y.set(8)                    
        
        self.yAddText2 = self.yAddText2 + 20
        self.subpixel = tk.BooleanVar()
        self.subpixel.set(True)
        self.subpixelBut = tk.Checkbutton(frame, text = "Subpix", variable=self.subpixel, font=("Helvetica", 10))
        self.subpixelBut.place(x=self.xText2-10, y=self.yAddText2)

        self.performLSM = tk.BooleanVar()
        self.performLSM.set(False)
        self.performLSMBut = tk.Checkbutton(frame, text = "LSM", variable=self.performLSM, font=("Helvetica", 10))
        self.performLSMBut.place(x=self.xText2 + 67, y=self.yAddText2)

        self.savePlotData = tk.BooleanVar()
        self.savePlotData.set(True)
        self.savePlotDataBut = tk.Checkbutton(frame, text = "Plot results", variable=self.savePlotData, font=("Helvetica", 10))
        self.savePlotDataBut.place(x=self.xText2 + 126, y=self.yAddText2)
        
        self.yAddText2 = self.yAddText2# + 20
        self.saveGif = tk.BooleanVar()
        self.saveGif.set(True)
        self.saveGifBut = tk.Checkbutton(frame, text = "Save gif", variable=self.saveGif, font=("Helvetica", 10))
        self.saveGifBut.place(x=self.xText2 + 231, y=self.yAddText2)   
                
         
        #parameters iterations
        self.yAddText = self.yAddText2 + 40
        Label(frame, text="Iterations", font=("Courier", 10)).place(x=10, y=self.yAddText)
        self.yAddText = self.yAddText + 20
        Label(frame, text="FD every nth frame: ").place(x=10, y=self.yAddText)
        self.FD_everyIthFrame = tk.IntVar()
        self.FD_everyIthFrame_Param = Entry(frame, textvariable=self.FD_everyIthFrame, font=("Helvetica", 10, 'italic'))
        self.FD_everyIthFrame_Param.place(x=self.xText, y=self.yAddText, width=75, height=20)
        self.FD_everyIthFrame.set(10)                  
         
        self.yAddText = self.yAddText + 20 
        Label(frame, text="Track for n frames: ").place(x=10, y=self.yAddText)
        self.FT_forNthNberFrames = tk.IntVar()
        self.FT_forNthNberFrames_Param = Entry(frame, textvariable=self.FT_forNthNberFrames, font=("Helvetica", 10, 'italic'))
        self.FT_forNthNberFrames_Param.place(x=self.xText, y=self.yAddText, width=75, height=20)
        self.FT_forNthNberFrames.set(20)                 
        
        self.yAddText = self.yAddText + 20
        Label(frame, text="Track every nth frame: ").place(x=10, y=self.yAddText)
        self.TrackEveryNthFrame = tk.IntVar()
        self.TrackEveryNthFrame_Param = Entry(frame, textvariable=self.TrackEveryNthFrame, font=("Helvetica", 10, 'italic'))
        self.TrackEveryNthFrame_Param.place(x=self.xText, y=self.yAddText, width=75, height=20)
        self.TrackEveryNthFrame.set(2)


        #only filter tracks
        self.yAddText = self.yAddText + 30
        self.filterOnly = tk.BooleanVar()
        self.filterOnly.set(False)
        self.filterOnlyBut = tk.Checkbutton(frame, text = "Filter only tracks", variable=self.filterOnly, font=("Helvetica", 10),
                                            command = lambda:self.checkFilter())
        self.filterOnlyBut.place(x=self.xText - 250, y=self.yAddText)                                                           
        self.filterOnlyBut.config(font=("Helvetica", 10))
        
        #save parameter settings to file
        self.yAddText = self.yAddText + 30
        Label(frame, text="Save parameter settings: ", font=("Helvetica", 10)).place(x=10, y=self.yAddText)
        self.ParamSave_Button = Button(frame, text = '...', command = lambda:self.safeParemeterSetting())
        self.ParamSave_Button.place(x=self.xText - 50, y=self.yAddText, width=20, height=20)                                 

        #load parameter settings to file
        self.yAddText = self.yAddText + 20
        Label(frame, text="Load parameter settings: ", font=("Helvetica", 10)).place(x=10, y=self.yAddText)
        self.ParamLoad_Button = Button(frame, text = '...', command = lambda:self.loadParameterSettings())
        self.ParamLoad_Button.place(x=self.xText - 50, y=self.yAddText, width=20, height=20)
        
         
        #parameters filtering tracks    
        self.yAddText2 = self.yAddText2 + 40             
        Label(frame, text="Filtering tracks", font=("Courier", 10)).place(x=self.xText2, y=self.yAddText2)
        
        self.yAddText2 = self.yAddText2 + 20
        Label(frame, text="Minimum count features [%]: ").place(x=self.xText2, y=self.yAddText2)
        self.minimumTrackedFeatures = tk.DoubleVar()
        self.minimumTrackedFeatures_Param = Entry(frame, textvariable=self.minimumTrackedFeatures, font=("Helvetica", 10, 'italic'))
        self.minimumTrackedFeatures_Param.place(x=self.xText2 + 240, y=self.yAddText2, width=75, height=20)
        self.minimumTrackedFeatures.set(66)   
        
        self.yAddText2 = self.yAddText2 + 20
        Label(frame, text="Minimum track distance [px]: ").place(x=self.xText2, y=self.yAddText2)
        self.minDistance_px = tk.DoubleVar()
        self.minDistance_px_Param = Entry(frame, textvariable=self.minDistance_px, font=("Helvetica", 10, 'italic'))
        self.minDistance_px_Param.place(x=self.xText2 + 240, y=self.yAddText2, width=75, height=20)
        self.minDistance_px.set(2)                
        
        self.yAddText2 = self.yAddText2 + 20
        Label(frame, text="Maximum track distance [px]: ").place(x=self.xText2, y=self.yAddText2)
        self.maxDistance_px = tk.DoubleVar()
        self.maxDistance_px_Param = Entry(frame, textvariable=self.maxDistance_px, font=("Helvetica", 10, 'italic'))
        self.maxDistance_px_Param.place(x=self.xText2 + 240, y=self.yAddText2, width=75, height=20)
        self.maxDistance_px.set(50)            
        
        self.yAddText2 = self.yAddText2 + 20        
        Label(frame, text="Steadiness [deg]: ").place(x=self.xText2, y=self.yAddText2)
        self.threshAngleSteadiness = tk.DoubleVar()
        self.threshAngleSteadiness_Param = Entry(frame, textvariable=self.threshAngleSteadiness, font=("Helvetica", 10, 'italic'))
        self.threshAngleSteadiness_Param.place(x=self.xText2 + 240, y=self.yAddText2, width=75, height=20)
        self.threshAngleSteadiness.set(25)  
        
        self.yAddText2 = self.yAddText2 + 20
        Label(frame, text="Range track directions [deg]: ").place(x=self.xText2, y=self.yAddText2)
        self.threshAngleRange = tk.DoubleVar()
        self.threshAngleRange_Param = Entry(frame, textvariable=self.threshAngleRange, font=("Helvetica", 10, 'italic'))
        self.threshAngleRange_Param.place(x=self.xText2 + 240, y=self.yAddText2, width=75, height=20)
        self.threshAngleRange.set(90)                        
        
        self.yAddText2 = self.yAddText2 + 20
        Label(frame, text="Bin nbr main flow direction: ").place(x=self.xText2, y=self.yAddText2)
        self.binNbrMainflowdirection = tk.IntVar()
        self.binNbrMainflowdirection_Param = Entry(frame, textvariable=self.binNbrMainflowdirection, font=("Helvetica", 10, 'italic'))
        self.binNbrMainflowdirection_Param.place(x=self.xText2 + 240, y=self.yAddText2, width=75, height=20)
        self.binNbrMainflowdirection.set(0)                
        
        self.yAddText2 = self.yAddText2 + 20
        Label(frame, text="Buffer main flow [deg]: ").place(x=self.xText2, y=self.yAddText2)
        self.MainFlowAngleBuffer = tk.DoubleVar()
        self.MainFlowAngleBuffer_Param = Entry(frame, textvariable=self.MainFlowAngleBuffer, font=("Helvetica", 10, 'italic'))
        self.MainFlowAngleBuffer_Param.place(x=self.xText2 + 240, y=self.yAddText2, width=75, height=20)
        self.MainFlowAngleBuffer.set(10)                                    
        
        self.yAddText2 = self.yAddText2 + 20 
        Label(frame, text="setValue velocity threshold: ").place(x=self.xText2, y=self.yAddText2)
        self.veloStdThresh = tk.DoubleVar()
        self.veloStdThresh_Param = Entry(frame, textvariable=self.veloStdThresh, font=("Helvetica", 10, 'italic'))
        self.veloStdThresh_Param.place(x=self.xText2 + 240, y=self.yAddText2, width=75, height=20)
        self.veloStdThresh.set(1.5)      
         
  
        #referencing
        self.yAddText = self.yAddText2 + 40
        Label(frame, text="Scaling", font=("Courier", 10)).place(x=10, y=self.yAddText)
        self.yAddText = self.yAddText + 20
        Label(frame, text="Frame rate: ", font=("Helvetica", 10)).place(x=10, y=self.yAddText)
        self.frame_rate_cam = tk.IntVar()
        self.frame_rate_cam_Param = Entry(frame, textvariable=self.frame_rate_cam, font=("Helvetica", 10, 'italic'))
        self.frame_rate_cam_Param.place(x=110, y=self.yAddText, width=70, height=20)
        self.frame_rate_cam.set(30)

        #water level        
        self.yAddText2 = self.yAddText2 + 40
        Label(frame, text="Define feature search area and set water level", font=("Courier", 10)).place(x=200, y=self.yAddText2)
        self.yAddText2 = self.yAddText2 + 20
        Label(frame, text="Water level [m]: ", font=("Helvetica", 10)).place(x=200, y=self.yAddText2)
        self.waterlevel_pt = tk.DoubleVar()
        self.waterlevel_pt_Param = Entry(frame, textvariable=self.waterlevel_pt, font=("Helvetica", 10, 'italic'))    #, state=DISABLED
        self.waterlevel_pt_Param.place(x=320, y=self.yAddText2, width=75, height=20)
        self.waterlevel_pt.set(94.6)
        Label(frame, text="Buffer [m]: ", font=("Helvetica", 10)).place(x=410, y=self.yAddText2)
        self.waterlevel_buffer = tk.DoubleVar()
        self.waterlevel_buffer_Param = Entry(frame, textvariable=self.waterlevel_buffer, font=("Helvetica", 10, 'italic'))
        self.waterlevel_buffer_Param.place(x=490, y=self.yAddText2, width=75, height=20)
        self.waterlevel_buffer.set(0.3)    

        self.yAddText2 = self.yAddText2 + 25
        self.importAoIextent = tk.BooleanVar()
        self.importAoIextent.set(False)
        self.importAoIextentBut = tk.Checkbutton(frame, text = "Import search area file", variable=self.importAoIextent, 
                                                 font=("Helvetica", 10), command = lambda:self.checkSearchArea())
        self.importAoIextentBut.place(x=190, y=self.yAddText2)
        self.AoI_file = tk.StringVar()
        self.AoI_file_Param = Entry(frame, textvariable=self.AoI_file, font=("Helvetica", 10, 'italic'), state='disabled')
        self.AoI_file_Param.place(x=410, y=self.yAddText2, width=200, height=20)
        self.AoI_file_Button = Button(frame, text = '...', command = lambda:self.select_AoIFile())
        self.AoI_file_Button.place(x=385, y=self.yAddText2, width=20, height=20)   
                       
           
        #starting flow velocity estimation
        self.yAddText = self.yAddText + 45
        self.waterlineDetection = Button(frame, text="Estimate Flow Velocity", style="RB.TButton", command=self.EstimateVelocity)
        self.waterlineDetection.place(x=250, y=self.yAddText+30)
        

        '''----------------frame co-registration-------------------'''
        frame2 = Frame(note)  
        note.add(frame2, text="co-registration")
        
        self.xButton = 370
        self.xText = 250
        self.xText2 = 350
        self.yAddText = 10
        self.yAddText2 = 10

        #set parameters for co-registration
        Label(frame2, text="Perform co-registration of frames", font=("Courier", 10)).place(x=10, y=self.yAddText)
        self.yAddText = self.yAddText + 20
        Label(frame2, text="Maximum number of keypoints: ", font=("Helvetica", 10)).place(x=10, y=self.yAddText)
        self.coregist_kpnbr = tk.IntVar()
        self.coregist_kpnbr_Param = Entry(frame2, textvariable=self.coregist_kpnbr, font=("Helvetica", 10, 'italic'))
        self.coregist_kpnbr_Param.place(x=self.xText, y=self.yAddText, width=75, height=20)
        self.coregist_kpnbr.set(5000)
        
        self.yAddText = self.yAddText + 20
        Label(frame2, text="Number of good matches: ", font=("Helvetica", 10)).place(x=10, y=self.yAddText)
        self.nbr_good_matches = tk.IntVar()
        self.nbr_good_matches_Param = Entry(frame2, textvariable=self.nbr_good_matches, font=("Helvetica", 10, 'italic'))
        self.nbr_good_matches_Param.place(x=self.xText, y=self.yAddText, width=75, height=20)
        self.nbr_good_matches.set(10)
       
        self.yAddText = self.yAddText + 20
        self.sift = tk.BooleanVar()
        self.sift.set(True)     
        self.siftBut = tk.Checkbutton(frame2, text = "Matching with SIFT", variable=self.sift, font=("Helvetica", 10))
        self.siftBut.place(x=0, y=self.yAddText)
        
        self.feature_match_twosided = tk.BooleanVar()
        self.feature_match_twosided.set(True)     
        self.feature_match_twosidedBut = tk.Checkbutton(frame2, text = "Feature matching 2sided", font=("Helvetica", 10), 
                                                        variable=self.feature_match_twosided)
        self.feature_match_twosidedBut.place(x=165, y=self.yAddText)

        self.master_0 = tk.BooleanVar()
        self.master_0.set(True)     
        self.master_0But = tk.Checkbutton(frame2, text = "Register to first frame", variable=self.master_0, font=("Helvetica", 10))
        self.master_0But.place(x=370, y=self.yAddText)

        #starting co-registration
        self.yAddText = self.yAddText + 30
        self.coregister = Button(frame2, text="Co-register frames", style="RB.TButton", command=self.coregistration)
        self.coregister.place(x=10, y=self.yAddText)   
        
        
        self.yAddText = self.yAddText + 70             
        Label(frame2, text="Accuracy co-registration", font=("Courier", 10)).place(x=10, y=self.yAddText)        
        self.yAddText = self.yAddText + 20
        Label(frame2, text="Template size for co-registration accuracy: ", font=("Helvetica", 10)).place(x=10, y=self.yAddText)
        self.template_size_coregAcc = tk.IntVar()
        self.template_size_coregAcc_Param = Entry(frame2, textvariable=self.template_size_coregAcc, font=("Helvetica", 10, 'italic'))
        self.template_size_coregAcc_Param.place(x=self.xText+70, y=self.yAddText, width=75, height=20)
        self.template_size_coregAcc.set(30)        

        #starting accuracy assessment co-registration
        self.yAddText = self.yAddText + 30
        self.coregisteracc = Button(frame2, text="Accuracy co-registration", style="RB.TButton", command=self.accuracy_coregistration)
        self.coregisteracc.place(x=10, y=self.yAddText) 


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
            self.ptv.set(False)            
        else:
            self.maxFtNbr_FD_Param.config(state = 'normal')
            self.minimumThreshBrightness_Param.config(state = 'normal')
            self.neighborSearchRadius_FD_Param.config(stat = 'normal')
            self.maximumNeighbors_FD_Param.config(state = 'normal')
            self.sensitiveFD_Param.config(stat = 'normal')
            self.ptv.set(True)

    def checkPTV(self):
        if self.ptv.get() == False:
            self.maxFtNbr_FD_Param.config(state = 'disabled')
            self.minimumThreshBrightness_Param.config(state = 'disabled')
            self.neighborSearchRadius_FD_Param.config(stat = 'disabled')        
            self.maximumNeighbors_FD_Param.config(state = 'disabled')
            self.sensitiveFD_Param.config(stat = 'disabled')
            self.lspiv.set(True)        
        else:
            self.maxFtNbr_FD_Param.config(state = 'normal')
            self.minimumThreshBrightness_Param.config(state = 'normal')
            self.neighborSearchRadius_FD_Param.config(stat = 'normal')
            self.maximumNeighbors_FD_Param.config(state = 'normal')
            self.sensitiveFD_Param.config(stat = 'normal')
            self.lspiv.set(False)                     

    def checkLK(self):
        if self.lk.get() == False:
            self.ncc.set(True)
            self.subpixelBut.config(state='normal')
            self.performLSMBut.config(state='normal')
            self.initialLKBut.config(state='disabled')
#            self.search_area_y_CC_Param.config(state='normal')
#            self.search_area_x_CC_Param.config(state='normal')
            self.shiftSearchFromCenter_x_Param.config(state = 'normal')  
            self.shiftSearchFromCenter_y_Param.config(state = 'normal')             
        else:
            self.ncc.set(False)
            self.subpixelBut.config(state='disabled')
            self.performLSMBut.config(state='disabled')
            self.initialLKBut.config(state='normal')
#            self.search_area_y_CC_Param.config(state='disabled')
#            self.search_area_x_CC_Param.config(state='disabled') 
            self.shiftSearchFromCenter_x_Param.config(state = 'disabled')  
            self.shiftSearchFromCenter_y_Param.config(state = 'disabled') 
            
        if self.initialLK.get() == True:
            self.shiftSearchFromCenter_x_Param.config(state = 'normal')  
            self.shiftSearchFromCenter_y_Param.config(state = 'normal')
        else:
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
        else:
            self.maxFtNbr_FD_Param.config(state='normal')
            self.minimumThreshBrightness_Param.config(state='normal')
            self.neighborSearchRadius_FD_Param.config(state='normal')
            self.sensitiveFD_Param.config(state='normal')
            self.maximumNeighbors_FD_Param.config(state='normal')
            self.template_width_Param.config(state='normal')
            self.template_height_Param.config(state='normal')
            self.search_area_x_CC_Param.config(state='normal')
            self.search_area_y_CC_Param.config(state='normal')
            self.shiftSearchFromCenter_x_Param.config(state='normal')
            self.shiftSearchFromCenter_y_Param.config(state='normal')
            self.subpixelBut.config(state='normal')
            self.performLSMBut.config(state='normal')
            self.savePlotDataBut.config(state='normal')
            self.saveGifBut.config(state='normal')
            self.FD_everyIthFrame_Param.config(state='normal')
            self.FT_forNthNberFrames_Param.config(state='normal')
            self.TrackEveryNthFrame_Param.config(state='normal')
            
    def checkSearchArea(self):
        if self.importAoIextent.get() == True:
            self.AoI_file_Param.config(state = 'normal')
            self.ptCloud_file_Param.config(state = 'disabled')
            self.waterlevel_buffer_Param.config(state = 'disabled')   
        else:
            self.AoI_file_Param.config(state = 'disabled')
            self.ptCloud_file_Param.config(state = 'normal')
            self.waterlevel_buffer_Param.config(state = 'normal')       
        

    '''functions to load input data'''
    def select_dirOutput(self):
        outputDir = tkFileDialog.askdirectory(title = 'Select output directory')
        if not outputDir:
            self.directoryOutput.set("")
        else:
            self.directoryOutput.set(outputDir  + '/')
            
    def select_dirImgs(self):
        imgsDir = tkFileDialog.askdirectory(title = 'Select directory of frames')
        if not imgsDir:
            self.dir_imgs.set("")
        else:
            self.dir_imgs.set(imgsDir  + '/')
                                         
    def select_imgName(self):
       imgName = tkFileDialog.askopenfilename(title='Image to draw velocity tracks for visualisation', 
                                              initialdir=os.getcwd())
       if not imgName:
           self.img_name.set("")
       else:
           self.img_name.set(imgName)
           
    def select_GCPcooFile(self):
       gcpCooFile = tkFileDialog.askopenfilename(title='Set file with GCP coordinates (object space)', 
                                                 filetypes=[('Text file (*.txt)', '*.txt')],initialdir=os.getcwd())
       if not gcpCooFile:
           self.gcpCoo_file.set("")
       else:
           self.gcpCoo_file.set(gcpCooFile)    
           
    def select_GCPimgCooFile(self):
       gcpImgCooFile = tkFileDialog.askopenfilename(title='Set file with GCP coordinates (image space)', 
                                                    filetypes=[('Text file (*.txt)', '*.txt')],initialdir=os.getcwd())
       if not gcpImgCooFile:
           self.imgCoo_GCP_file.set("")
       else:
           self.imgCoo_GCP_file.set(gcpImgCooFile)                    

    def select_iorFile(self):
       iorFile = tkFileDialog.askopenfilename(title='Set file with interior camera orientation parameters', 
                                              filetypes=[('Text file (*.txt)', '*.txt')],initialdir=os.getcwd())
       if not iorFile:
           self.ior_file.set("")
       else:
           self.ior_file.set(iorFile)
           
    def select_ptClFile(self):
       ptClFile = tkFileDialog.askopenfilename(title='Set file with point cloud of river topography/bathymetry', 
                                               filetypes=[('Text file (*.txt)', '*.txt')],initialdir=os.getcwd())
       if not ptClFile:
           self.ptCloud_file.set("")
       else:
           self.ptCloud_file.set(ptClFile)    
           
    def select_AoIFile(self):
       aoiFile = tkFileDialog.askopenfilename(title='Set file with AoI extent coordinates (xy, image space)', 
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
                       'PTV ' + str(self.ptv.get())]
        
        listParams = pd.DataFrame(listParams)
        listParams.to_csv(self.directoryOutput.get() + 'parameterSettings.txt', index=False, header=None)
        print('parameters saved')   
        self.printTxt('parameters saved')
        
    def loadParameterSettings(self):
        paramFile = tkFileDialog.askopenfilename(title='Load parameter file', 
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

        print('parameters loaded')
        self.printTxt('parameters loaded')


    def printTxt(self, txt):
        self.textbox.insert(END, txt)
        return
    

    '''functions for data processing'''
    def accuracy_coregistration(self):

        #read parameters from directories
        failing = True
        while failing:
            try:
                directoryOutput_coreg_acc = tkFileDialog.askdirectory(title='Output directory accuracy results') + '/'
                
                image_list_coreg = tkFileDialog.askopenfilenames(title='Open co-registered frames')
                
                check_points_forAccCoreg =  tkFileDialog.askopenfilename(title='File with CP coordinates (image space)', 
                                                                         filetypes=[('Text file (*.txt)', '*.txt')],initialdir=os.getcwd())
                         
                failing = False
            
            except Exception as e:
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print('failed reading data, please try again\n')
            
        image_list_coreg = sorted(image_list_coreg, key=lambda image: (image[0], image[1]))

        img_check_pts = np.asarray(pd.read_table(check_points_forAccCoreg), dtype=np.float32)        
        
        coregF.accuracy_coregistration(image_list_coreg, img_check_pts, self.template_size_coregAcc.get(), directoryOutput_coreg_acc)        
        
        print('Accuracy assessment co-registration finished.')               
        
    
    def coregistration(self):        
        #read parameters from directories
        print(cv2.__version__)
        failing = True
        while failing:
            try:
                directoryOutput_coreg = tkFileDialog.askdirectory(title='Output directory co-registration') + '/'
                
                image_list = tkFileDialog.askopenfilenames(title='Open frames for co-registration')
                                         
                failing = False
            
            except Exception as e:
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print('failed reading data, please try again\n')
        

        image_list = sorted(image_list, key=lambda image: (image[0], image[1]))
        
        coregF.coregistration(image_list, directoryOutput_coreg, self.coregist_kpnbr.get(), self.sift.get(), 
                              self.feature_match_twosided.get(), self.nbr_good_matches.get())      
                
        print('Co-registration finished.')
        self.printTxt('finished co-registration')
        

    def EstimateVelocity(self):
        '''-------set parameters-------'''
        test_run = self.test_run.get()
        
        #params exterior orientation estimation
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

        #params search area definition
        waterlevel_pt = self.waterlevel_pt.get()  #float
        waterlevel_buffer = self.waterlevel_buffer.get()

        #params feature detection
        minimumThreshBrightness = self.minimumThreshBrightness.get()
        neighborSearchRadius_FD = self.neighborSearchRadius_FD.get()
        maximumNeighbors_FD = self.maximumNeighbors_FD.get()
        maxFtNbr_FD = self.maxFtNbr_FD.get()
        sensitiveFD = self.sensitiveFD.get()

        #params tracking
        threshLSM = 0.001  #for adjustment
        lsmBuffer = 3 #increases lsm search compared to patch
        template_width = self.template_width.get() #has to be even
        template_height = self.template_height.get()
        search_area_x_CC = self.search_area_x_CC.get()
        search_area_y_CC = self.search_area_y_CC.get()
        shiftSearchFromCenter_x = self.shiftSearchFromCenter_x.get()
        shiftSearchFromCenter_y = self.shiftSearchFromCenter_y.get()
        subpixel = self.subpixel.get()

        performLSM = self.performLSM.get()
        savePlotData = self.savePlotData.get()
        save_gif = self.saveGif.get()

        #params iterations
        FD_everyIthFrame = self.FD_everyIthFrame.get()
        FT_forNthNberFrames = self.FT_forNthNberFrames.get()
        TrackEveryNthFrame = self.TrackEveryNthFrame.get()
        minimumTrackedFeatures = self.minimumTrackedFeatures.get()
        minimumTrackedFeatures = np.int(FT_forNthNberFrames*(minimumTrackedFeatures/100)/TrackEveryNthFrame)

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

        #read parameters from directories
        directoryOutput = self.directoryOutput.get()
        dir_imgs = self.dir_imgs.get()
        img_name = self.img_name.get()
        gcpCoo_file = self.gcpCoo_file.get()
        imgCoo_GCP_file = self.imgCoo_GCP_file.get()
        ior_file = self.ior_file.get()
        if not self.importAoIextent.get():
            ptCloud_file = self.ptCloud_file.get()
        AoI_file = self.AoI_file.get()
                
        '''-------read data and prepare some for following processing-------'''                    
        interior_orient = photogrF.read_aicon_ior(ior_file) #read interior orientation from file (aicon)   
        if not self.importAoIextent.get():
            ptCloud = np.asarray(pd.read_table(ptCloud_file, header=None, delimiter=',')) #read point cloud
        else: 
            ptCloud = []          
        img_list = ioF.read_imgs_folder(dir_imgs) #read image names in folder
    
        #prepare output
        if not os.path.exists(directoryOutput):
            os.system('mkdir ' + directoryOutput)
            
        print('all input data read\n')
        print('------------------------------------------')
        self.printTxt('finished reading input data')
                
                
        '''-------get exterior camera geometry-------'''
        eor_mat = ptv.EstimateExterior(gcpCoo_file, imgCoo_GCP_file, interior_orient, estimate_exterior,
                                       unit_gcp, max_orientation_deviation, ransacApprox, angles_eor, pos_eor,
                                       directoryOutput)

        #select points only below water level to extract river area to search for features...
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
                
            while frameCount < lenLoop:
                
                if frameCount % FD_everyIthFrame == 0:
                    
                    if first_loop:
                        feature_ID_max = None
                    
                    if self.lspiv.get():
                        featuresToTrack, first_loop, feature_ID_max = ptv.FeatureDetectionLSPIV(dir_imgs, img_list, frameCount, template_width, template_height, searchMask, 
                                                                                                FD_everyIthFrame, savePlotData, directoryOutput, first_loop, feature_ID_max)
                    else:
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
            self.printTxt('finished feature tracking')
        
        else:
            trackFile = tkFileDialog.askopenfilename(title='File with tracks', filetypes=[('Text file (*.txt)', '*.txt')],initialdir=os.getcwd())
            trackedFeaturesOutput_undist = pd.read_table(trackFile)


        '''-------filter tracking results in image space-------'''
        filteredFeatures, [nbr_features_raw, nbr_features_mindist, 
                           nbr_features_maxdist,minimumTrackedFeatures,steady_angle, 
                           nbr_features_steady,range_angle, nbr_features_rangeangle, 
                           flowdir_angle,nbr_features_mainflowdir] = ptv.FilterTracks(trackedFeaturesOutput_undist, dir_imgs, img_list, directoryOutput,
                                                                                      minDistance_px, maxDistance_px, minimumTrackedFeatures, 
                                                                                      threshAngleSteadiness, threshAngleRange,
                                                                                      binNbrMainflowdirection, MainFlowAngleBuffer)

        print('filtering tracks done\n')
        print('------------------------------------------')
        self.printTxt('finished filtering tracks')


        '''-------transform img measurements into object space-------'''
        ptv.TracksPx_to_TracksMetric(filteredFeatures, minimumTrackedFeatures, interior_orient, eor_mat, unit_gcp,
                                     frame_rate_cam, TrackEveryNthFrame, waterlevel_pt, directoryOutput, dir_imgs, img_list)
        self.printTxt('finished transforming pixel values to velocities')
        
        
        '''-------logfile-------'''
        log_file_wirter, logfile = ioF.logfile_writer(directoryOutput + 'logfile.txt')
        log_file_wirter.writerow([['test run: ', test_run],['exterior angles: ', angles_eor],['exterior position: ', pos_eor],
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


        print('finished\n')
        print('------------------------------------------')
        self.printTxt('finished velocity estimations')
        
 
def main():        

    root = tk.Tk()
    
    #root.geometry("500x500")
    
    app = FlowVeloTool(root)   
    
    root.mainloop()
    
    root.destroy() # optional; see description below        


if __name__ == "__main__":
    main()  