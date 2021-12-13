FlowVelo: An image-based tool to automatically measure flow velocities

This program tracks flow velocities using image or frame sequences. It includes measuring the camera position and orientation (camera pose), automatic extraction of the water area for feature searching, particle detection and filtering, particle tracking and filtering and scaling the tracks to get flow velocities. Furthermore, a function is included to automatically co-register frames in case of camera movements.

Source code:
https://github.com/AnetteEltner/FlowVeloTool/python3_version/latest_version


Requirements:
Python 3.6
Additional libraries:
-scipy 1.4.1
-scikit-learn 0.22.2
-scikit-image 0.16.2
-shapely 1.7.0
-imageio 2.8.0
-opencv-python 4.1.1.26
-seaborn 0.10.0
-matplotlib 3.2.0
-pyttk 0.3.2
-pandas 1.0.1
-numpy 1.18.1


Running FlowVelo on Windows
- Install PyCharm
- In PyCharm:
  	o Go to File -> Create New Project…
      		Choose location to folder you like
      		Set Base interpreter to Python 3.6
  	o Go to File -> Settings 
      		Go to Project:“your chosen folder”
      			• Go to Project Interpreter
          			o Go to +
          			o install libraries mentioned in requirements
- Download source code of FlowVeloTool from GitHub 	  
- Unpack FlowVeloTool.zip and put all scripts including the tutorial folder into chosen location folder
- Open GUI_FeatureDetectionTracking.py

- There might be an issue with the screen resolution, hindering the visibility of the entire window. If this is the case, please lower the screen resolution.
	
Input data:
- Interior geometry of the camera. Minimum information needed is focal length in mm, sensor size in mm and sensor resolution in pixels. There is an example file interiorGeometry.txt in the tutorial data and corresponding explanation of parameters in interiorGeometry_explained.txt.
- Optional data to scale the velocities:
	o Object space coordinates (ID,X,Y,Z) of GCPs (Ground Control Points) used to estimate camera pose. There is an example file markersGCPobj.txt in the tutorial data. If no GCPs are given it is possible to set the exterior camera geometry (i.e. camera position and orientation) directly, if it is known.  Image coordinates (id,x,y) of GCPs used to estimate camera pose in pixels (origin of image coordinate system is the top left corner of the image). There is an example file markersGCPimg.txt in the tutorial data. Locations of GCPs in the image are illustrated in locationGCPs_frame3.jpg. Image coordinates might also be measured interactively with the FlowVeloTool (although it might be buggy) instead of providing the file. 
	o In case the exterior camera geometry is not used, the perpendicular distance of the camera to the water surface can be utilised to estimate the image scale and thus scale the velocities. This should be provided in [m]. For instance, in case of UAV imagery the flying height above the water surface can be used scale the velocity values assuming a Nadir view of the camera. In case the image scale is used, information about the water level is not needed.
- Optional data to mask the water area:
	o 3D point cloud (X,Y,Z) of area of interest to define search area. There is an example file 3DmodelPointCloud.txt in the tutorial data. If no point cloud is given, it is also possible to import a binary image file where the water area is masked (i.e. white water area in front of black background).
	o Instead of the 3D point cloud, also a binary mask file can be used to mask the water area. This might be useful in the case of using only the image scale for velocity estimation or if no 3D point cloud is given.
- Water level (not in case of using the image scale)
- Folder with the image/frame sequence


Tutorial data:
FlowVeloTool.zip available at 
https://cloudstore.zih.tu-dresden.de/index.php/s/2XWTnzqCkoOJvkF


For questions please contact: Anette.Eltner@tu-dresden.de


How to cite:
Eltner, A., Sardemann, H., Grundmann, J.: Technical Note: Flow velocity and discharge measurement in rivers using terrestrial and unmanned-aerial-vehicle imagery. Hydrol. Earth Syst. Sci., 24, 1429–1445, 2020
