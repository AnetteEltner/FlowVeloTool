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


import sys, math
import numpy as np

import cv2

import photogrammetry_functions as photo_tool


def LineWaterSurfaceIntersect(imgPts, cameraGeometry_interior, cameraGeometry_exterior, pointCloud, epsilon=1e-6):    
    #get water plane with plane fitting (when water surface not horizontal)
    planeParam = ausgl_ebene(pointCloud)   #planeParam = [a,b,c,d]
    try:
        np.sum(np.asarray(planeParam))
        return np.asarray(planeParam)
    except Exception as e:
        _, _, exc_tb = sys.exc_info()
        print(e, 'line ' + str(exc_tb.tb_lineno))
        print('plane fitting failed')
        return    

    #calculate plane normal
    planeNormal = np.array([planeParam[0],planeParam[1],planeParam[2]]) #normal vector
    planePoint = np.array([0,0,-1*planeParam[3]/planeParam[2]]) #support vector (for plane in normal form)
    
    #calculate angle of plane
    PlanarPlaneNorm = np.asarray([0,0,1])
    len_NivelPlaneNorm = np.sum(np.sqrt((PlanarPlaneNorm**2)))
    len_planeNormal = np.sum(np.sqrt((planeNormal**2)))
    zaehler = planeNormal[0] * PlanarPlaneNorm[0] + planeNormal[1] * PlanarPlaneNorm[1] + planeNormal[2] * PlanarPlaneNorm[2]
    angleNormVec = np.arccos(zaehler / (len_NivelPlaneNorm * len_planeNormal)) * 180/np.pi
    print('angle of plane: ' + str(angleNormVec))
    
    #origin of ray is projection center
    rayPoint = np.asarray([cameraGeometry_exterior[0], cameraGeometry_exterior[1], cameraGeometry_exterior[2]])
    
    #transform image ray into object space
    imgPts_undist_mm = photo_tool.undistort_img_coos(imgPts, cameraGeometry_interior)
    rayDirections = photo_tool.imgDepthPts_to_objSpace(imgPts_undist_mm, cameraGeometry_exterior, cameraGeometry_interior.resolution_x, cameraGeometry_interior.resolution_y, 
                                                     cameraGeometry_interior.sensor_size_x / cameraGeometry_interior.resolution_x, cameraGeometry_interior.ck)   
        
    PtsIntersectedWaterPlane = []
    for ray in rayDirections:        
        #perform intersection 
        ndotu = planeNormal.dot(ray)
        if abs(ndotu) < epsilon:
            raise RuntimeError("no intersection with plane possible")
     
        w = rayPoint - planePoint
        si = -planeNormal.dot(w) / ndotu
        Psi = w + si * ray + planePoint
        
        PtsIntersectedWaterPlane.append(Psi)
        
    PtsIntersectedWaterPlane = np.asarray(PtsIntersectedWaterPlane)
    
    return PtsIntersectedWaterPlane


def LinePlaneIntersect(imgPts, waterlevel, cameraGeometry_interior, cameraGeometry_exterior, unit_gcp=1, epsilon=1e-6):
    #assume water is horizontal plane
    planeNormal = np.array([0,0,1]) #normal vector
    planePoint = np.array([0,0,waterlevel*unit_gcp]) #support vector (for plane in normal form)   
    planeNormal_norm = planeNormal * (1/np.linalg.norm(planeNormal))
    
    #origin of ray is projection center
    rayPoint = np.asarray([cameraGeometry_exterior[0,3], cameraGeometry_exterior[1,3], cameraGeometry_exterior[2,3]])
    
    #transform image ray into object space
    imgPts_undist_mm = photo_tool.undistort_img_coos(imgPts, cameraGeometry_interior)
    imgPts_undist_forObj_x = imgPts_undist_mm[:,0] * -1
    imgPts_undist_forObj_y = imgPts_undist_mm[:,1]
    imgPts_undist_forObj = np.hstack((imgPts_undist_forObj_x.reshape(imgPts_undist_forObj_x.shape[0],1), imgPts_undist_forObj_y.reshape(imgPts_undist_forObj_y.shape[0],1)))
    imgPts_undist_forObj = np.hstack((imgPts_undist_forObj, np.ones((imgPts_undist_mm.shape[0],1)) * cameraGeometry_interior.ck))
    
    #transform into object space
    imgPts_XYZ = np.matrix(cameraGeometry_exterior) * np.matrix(np.vstack((imgPts_undist_forObj.T, np.ones(imgPts_undist_forObj.shape[0]))))
    rayPts = np.asarray(imgPts_XYZ.T)[:,0:3] 
    
    #plot ray of camera viewing direction
#     z_range = np.asarray(range(500))
#     Z_range = z_range.reshape(z_range.shape[0],1) * (np.ones((z_range.shape[0],3)) * cameraGeometry_exterior[0:3,2].T) + np.ones((z_range.shape[0],3)) * cameraGeometry_exterior[0:3,3].T
  
    rayDirections = np.ones((rayPts.shape)) * rayPoint - rayPts    
    rayDirections_norm = rayDirections * (1/np.linalg.norm(rayDirections))

    PtsIntersectedWaterPlane = []
    for ray in rayDirections_norm:        
        #perform intersection 
        ndotu = planeNormal_norm.dot(ray)
        if abs(ndotu) < epsilon:
            raise RuntimeError("no intersection with plane possible")
     
        w = rayPoint - planePoint
        si = -planeNormal_norm.dot(w) / ndotu
        Psi = w + si * ray + planePoint
        
        PtsIntersectedWaterPlane.append(Psi)
        
    PtsIntersectedWaterPlane = np.asarray(PtsIntersectedWaterPlane)
    
    return PtsIntersectedWaterPlane


def getTransformationMat(XY, xy):
#XYxy: np.array with assigned object and img space lateral coordinates    
    #get transformation matrix
    transform_mat, _ = cv2.findHomography(xy, XY, cv2.RANSAC)  
    transform_mat = np.asarray(transform_mat, dtype=np.float32) 
    print('Transformation matrix:')
    print(transform_mat)
    print('')
    
    #control transformation result by re-projecting transformed points into object space
    xy_calc = xy.reshape(xy.shape[0],2)
    xy_calc = np.array([xy_calc])
    xy_transformed = cv2.perspectiveTransform(xy_calc, transform_mat)
    print('XY object:')
    print(XY)
    print('')
    print('xy transformed to object space:')
    print(xy_transformed)
    print('')
    
    return transform_mat


def TracksToVelocityWithTransformMat(start_point, end_point, transform_mat, frame_rate):
    #transform tracks to scale correctly (when rectification performed with GCPs in plane)
    xy_start_transformed = cv2.perspectiveTransform(start_point, transform_mat)
    x_start_t = xy_start_transformed.flatten()[0]
    y_start_t = xy_start_transformed.flatten()[1]   
                         
    xy_transformed = cv2.perspectiveTransform(end_point, transform_mat)
    x_t = xy_transformed.flatten()[0]
    y_t = xy_transformed.flatten()[1]
    dist = np.sqrt(np.square(x_start_t-x_t) + (np.square(y_start_t-y_t)))
    velo = dist/(1/np.float(frame_rate))
    
    return([start_point[0], start_point[1], x_start_t, y_start_t,
            end_point[0], end_point[1], x_t, y_t, dist, velo])


def TracksToVelocity_PerPoint(start_point, end_point, frame_rate):        
    x_start_t = start_point[0]
    y_start_t = start_point[1]   
    x_t = end_point[0]
    y_t = end_point[1]

    dist = np.sqrt(np.square(x_start_t-x_t) + (np.square(y_start_t-y_t)))
    velo = dist/(1/np.float(frame_rate))
    
    return([start_point[0], start_point[1], x_start_t, y_start_t,
            end_point[0], end_point[1], x_t, y_t, dist, velo])
    

''' plan adjustment '''
from scipy import linalg, sparse
def ausgl_ebene(Punkte, ausgabe='no'):    
## Calculation adjusted plane parameters with Gaus-Helmert-Modell
# v1.03
#
# Mordwinzew Waldemar (2011)
# http://www.mordwinzew.de/ausgleichung/ghm/ebene
# Translated from Matlab to Python by Anette Eltner
# 
# Liest eine Koordinatendatei im folgenden Format ein
# 0.0   0.0   0.0
# ...   ...   ...
#  xi    yi    zi
         
    ## Hilfsvariablen
    # Anzahl der Beobachtungen
    nl = int(Punkte.shape[0])
     
    # Dimension
    nd = int(Punkte.shape[1])
    
    ## Schaetzung der Naeherungswerte
    if Punkte.shape[0] <= 3:
        return ['skipped1', -1, -1, -1]
    
    # nx, ny, nz, d
    X_dach = (np.cross(Punkte[1,:] - Punkte[0,:], Punkte[2,:] - Punkte[0,:])).conj().T  #conjugate transpose   
    # Vermeidung unguenstiger Naeherungswerte, welche Division mit 0 ergeben
    if math.sqrt(np.sqrt(np.dot(X_dach[0:3],X_dach[0:3]))**2) == 0:
        X_dach = (np.cross(Punkte[1,:] - Punkte[0,:], Punkte[-1,:] - Punkte[0,:])).conj().T
        if math.sqrt(np.sqrt(np.dot(X_dach[0:3],X_dach[0:3]))**2) == 0:
            return ['skipped2', -1, -1, -1]
    
    X_dach = X_dach / math.sqrt(np.sqrt(np.dot(X_dach[0:3],X_dach[0:3]))**2)
    X_dach_4 = math.sqrt(np.average(Punkte[:,0])**2+np.average(Punkte[:,1])**2+np.average(Punkte[:,2])**2)
    X_dach = np.hstack((X_dach, X_dach_4))

    ## Schaetzung der exakten Minimierer X0,V0
    # Vektor der Verbesserungen
    v = np.zeros((nd * nl,1))
     
    # Anzahl der Unbekannten
    nu = int(X_dach.shape[0])
     
    # Bedingungsvektor
    C = np.zeros((nu,1)) 
     
    # Anzahl der Bedingungen
    nb = int(C.shape[1])
     
    # Legt den gewuenschten Konvergenzbetrag fest ab dem die
    # Iterationen abgebrochen werden koennen.
    EPSILON = 1e-12
     
    # Standardabweichung der Gewichtseinheit a priori
    sigma_0_a_pri = 1.0
     
    # Kofaktormatrix (hier gleichgenaue unkorrelierte Beobachtungen)
    Cll = sparse.eye(3*nl,3*nl)
     
    # Kovarianzmatrix
    Qll = (1/(sigma_0_a_pri**2)) * Cll
     
    # Maximale Iterationsanzahl
    maxit = 20
    
    # Beginn des Iterationsvorgangs
    iteration = 0 
    while iteration <= maxit: 
        # Modellmatrix A
        r = np.ones((nl,1))
        A = np.c_[Punkte + v.reshape(nl,nd,order='F').copy() , np.ones((nl,1))]
        
        if(np.max(np.absolute(np.dot(A, X_dach))) < EPSILON ):
            break
     
        # Hilfvariable
        h = linalg.norm(X_dach[0:3])
     
        # Bildung des normierten Bedingungsvektors
        C_03 = (X_dach[0:3]/h)  #komponentenweise, da X_dach np.array und nicht np.matrix
        C_add0 = np.zeros((int(C.shape[0]) - int(C_03.shape[0]),1)).flatten()
        C = np.hstack((C_03, C_add0))[:, np.newaxis]
        del C_03, C_add0
        
        # Bildung der B-Modellmatrix
        # Da diese 3(n#-n) Nullelemente hat, eignet sich der Sparse-Datentyp fuer duennbesetzte Matrizen.
        B = sparse.spdiags(np.tile([X_dach[2],X_dach[1],X_dach[0]],(nl,1)).conj().T, np.array([-2*nl,-nl,0]), 3*nl,nl).conj().T #Unterschied zu Matlab --> zeilenweise (anstatt spaltenweise)
        
        # Bildung der Inversen zur B*B^T Matrix
        BQBT = (B * Qll) * B.conj().T
        #BQBT = np.dot(np.dot(B,Qll),B.conj().T)    #dot(dot(A,B),C)
    
        # Bildung der Normalgleichungsmatrix
        N = np.r_[np.c_[np.dot(-1*A.conj().T,linalg.solve(BQBT.todense(),A)), C], np.c_[C.conj().T, np.zeros((nb,nb))]]
        
        r = 1/np.linalg.cond(N)*1e+3  #close to matrix reciprocal condition number estimate
        d = np.absolute(np.linalg.det(N))
     
        if ( ( d < EPSILON ) != ( r < EPSILON ) ):
            print( 'Fehler: Normalgleichungsmatrix singulaer oder schlecht konditioniert')
            print( '{0} and {1}'.format('Determinante: ', d) )
            print( '{0} and {1}'.format('Konditionierung:', r))
            sys.exit()
        
        # Bilde den Widerspruchsvektor
        w1 = np.dot(np.c_[Punkte, np.ones((nl,1))], X_dach)
     
        # Bilde den Widerspruchsskalar
        w2 = h - 1
     
        # Berechne die Differenzloesung
        x = linalg.solve(N,np.r_[np.dot(A.conj().T, (linalg.solve(BQBT.todense(), w1))), -1*w2])
     
        # Berechne die Verbesserungen dieser Iteration        
        v = ((Qll * B.conj().T) * (sparse.csr_matrix(linalg.solve(BQBT.todense(), (np.dot(-1*A, x[0:4]) - w1))).conj().T)).todense()
        #v = np.dot(np.dot(Qll, B.conj().T), sparse.csr_matrix(linalg.solve(BQBT.todense(), (np.dot(-1*A, x[0:4]) - w1))).conj().T).todense()
        
        #v = np.dot(np.dot(Qll, B.conj().T).todense(), (linalg.solve(BQBT.todense(), (np.dot(-1*A, x[0:4]) - w1))))
     
        # Addiere die Differenzloesung
        X_dach = X_dach + x[0:nu]
     
        # Zweite Abbruchbedingung
        if(np.max(np.absolute(x[0:nu])) < EPSILON ):
            break
       
        iteration = iteration + 1 
    
     
    ## Ausgabe
    if (iteration < maxit):
        if ausgabe == 'yes':
            print( 'Ergebnis ausgleichende Ebene')
            print( 'Konvergenz: Erfolgt')
            print( '{0} and {1}'.format('Konvergenzgrenze: ', EPSILON))
            print( '{0} and {1}'.format('Anzahl Iterationen: ', iteration))
     
            print( '-- Statistik --')
        
        if int(Punkte.shape[0]) <= 1000:
            redundanz = nl - nu + nb
            s0_a_post = np.sqrt(sparse.csr_matrix(v).conj().T * sparse.csr_matrix(linalg.solve(Qll.todense(),v)/redundanz).todense())
            if np.linalg.cond(-1*N[0:nu,0:nu]) < 1/sys.float_info.epsilon:
                Qxx = np.linalg.inv(-1*N[0:nu,0:nu])
            else:
                #print('matrix inversion failed')
                return('skipped', -1, -1, -1)
                
            #Qxx = np.linalg.inv(-1*N[0:nu,0:nu])
     
        del v, A, B, Qll
     
        if ausgabe == 'yes':
            print( '{0} and {1}'.format('Anzahl Beobachtungen: ', nl))
            print( '{0} and {1}'.format('Anzahl Parameter: ', int(X_dach.shape[0])))
            print( '{0} and {1}'.format('Anzahl Bedingungent: ', nb))
        
        if int(Punkte.shape[0]) <= 1000:
            if ausgabe == 'yes':
                print( '{0} and {1}'.format('Gesamtredundanz: ', redundanz))
                print( '{0} and {1}'.format('ns0_a_prio: ', sigma_0_a_pri))
                print( '{0} and {1}'.format('s0_a_post: ', s0_a_post) )
                print( '{0} and {1}'.format('sNx: ', s0_a_post*np.sqrt(Qxx[0,0])))
                print( '{0} and {1}'.format('sNy: ', s0_a_post*np.sqrt(Qxx[1,1])))
                print( '{0} and {1}'.format('sNz: ', s0_a_post*np.sqrt(Qxx[2,2])))
                print( '{0} and {1}'.format('sd: ' ,s0_a_post*np.sqrt(Qxx[3,3])))
    
        if ausgabe == 'yes':
            print( '-- Parameter --')
            print( '{0} and {1}'.format('Nx0: ', X_dach[0]))
            print( '{0} and {1}'.format('Ny0: ', X_dach[1]))
            print( '{0} and {1}'.format('Nz0: ', X_dach[2]))
            print( '{0} and {1}'.format('d0: ', X_dach[3]))
        
        Param_Ebene = [X_dach[0], X_dach[1], X_dach[2], X_dach[3]]
        
        return Param_Ebene
     
    else:
        print( 'nKonvergenz: Nicht erfolgt') 