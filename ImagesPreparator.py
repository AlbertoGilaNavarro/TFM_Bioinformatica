#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 23:48:47 2020

@author: albert
"""

import cv2 as cv
import os
import re
import numpy as np
import tifffile as tiff

#Objetivo:
 #Formar un solo hipercubo por flor, componiendo las imágenes a y b y todas
 #las reflectancias VIS y IR

#%% Hipercubos visibles
#Lee directorios
visImagesDir="/home/albert/AlbertData/Bioinfo/TFM/NuevoBorradorPrograma/Data/flores/mandar/visible/hipercubo/"
visImagesNames=sorted(os.listdir(visImagesDir))

#Inicializa las estructuras de datos con la primera iteración
flowerID=visImagesNames[0].split("_")[6]
image3chanel=cv.imread(visImagesDir+"/"+visImagesNames[0])
image3chanel=cv.resize(image3chanel, (64,32))#Para componer luego
image1chanel=cv.cvtColor(image3chanel, cv.COLOR_BGR2GRAY)
hyperCube=[image1chanel]
hyperCubes_VIS={}

for i in range(1, len(visImagesNames)):
    imPath=visImagesDir+"/"+visImagesNames[i]
    im3chanel=cv.imread(imPath)#Lee la imagen
    im3chanel=cv.resize(im3chanel, (64,32))#Redimensiona
    im1chanel=cv.cvtColor(im3chanel, cv.COLOR_BGR2GRAY)#Escala de gris ortodoxa
    
    nFlowerID=visImagesNames[i].split("_")[6]#Extrae ID
    if(nFlowerID==flowerID and i<(len(visImagesNames)-1)):#Chequea si pertenece a la misma flor
        hyperCube.append(im1chanel)#Apila en el hipercubo
    elif(nFlowerID != flowerID and i<(len(visImagesNames)-1)): #Cuando la flor cambie
        hyperCubeArr=np.array(hyperCube, dtype=np.uint8)
        hyperCubes_VIS.update({flowerID:hyperCubeArr})#Añade al dict
        flowerID=nFlowerID#Actualiza
        hyperCube=[im1chanel]
    elif(i==(len(visImagesNames)-1)):#Último hipercubo
        hyperCube.append(im1chanel)
        hyperCubeArr=np.array(hyperCube, dtype=np.uint8)
        hyperCubes_VIS.update({flowerID:hyperCubeArr})#Añade al dict
        
    
        
#%% Hipercubos IR
#Lee directorios
irImagesDir="/home/albert/AlbertData/Bioinfo/TFM/NuevoBorradorPrograma/Data/flores/mandar/ir/hipercubo/"
irImagesNames=sorted(os.listdir(irImagesDir))

#Inicializa las estructuras de datos con la primera iteración
flowerID=irImagesNames[0].split("_")[6]
image3chanel=cv.imread(irImagesDir+"/"+irImagesNames[0])
image3chanel=cv.resize(image3chanel, (64,32))
image1chanel=cv.cvtColor(image3chanel, cv.COLOR_BGR2GRAY)
hyperCube=[image1chanel]
hyperCubes_IR={}

for i in range(1, len(irImagesNames)):
    imPath=irImagesDir+"/"+irImagesNames[i]
    im3chanel=cv.imread(imPath)#Lee la imagen
    #nFlowerID=irImagesNames[i].split("_")[6]#Extrae ID
    #print(nFlowerID, im3chanel.shape)
    im3chanel=cv.resize(im3chanel, (64,32))#Redimensiona
    im1chanel=cv.cvtColor(im3chanel, cv.COLOR_BGR2GRAY)#Escala de gris ortodoxa
    
    nFlowerID=irImagesNames[i].split("_")[6]#Extrae ID
    if(nFlowerID==flowerID and i<(len(irImagesNames)-1)):#Chequea si pertenece a la misma flor
        hyperCube.append(im1chanel)#Apila en el hipercubo
    elif(nFlowerID != flowerID and i<(len(irImagesNames)-1)):
        hyperCubeArr=np.array(hyperCube, dtype=np.uint8)
        hyperCubes_IR.update({flowerID:hyperCubeArr})#Añade al dict
        flowerID=nFlowerID#Actualiza
        hyperCube=[im1chanel]
    elif(i==(len(irImagesNames)-1)):
        hyperCube.append(im1chanel)
        hyperCubeArr=np.array(hyperCube, dtype=np.uint8)
        hyperCubes_IR.update({flowerID:hyperCubeArr})#Añade al dict
        
        
        
#%% Combina los hipercubos VIS y IR 
hyperCubes_Total={}
for ids in hyperCubes_VIS.keys():
    if (ids in hyperCubes_IR.keys()):
        vis=hyperCubes_VIS[ids]
        ir=hyperCubes_IR[ids]
        hyperCube_TotalArr=np.append(vis[1::,:,:], ir, axis=0) #Quita lambda 488.38
        #Formato canales al final
        hyperCube_TotalArr=np.transpose(hyperCube_TotalArr,
                                        (1,2,0))
        hyperCubes_Total.update({ids:hyperCube_TotalArr})
    else:
        print(ids)
#%%Crea hiperimagen compuesta con foto a y b
outputDir="/home/albert/AlbertData/Bioinfo/TFM/NuevoBorradorPrograma/Data/HipercubosReconstruidos/"
cmd="mkdir "+outputDir
os.system(cmd)
 
for fID in hyperCubes_Total.keys():
    if(re.match("f\d+a", fID)):#Junta las a
        compID=re.sub("a", repl="b", string=fID)#Con las b
        hyperCube_Tiled=np.zeros((64,64, 36), dtype=np.uint8)
        hyperCube_Tiled[0:32, 0:64, 0:36]=hyperCubes_Total[fID]
        hyperCube_Tiled[32:64, 0:64 , 0:36]=hyperCubes_Total[compID]
        
        imPath=outputDir+"/"+re.sub("a", repl="ab", string=fID)+".tif"
        tiff.imwrite(imPath, data=hyperCube_Tiled)


      