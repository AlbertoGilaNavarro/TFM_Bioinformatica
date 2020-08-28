#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 12:00:39 2020

@author: albert
"""

import os 
import re

#Objetivo: 
    #Separar 15000 imágenes en 5 conjuntos de 3000
#%% Crea 5 directorios
outputDir="/home/albert/AlbertData/Bioinfo/TFM/NuevoBorradorPrograma/Data/"
for i in range(0,5):
    subsetName="AugDataset"+str(i+1)
    outputPath=outputDir+subsetName
    cmd="mkdir "+outputPath
    os.system(cmd)
#%% Corta y pega las imagenes en sus nuevos directorios
augImagesDir="/home/albert/AlbertData/Bioinfo/TFM/NuevoBorradorPrograma/Data/Aug_HipercubosReconstruidos/AugHyperspectralImages/"
augImagesNames=os.listdir(augImagesDir)
regex=re.compile("Batch(\d+$)")
cte=int(3000/106)+1
for imName in augImagesNames:
    batch=imName.split("_")[0]
    batch=int(regex.search(batch).group(1))#Extrae el lote
    cmd="mv "+augImagesDir+imName+" "+outputDir
    if(batch<=cte):#Pega la imagen según el lote
        cmd=cmd+"AugDataset1"
    elif(batch>cte and batch<=cte*2):
        cmd=cmd+"AugDataset2"
    elif(batch>cte*2 and batch<=cte*3):
        cmd=cmd+"AugDataset3"
    elif(batch>cte*3 and batch<=cte*4):
        cmd=cmd+"AugDataset4"
    else:
        cmd=cmd+"AugDataset5"
    os.system(cmd)
    
    
    
    
    
    
    
    
    
    
    
    