#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 10:49:57 2020

@author: albert
"""
#%% Generaador Aumentador
import os
import random
import numpy as np
import argparse as arg
import cv2 as cv
import tifffile as tiff
import pandas as pd
from skimage.transform import AffineTransform, SimilarityTransform, warp
import matplotlib.pyplot as plt

#%%Crea analizador sintactico de linea de comandos
parser=arg.ArgumentParser()
parser.add_argument("-i", "--inputImages", type=str, required=True,
                    help="Directory with the input images")
parser.add_argument("-I", "--rgbImages", type=str, required=False, default=None,
                    help="Optional directory with rgb images to augment alongside multispectral ones")
parser.add_argument("-m", "--metadata", type=str, required=True,
                    help="Metadata with labels for each image")
parser.add_argument("-o", "--outputDir", type=str, required=True,
                    help="Directoty to save output to")
parser.add_argument("-b", "--batchSize", type=int, required=False, default=32,
                    help="Number of images generated per augmentation iteration")
parser.add_argument("-a", "--augmCycles", type=int, required=False, default=1,
                    help="Number of augmentation iterations")
parser.add_argument("-R", "--rotationRange", type=int, required=False, 
                    default=30, help="Degrees of rotation to apply")
parser.add_argument("-S", "--shearRange", type=int, required=False,
                    default=0, help="Degrees os shear to apply" )
parser.add_argument("-Z", "--zoomRange", type=float, required=False,
                    default=2, help="Rate of zoom scaling to apply")
parser.add_argument("-T", "--translationRange",type=float,required=False,
                    default=0.1, help="Rate of translation to apply ")
parser.add_argument("-H", "--horizontalFlip", action="store_true",
                    required=False, default=False, help="Whether to apply horizontal flip")
parser.add_argument("-V", "--verticalFlip", action="store_true",
                    required=False, default=False, help="Whether to apply vertical flip")
args=vars(parser.parse_args())
#%% Funcion de aumentacion

def Custom_AugGenerator_Builder(imagesArray, imagesNames, labelsArray=None, 
                                batchSize=32, rotationRange=0, 
                                shearRange=0, scaleRange=0, 
                                translationRange=0, hFlip=False, 
                                vFlip=False, fillMode="edge",seed=42):
    np.random.seed(seed=seed)
    while(True):
        #Selecciona al azar un lote de tamaño n para modificar
        Index=np.random.choice(range(0, imagesArray.shape[0]), batchSize, replace=False)
        batchImages=imagesArray[Index]
        dims=list(imagesArray.shape)
        dims[0]=batchSize
        augImagesBatch=np.zeros(shape=dims, dtype=np.uint8)
        
        #Selecciona los nombres correspondientes
        batchNames=[imagesNames[i] for i in Index]
        
        for i in range(0,batchSize):
            image=batchImages[i]
            
            #Voltea las imagenes al azar si se pide
            if(vFlip==True):
                if(np.random.choice([True, False], size=1)):
                    image=np.flipud(image)
            if(hFlip==True):
                if(np.random.choice([True, False], size=1)):
                    image=np.fliplr(image)
            
            #Establece los rangos de transformación con distribuciones uniformes
            
            #La rotacion y la cizalla van en radianes luego
            rotationAngle=np.random.uniform(low=-abs(rotationRange), high=abs(rotationRange))
            shearAngle=np.random.uniform(low=-abs(shearRange), high=abs(shearRange))
            rotationAngle=np.deg2rad(rotationAngle)
            shearAngle=np.deg2rad(shearAngle)
            
            #EL escalado va de casi 0 a lo que se especifique
            scaleValue=np.random.uniform(low=abs(1/scaleRange), high=abs(scaleRange))
            #La translacion se tiene que indicar 2 veces, para el eje X y el Y
            translationValues=(np.random.uniform(low=-abs(translationRange), high=abs(translationRange)),
                              np.random.uniform(low=-abs(translationRange), high=abs(translationRange)))
        
            #Transforma con skimage. La version antigua de skimage necesita (scaleValue, scaleValue)
            transform=AffineTransform(scale=scaleValue,rotation=rotationAngle,
                                        shear=shearAngle, translation=translationValues)
            #Lleva la imagen al origen de coordenadas
            transform_2Ori=SimilarityTransform(scale=1, rotation=0, 
                                               translation=(-image.shape[0], -image.shape[1]))
            #Devuelve la imagen a donde estaba
            transform_revert=SimilarityTransform(scale=1, rotation=0,
                                                 translation=(image.shape[0], image.shape[1]))
            
            imageAug=warp(image, inverse_map=(transform_2Ori+transform)+transform_revert, 
                          preserve_range=True,  mode=fillMode)
            #imageAug=warp(image, inverse_map=transform,mode=fillMode)
            
            imageAug=np.array(imageAug, dtype=np.uint8)
            #Rellena el tensor de salida
            augImagesBatch[i]=imageAug
        
        #Selecciona las etiquetas correspondientes
        if(labelsArray is None):
            yield(augImagesBatch, batchNames)
        else:
            augLabelsBatch=[labelsArray[i] for i in Index]
            yield(augImagesBatch, augLabelsBatch, batchNames)        
            
#%% Carga las imagenes y las etiquetas
print("Loading images and labels...")
dataDir=args["inputImages"]
metadata=pd.read_table(args["metadata"])
index=list(range(0,metadata.shape[0]))
random.seed(123)
random.shuffle(index)

hypImages=[]
hypNames=[]
labels=[]
if(args["rgbImages"]!=None):
    rgbImages=[]
    rgbNames=os.listdir(args["rgbImages"])
    rgbNames2=[]
    print("Loading supplementary rgb images...")

for i in index:
   hyp=tiff.imread(dataDir+metadata.iloc[i,0]+".tif")
   hypImages.append(hyp)
   hypNames.append(metadata.iloc[i,0])
   labels.append(metadata.iloc[i,1])
   if(args["rgbImages"]!=None):#Se supone que estan ordenados igual
       rgb=cv.imread(args["rgbImages"]+rgbNames[i])
       rgbImages.append(rgb)
       rgbNames2.append(rgbNames[i])

#%%Inicilaiza el generador
print("Initializing augmentation generator...")
hypImages=np.array(hypImages, dtype=np.uint8)
if(args["rgbImages"]!=None):
    rgbImages=np.array(rgbImages, dtype=np.uint8)
labels=np.array(labels, dtype=type(labels[0]))

augArgs=dict(imagesArray=hypImages, imagesNames=hypNames, 
             labelsArray=labels, batchSize=args["batchSize"], 
             rotationRange=args["rotationRange"],
             shearRange=args["shearRange"], scaleRange=args["zoomRange"],
             translationRange=args["translationRange"], 
             hFlip=args["horizontalFlip"], vFlip=args["verticalFlip"])

Generator=Custom_AugGenerator_Builder(**augArgs)
#%% Aumenta las imagenes hyperespectrales
print("Augmenting multichannel images...")
outDir=args["outputDir"]+"AugHyperspectralImages/"
cmd="mkdir "+outDir
os.system(cmd)
AugLabels=open(args["outputDir"]+"AugLabels.txt", "w")
i=1
while(i<=args["augmCycles"]):
    (imagesBatch, labelsBatch, namesBatch)=next(Generator)
    for j in range(0, imagesBatch.shape[0]):
        name="Batch"+str(i)+"_Aug"+str(j+1)+"_"+namesBatch[j]+".tif"
        tiff.imwrite(outDir+name, imagesBatch[j])
        AugLabels.write(name+"\t"+str(labelsBatch[j])+"\n")
    i+=1
AugLabels.close()
print("Done")
#%%Aumenta las imagenes rgb, si se especifica
if(args["rgbImages"]!=None):
    print("Augmenting rgb images...")
    augArgs.update(labelsArray=None, imagesArray=rgbImages, 
                   imagesNames=rgbNames2)
    Generator=Custom_AugGenerator_Builder(**augArgs)
    
    outDir=args["outputDir"]+"AugRGBImages/"
    cmd="mkdir "+outDir
    os.system(cmd)

    i=1
    while(i<=args["augmCycles"]):
        (imagesBatch, namesBatch)=next(Generator)
        for j in range(0, imagesBatch.shape[0]):
            name="Batch"+str(i)+"_Aug"+str(j+1)+"_"+namesBatch[j]+".png"
            cv.imwrite(outDir+name, imagesBatch[j])
        i+=1
    print("Done")
