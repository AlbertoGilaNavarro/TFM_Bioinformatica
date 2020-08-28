#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 20:55:39 2020

@author: albert
"""
#%% Carga modulos
import tifffile as tiff
import cv2 as cv
import os
import pickle
import argparse as arg
import numpy as np
import pandas as pd
from keras.models import load_model


#%% Crea analizador de argumentos

parser=arg.ArgumentParser()
parser.add_argument("-i", "--inputImages", type=str, required=True,
                    help="Input file directory with Hyperspectral images")
parser.add_argument("-o", "--outputDir", type=str, required=True,
                    help="Output directory to write prediction file")
parser.add_argument("-I", "--rgbImages", type=str, required=False, default=None,
                    help="Optional rgb version of input images, used for showing results visually")
parser.add_argument("-d", "--imageDimensions", type=int, nargs=3, required=True,
                    help="Expected dimensions of the images, channels last format")   
parser.add_argument("-m", "--model", type=str, required=True,
                    help="Model to use")
parser.add_argument("-r", "--regression", required=False,
                    default=False, action='store_true',
                    help="Whether model performs regression, instead of classification")
parser.add_argument("-c", "--covars", type=str, required=False, default=None,
                    help="For models of mixed inputs, the covariable file in tsv format")
parser.add_argument("-b", "--binarizer", type=str, required=False,
                    help="Which label binarizer to use")
parser.add_argument("-s", "--standardizer", type=str, required=False, default=None,
                    help="Which data standardizer to use")
parser.add_argument("-n", "--normalizer", type=str, required=False, default=None,
                    help="Which data normalizer to use")
parser.add_argument("-S", "--metaStandardizer", type=str, required=False, default=None,
                    help="Which metadata standardizer to use")
parser.add_argument("-N", "--metaNormalizer", type=str, required=False, default=None,
                    help="Which metadata normalizer to use")
parser.add_argument("-y", "--yMaxValue", type=float, required=False,
                    help="If regression, the max value of trainY to revert predictions from [0-1] range to real values")
parser.add_argument("-R", "--random", type=int, default=0,
                    help="Select n instances randomly instead of all of them")
                 
args=vars(parser.parse_args())

#%% Carga las imagenes

print("Loading images...")
imagesNames=sorted(os.listdir(args["inputImages"]))
images=[]

if(args["random"]!=0):
    index=list(np.random.randint(0, len(imagesNames), size=args["random"]))
else:
    index=list(range(0, len(imagesNames)))

if(args["rgbImages"]!=None):
    print("Loading supplementary RGB images...")
    rgbImages=[]
    rgb_imagesNames=sorted(os.listdir(args["rgbImages"]))
    for i in index:
        image=tiff.imread(args["inputImages"]+"/"+imagesNames[i])
        images.append(image)
        rgb=cv.imread(args["rgbImages"]+"/"+rgb_imagesNames[i])
        rgb=cv.resize(rgb, (256,256))
        rgbImages.append(rgb)
else:
    for i in index:
        image=tiff.imread(args["inputImages"]+"/"+imagesNames[i])
        images.append(image)
        
#%%Carga los metadatos, si se pide
if(args["covars"]!=None):
    import re
    print("Loading metadata...")
    metadata=pd.read_table(args["covars"], index_col=0)
    imagesFileNames=[imagesNames[i] for i in index]
    imagesID=[re.sub("(.+)\\.tif", "\g<1>", im) for im in imagesFileNames]
    covars=metadata.loc[imagesID] #Selecciona las filas en el mismo orden que se leen las imagenes
    covars=metadata.iloc[:, 0:(metadata.shape[1]-1)]
#%% Carga el modelo y los objetos preprocesadores      
print("Loading model...")

if (args["regression"]==False):
    model=load_model(args["model"])
    print("Loading binarizer...")
    pickled_file=open(args["binarizer"], "rb")
    binarizer=pickle.load(pickled_file)
    pickled_file.close()
else:
    from myCNN_ModelsV2 import detCoefficient
    model=load_model(args["model"], 
                     custom_objects={"detCoefficient":detCoefficient})

if(args["normalizer"] !=None and args["standardizer"]!=None):
    print("Loading data preprocessers...")
    pickled_file=open(args["standardizer"], "rb")
    standardizer=pickle.load(pickled_file)
    pickled_file.close()

    pickled_file=open(args["normalizer"], "rb")
    normalizer=pickle.load(pickled_file)
    pickled_file.close()

if(args["covars"]!=None):
    print("Loading metadata preprocessers...")
    pickled_file=open(args["metaStandardizer"], "rb")
    metaStandardizer=pickle.load(pickled_file)
    pickled_file.close()

    pickled_file=open(args["metaNormalizer"], "rb")
    metaNormalizer=pickle.load(pickled_file)
    pickled_file.close()
    
#%% Preprocesa las imagenes para que las acepte el modelo
images=np.array(images, dtype=type(images[0][0][0][0]))
if(list(images.shape[1::]) != args["imageDimensions"]):
    assInfo="The dimensions of the images are expected to be"+str(args["imageDimensions"])
    raise AssertionError(assInfo)

#Si no hay preprocesadores, por defecto se aplica normalizacion a rango 0-1 diviendo por 255

if(args["normalizer"] !=None and args["standardizer"]!=None):
    print("Flattening tensor to preprocess reflectance data...")
    nPixels_Total=images.shape[0]*images.shape[1]*images.shape[2]
    flatImages=np.reshape(images, (nPixels_Total, images.shape[3]))

    print("Standardizing reflectances as in training... ")
    Std_flatImages=standardizer.transform(flatImages)

    print("Normalizing data to [0-1] range as in training...")
    Norm_flatImages=normalizer.transform(Std_flatImages)

    print("Compressing matrix back to tensor...")
    imagesReady=np.reshape(Norm_flatImages, (images.shape))
else:
    imagesReady=images/255
    
#%% Preprocesa las covariables, si se pide
if(args["covars"]!=None):
    print("Standardizing covariables...")
    Std_Covars=metaStandardizer.transform(covars)
    
    print("Normalizing covariables to [0-1] range...")
    Norm_Covars=metaNormalizer.transform(Std_Covars)
    
    covarsReady=Norm_Covars

#%% Realiza las predicciones
print("Predicting values...")
if(args["covars"]!=None):
    predictions=model.predict([imagesReady,covarsReady])
else:
    predictions=model.predict(imagesReady)

if(len(predictions)==3):#Caso de inception, tiene 3 predicciones
    predictions=predictions[2]
    
file=open(args["outputDir"]+"/Predictions.txt", "w")
if(args["regression"]==False):
    file.write("Image\tPrediction\tProbability\n")
    for (i,p) in enumerate (predictions):
        maxProbIndex=np.argmax(p)#Extrae maxima probabilidad
        Class=binarizer.classes_[maxProbIndex]#Extrae clase correspondiente
        maxProb=round(p[maxProbIndex]*100,3)
        imName=imagesNames[index[i]]
        file.write(imName+"\t"+Class+"\t"+str(maxProb)+"\n")
        
        if(args["rgbImages"]!=None): #Pega texto en imagen si se pide
            imText=Class + ":" + str(maxProb)+'%'
            cv.putText(rgbImages[i], text=imText, org=(10,30), 
                       fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                       color=(0,0,255), thickness=2)    
else:
    file.write("Image\tPrediction\n")
    for(i,p) in enumerate(predictions):
        value=round(p[0]*args["yMaxValue"],3) #Convierte de 0-1 a real
        imName=imagesNames[index[i]]
        file.write(imName+"\t"+str(value)+"\n")
        
        if(args["rgbImages"]!=None): #Pega texto en imagen
            imText="Predicted value:"+str(value)
            cv.putText(rgbImages[i], text=imText, org=(10,30), 
                       fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                       color=(0,0,255), thickness=2)
file.close()

print("Done")
    

