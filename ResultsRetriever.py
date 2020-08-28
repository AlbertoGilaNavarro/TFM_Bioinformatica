#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 20:09:35 2020

@author: albert
"""

import pickle 
#from keras.models import load_model
import os
import pandas as pd
import csv

#Objetivo: Extraer las metricas de los resultados y devolver 
#una tabla con cada metrica de los historiales, 

mainPath="/home/albert/AlbertData/Bioinfo/TFM/NuevoBorradorPrograma/Data/Run2Results/"
algoDirs=os.listdir(mainPath)
#%% Extrae valores de metricas
histories_mse={}
histories_rsquared={}

for a in algoDirs:
    metr=[]
    modelDirs=os.listdir(mainPath+a)
    for m in modelDirs:
        #Historial
        pickled_file=open(mainPath+a+"/"+m+"/ModelHistory", "rb")
        hist=pickle.load(pickled_file)
        histories_mse.update(
            {a+"_"+m+"_train":hist["mse"],
             a+"_"+m+"_vali":hist["val_mse"]})
        histories_rsquared.update(
            {a+"_"+m+"_train":hist["detCoefficient"],
             a+"_"+m+"_vali":hist["val_detCoefficient"]})
#%% Convierte en data frames y guardalos como tablas tsv        
mse_df=pd.DataFrame.from_dict(histories_mse)
rsquared_df=pd.DataFrame.from_dict(histories_rsquared)

outputDir="/home/albert/AlbertData/Bioinfo/TFM/NuevoBorradorPrograma/Data/"

mse_df.to_csv(outputDir+"MSE.tsv", sep='\t', index=False, header=True,
                  quoting=csv.QUOTE_NONE, quotechar=None, doublequote=False,
                  escapechar="\n")
rsquared_df.to_csv(outputDir+"Rsquared.tsv", sep="\t", index=False, header=True,
                   quoting=csv.QUOTE_NONE, quotechar=None, doublequote=False,
                   escapechar="\n")




    