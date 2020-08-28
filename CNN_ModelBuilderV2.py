#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 12:44:23 2020

@author: albert
"""

#%% Carga los modulos

import tifffile as tiff
import random
import pickle
import numpy as np
import pandas as pd
import argparse as arg
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler, Normalizer
from sklearn.metrics import classification_report
#from keras.utils import to_categorical
from keras.optimizers import Adam
#from tfa.metrics.cohens_kappa import CohenKappa

import myCNN_ModelsV2


#%%Crea analizador de argumentos en linea de comando

parser=arg.ArgumentParser(description="Train a CNN on specified images and returns the model")
parser.add_argument("-i", "--inputImages", type=str, required=True,
                    help="Path to the input images")
parser.add_argument("-o", "--outputDir", type=str, required=True,
                    help="Directory to save output to")
parser.add_argument("-r", "--regression", default=False, 
                    required=False, action='store_true',
                    help="Regression mode instead of classification")
parser.add_argument("-m", "--metadata", type=str, required=True,
                    help="metadata file with class information")
#parser.add_argument("-s", "--Segmentation", type=str, required=False,
#                    default=None, help="Segmentation algorithm to apply, one of ndvi, kmeans or sam")
parser.add_argument("-l", "--learningRate", type=float, default=0.001,
                    required=False, help="Learning rate applied during training")
parser.add_argument("-e", "--epochs", type=int, default=100, required=False,
                    help="Number of training epochs")
parser.add_argument("-b", "--batchSize", type=int, default=16, required=False,
                    help="Number of images to use in each training step")
parser.add_argument("-d", "--imageDimensions", type=int, required=True,
                    nargs=3, help="Expected dimensions of the images, channels last format")
parser.add_argument("-M", "--modelType", type=str, required=True, 
                    help="One of lenet, vgg, resnet or inception")
parser.add_argument("-c","--configFile", type=str, required=False,
                    help="Optional text file with argument values for the configuration of the CNN model")
parser.add_argument("-x", "--mixedInputs", action="store_true", required=False,
                    default=False, help="Whether to combine images and covariables in a mixed input model")
parser.add_argument("-g", "--grayscale", required=False, default=False,
                    action="store_true", help="Whether the pixels are in a 0-255 range grayscale or not")
                    
args=vars(parser.parse_args())
#%%Carga las imagenes

dataDir=args["inputImages"]
metadata=pd.read_table(args["metadata"])
index=list(range(0,metadata.shape[0]))
random.seed(123)
random.shuffle(index)

images=[]
labels=[]
print("Loading images and labels...")

for i in index:
   image=tiff.imread(dataDir+metadata.iloc[i,0]+".tif")
   images.append(image)
   labels.append(metadata.iloc[i,-1])#Se supone que la ultima columna tiene siempre las etiquetas

if(args["mixedInputs"]==True):
    print("Loading covariables...")
    covars=metadata.iloc[index, 1:(metadata.shape[1]-1)]

    
      
#%% Preprocesamiento

#Segmentación con el metodo elegido

#Separa primero, normaliza después. Evita transferencia de informacion entre subconjuntos
#Vectorizacion. Convierte en arrays
images=np.array(images, dtype=type(images[0][0][0][0])) #Como saber el tipo uint???
if(list(images.shape[1::]) != args["imageDimensions"]):
    assInfo="The dimensions of the images are expected to be"+str(args["imageDimensions"]+":Received"+str(images.shape[1::]))
    raise AssertionError(assInfo)

labels=np.array(labels, dtype=type(labels[0]))

#Separa en conjunto de entrenamiento y evaluacion
print("Splitting data in train and test subsets...")
((trainX, testX, trainY, testY))=train_test_split(images, labels, 
test_size=0.25, random_state=25) 

#Separa también las covariables, si se trata de un modelo mixto
if(args["mixedInputs"]==True):
    (trainCovars, testCovars)=train_test_split(covars, test_size=0.25, 
                                               random_state=25)

#Normaliza las imagenes
if(args["grayscale"]==True):
    #Están en rango 0-255
    trainX=trainX/255
    testX=testX/255
else:
    #No se pueden normalizar la imagenes tal cual. 
    #Hay que comprimirla, transformarla y reconstruirla
    print("Flattening tensors to preprocess reflectance data...")
    nPixels_Subset=trainX.shape[0]*trainX.shape[1]*trainX.shape[2]
    flatTrain=np.reshape(trainX, (nPixels_Subset, trainX.shape[3]))
    nPixels_Subset=testX.shape[0]*testX.shape[1]*testX.shape[2]
    flatTest=np.reshape(testX, (nPixels_Subset, testX.shape[3]))

    print("Standardizing reflectances...")
    standardizer=StandardScaler().fit(flatTrain)
    Std_flatTrain=standardizer.transform(flatTrain)
    Std_flatTest=standardizer.transform(flatTest)

    print("Normalizing data to [0-1] range...")
    normalizer=Normalizer(norm='max').fit(Std_flatTrain)
    Norm_flatTrain=normalizer.transform(Std_flatTrain)
    Norm_flatTest=normalizer.transform(Std_flatTest)

    print("Compressing matrixes back to tensors...")
    trainX=np.reshape(Norm_flatTrain, (trainX.shape))
    testX=np.reshape(Norm_flatTest, (testX.shape))

#Formatea la variable de salida
if (args["regression"]==False):
    print("Encoding labels in one2hot form...")
    classesNames=np.unique(labels) #Para decodificar la prediccion binaria
    classes=len(classesNames)
    
    binarizer=LabelBinarizer().fit(trainY)
    trainY=binarizer.transform(trainY)
    testYori=testY #Para el informe de clasificacion
    testY=binarizer.transform(testY)
    
    #if(classes==2):#Si solo hay 2 clases, apicar to_categorical 
     #   trainY_bin=to_categorical(trainY)
     #  testY_bin=to_categorical(testY)
else: #Escala a rango [0-1] la variable de salida
    print("Scaling response variable to [0-1] range...")
    classes=0 #Para evitar fallos en definicion de argumentos
    maxY=max(trainY)
    trainY=trainY/maxY
    testY=testY/maxY

#Normaliza las covariables si el modelo es mixto
if(args["mixedInputs"]==True):

    print("Standardizing covariables...")
    covarStandardizer=StandardScaler().fit(trainCovars)
    Std_trainCovars=covarStandardizer.transform(trainCovars)
    Std_testCovars=covarStandardizer.transform(testCovars)
    
    print("Normalizing covariables to [0-1] range...")
    covarNormalizer=Normalizer(norm="max").fit(Std_trainCovars)
    Norm_trainCovars=covarNormalizer.transform(Std_trainCovars)
    Norm_testCovars=covarNormalizer.transform(Std_testCovars)
    
    trainCovars=Norm_trainCovars
    testCovars=Norm_testCovars    
#Aquí no se aumenta

#%% Ajusta el modelo

#Argumentos basicos de todos los modelos
modelArgs=dict(width=args["imageDimensions"][0],
               height=args["imageDimensions"][1],
               depth=args["imageDimensions"][2],
               classes=classes,
               regression=args["regression"])

#Argumentos de configuracion avanzados
if(args["configFile"]!=None):
    print("Reading configuration file...")
    extraArgs=myCNN_ModelsV2.ConfigFile_Reader(args["configFile"])
    modelArgs.update(extraArgs)

print("Initializing model....")
if(args["modelType"]=="lenet"):#Comprueba el tipo de arquitectura DL
    model=myCNN_ModelsV2.CNN_LeNet(**modelArgs)
elif(args["modelType"]=="vgg"):
    model=myCNN_ModelsV2.CNN_VGG(**modelArgs)
elif(args["modelType"]=="resnet"):
    model=myCNN_ModelsV2.CNN_ResNet(**modelArgs)
elif(args["modelType"]=="inception"):
    model=myCNN_ModelsV2.CNN_Inception(**modelArgs)
else:
    raise AssertionError("Model type must be one of lenet, vgg, resnet or inception")    

#Si el modelo usa entrada mixta, este es el momento
if(args["mixedInputs"]==True):
    auxModel=myCNN_ModelsV2.MLP_AuxBranch(nVars=trainCovars.shape[1],
                                          hiddenNodes=int(trainCovars.shape[1]/2),
                                          regression=True)
    model=myCNN_ModelsV2.MixedModel(CNN_model=model, MLP=auxModel, 
                                    hiddenNodes=4, regression=True)

if(args["regression"]==True): 
    lossFunc="mean_squared_error"
    metrics=["mse", myCNN_ModelsV2.detCoefficient]
else:
    lossFunc="categorical_crossentropy" if(classes>2) else "binary_crossentropy"
    metric="accuracy" if(classes>2) else "binary_accuracy"
    #metrics=[metric, CohensKappa]

optimizer=Adam(learning_rate=args["learningRate"])

if(args["modelType"]!="inception"): #Inception tiene tres salidas combinadas
    model.compile(optimizer=optimizer, loss=lossFunc, metrics=metrics)
else:
    model.compile(optimizer=optimizer, loss=[lossFunc, lossFunc, lossFunc],
                  loss_weights=[0.3,0.3,1], metrics=[metric])
    
fitArgs=dict(x=trainX, y=trainY,
            batch_size=args["batchSize"],
            epochs=args["epochs"],
            validation_data=(testX, testY)) #Caso simple

print("Fitting model...")
#Adapta el ajuste a la casuística
try:
    if(args["modelType"]!="inception" and args["mixedInputs"]==True):#Mixto
        fitArgs.update(x=[trainX, trainCovars], 
                       validation_data=([testX, testCovars], testY))
    elif(args["modelType"]=="inception" and args["mixedInputs"]==False):#Inception
        fitArgs.update(y=[trainY, trainY, trainY],
                       validation_data=(testX, [testY, testY, testY]))
    elif(args["modelType"]=="inception" and args["mixedInputs"]==True):#Ambos
        fitArgs.update(x=[trainX, trainCovars],
                       validation_data=([testX, testCovars],
                                        [testY, testY, testY]))
    modelFit=model.fit(**fitArgs)
except ValueError: #Caso de convoluciones 3D, hay que añadir una dimension
    newDims=list(trainX.shape)
    newDims.append(1)    
    trainX=np.reshape(trainX, (newDims))
    newDims=list(testX.shape)
    newDims.append(1)
    testX=np.reshape(testX, (newDims))
    fitArgs.update(x=trainX, validation_data=(testX, testY))#Caso simple
    
    if(args["modelType"]!="inception" and args["mixedInputs"]==True):#Mixto
        fitArgs.update(x=[trainX, trainCovars], 
                       validation_data=([testX, testCovars], testY))
    elif(args["modelType"]=="inception" and args["mixedInputs"]==False):#Inception
        fitArgs.update(y=[trainY, trainY, trainY],
                       validation_data=(testX, [testY, testY, testY]))
    elif(args["modelType"]=="inception" and args["mixedInputs"]==True):#Ambos
        fitArgs.update(x=[trainX, trainCovars],
                       validation_data=([testX, testCovars],
                                        [testY, testY, testY]))
    modelFit=model.fit(**fitArgs)
    

print("Saving model...")
model.save(args["outputDir"]+"ModelReady.h5")
print("Saving model history...")
pickle_file=open(args["outputDir"]+"ModelHistory", "wb")
pickle.dump(modelFit.history, pickle_file)
pickle_file.close()


if(args["grayscale"]==False):#Solo si se han usado
    print("Saving data preprocessers...")
    pickle_file=open(args["outputDir"]+"standardizer", "wb")
    pickle.dump(standardizer, pickle_file)
    pickle_file.close()
    pickle_file=open(args["outputDir"]+"normalizer", "wb")
    pickle.dump(normalizer, pickle_file)
    pickle_file.close()
    
if(args["mixedInputs"]==True):#Idem
    print("Saving covariable preprocessers...")
    pickle_file=open(args["outputDir"]+"covar_standardizer", "wb")
    pickle.dump(covarStandardizer, pickle_file)
    pickle_file.close()
    pickle_file=open(args["outputDir"]+"covar_normalizer", "wb")
    pickle.dump(covarNormalizer, pickle_file)
    pickle_file.close()

if (args["regression"]==False):
    print("Saving binarizer...")
    pickle_file=open(args["outputDir"]+"binarizer", "wb")
    pickle.dump(binarizer,pickle_file)
    pickle_file.close()
#%%Evalua el modelo
print("Evaluating model...")
if(args["modelType"]!="inception" and args["mixedInputs"]==False):
    metrics=model.evaluate(x=testX, y=testY, batch_size=args["batchSize"])
elif(args["modelType"]!="inception" and args["mixedInputs"]==True):
    metrics=model.evaluate(x=[testX, testCovars], y=testY, 
                           batch_size=args["batchSize"])
elif(args["modelType"]=="inception" and args["mixedInputs"]==False):
    metrics=model.evaluate(x=testX, y=[testY, testY, testY],
                           batch_size=args["batchSize"])
else:
    metrics=model.evaluate(x=[testX, testCovars], y=[testY, testY, testY],
                           batch_size=args["batchSize"])

if(args["regression"]==True):  
    print("Final mean squared error:", round(metrics[0],3))
    
    #Guarda el resultado
    if(args["mixedInputs"]==False):
        prediction=model.predict(testX, batch_size=args["batchSize"])
    else:
        prediction=model.predict([testX, testCovars], batch_size=args["batchSize"])
    
    if(args["modelType"]=="inception"): #Hace tres predicciones
        prediction=prediction[2]
    
    error=(prediction.flatten()-testY)**2
    mse=round(np.mean(error),3)
    mse_sd=round(np.std(error),3)
    
    file=open(args["outputDir"]+"RegressionReport.txt", "w")
    file.write("mse:\t"+str(mse)+"\n")
    file.write("mse_sd:\t"+str(mse_sd)+"\n")
    file.close()    
else:
    print("Final loss on test data:",round(metrics[0],3))
    print("Final accuracy on test data:", round(metrics[1],3))
    
    #Guarda el resultado
    if(args["mixedInputs"]==False):
        prediction=model.predict(testX, batch_size=args["batchSize"])
    else:
        prediction=model.predict([testX, testCovars], batch_size=args["batchSize"])
    
    if(args["modelType"]=="inception"): #Hace tres predicciones
        prediction=prediction[2]
    
    predClasses=[]
    for i in range(0, prediction.shape[0]):
        Class=np.argmax(prediction[i])
        predClasses.append(classesNames[Class])
        
    report=classification_report(testYori, predClasses, output_dict=True)
    report_df=pd.DataFrame.from_dict(report)
    report_df.to_csv(args["outputDir"]+"Report.csv", sep="\t")

#%%Grafica
print("Creating summary plot...")
N = np.arange(0, args["epochs"])
plt.style.use("ggplot")
plt.figure()
if(args["regression"]==False):
    plt.plot(N, modelFit.history["loss"], label="train_loss")
    plt.plot(N, modelFit.history["val_loss"], label="val_loss")
    if(classes>2 and args["modelType"]!="inception"):
        plt.plot(N, modelFit.history["accuracy"], label="train_acc")
        plt.plot(N, modelFit.history["val_accuracy"], label="val_acc")
    
    elif(classes>2 and args["modelType"]=="inception"): #Hay que especificar la precision principal
        plt.plot(N, modelFit.history["MainOutput_accuracy"], label="train_acc")
        plt.plot(N, modelFit.history["val_MainOutput_accuracy"], label="val_acc")
    
    elif(classes==2 and args["modelType"]!="inception"):
        plt.plot(N, modelFit.history["binary_accuracy"], label="train_acc")
        plt.plot(N, modelFit.history["val_binary_accuracy"], label="val_acc")
    
    else:
        plt.plot(N, modelFit.history["MainOutput_binary_accuracy"], label="train_acc")
        plt.plot(N, modelFit.history["val_MainOutput_binary_accuracy"], label="val_acc")
    plt.title("Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
else:
    if(args["modelType"]!="inception"):
        plt.plot(N, modelFit.history["mse"], label="train_mse")
        plt.plot(N, modelFit.history["val_mse"], label="val_mse")
    else:
        plt.plot(N, modelFit.history["MainOutput_mse"], label="train_mse")
        plt.plot(N, modelFit.history["val_MainOutput_mse"], label="val_mse")
    plt.title("Loss or MSE")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/MSE")

plt.legend()
plt.savefig(args["outputDir"]+"TrainingLoss.png")
plt.show()

print("Done")