    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 10:54:19 2020

@author: albert
"""
#%% 
import re
from keras.models import Input, Model
from keras.layers import Conv2D, Conv3D
from keras.layers import Activation,BatchNormalization
from keras.layers import ZeroPadding2D, ZeroPadding3D
from keras.layers import MaxPooling2D, MaxPooling3D, AveragePooling2D, AveragePooling3D
from keras.layers import Dropout, Flatten, Dense
from keras.layers import concatenate, UpSampling2D, Add
from keras.initializers import glorot_uniform, he_uniform
from keras import backend as K
import numpy as np
#Ahora está TODO parametrizado. 
#Modelos LeNet, VGG16, ResNet50, InceptionV1, UNet
#Exclusivamente API funcional
#Todos los modelos están en version 3D y 2D
#Posibilidad de crear un modelo mixto con un MLP auxiliar
#Incluye funcion para leer ficheros de configuración
#Incluye funcion para implementar metrica R^2 en regresion

#%% Lectura de ficheros de configuracion
def ConfigFile_Reader(file):
    file=open(file, "r")
    modelArgs={}
    for line in file:
        line=line.rstrip()
        sline=line.split("\t")
        if(len(sline)!=2):
            raise AssertionError("The config file must have two columns in tsv format, one with the args names and another with args values")
        value=sline[1] #Examina los valores y su tipo
        if(re.search("^\\(+\\d,\\d", value)):#Lista o matriz
            find=re.findall("^\\(+", value)[0]
            dim=find.count("(")
            value_list=list(value) #Convierte en lista de simbolos
            value_ready=[]
            if(dim==1):#Lista
                for symbol in value_list:
                    try:
                        value_ready.append(int(symbol))
                    except ValueError:
                        continue
            elif(dim==2):#Matriz
                sublist=[]
                for (i, symbol) in enumerate(value_list):
                    if(symbol!=")"):
                        try:
                            sublist.append(int(symbol))
                        except ValueError:
                            continue
                    else:
                        value_ready.append(sublist)
                        sublist=[]
                        if(i==(len(value_list)-2)):
                            break 
            modelArgs[sline[0]]=value_ready
        elif(re.match('True', value)):# Valor booleano True
            modelArgs[sline[0]]=bool(0==0) #Truco para sacar un valor booleano
        elif(re.match('False', value)): #Valor booleano False
            modelArgs[sline[0]]=bool(0==1)
        else: #Numero o caracter
            try:
                if(re.search("\\.", value)):#Real
                    modelArgs[sline[0]]=float(value) #Numero
                else:
                    modelArgs[sline[0]]=int(value) #Entero
            except ValueError:
                modelArgs[sline[0]]=sline[1] #Caracter
    file.close()
    return(modelArgs)

#%% Basado en LeNet-5. Un bloque más
def LeNet_Block(inputLayer, nFilters, filterSize, strides, poolReduction,
                batchNorm=False, chanDim=-1, paddType="same",
               kernelInit="glorot_uniform", actiFunc="relu"):
    #Comrpueba coherencia de tamaños
    sizes=(len(filterSize), len(strides), len(poolReduction))
    if(all(s==2 for s in sizes) or all(s==3 for s in sizes)): #Maxima eficiencia
        
        #Muy importante el inicializador de los pesos. Y la semilla de aleatorizacion para hacerlo reproducible
        if(kernelInit=="glorot_uniform"):
            initFunc=glorot_uniform(seed=0)
        elif(kernelInit=="he_uniform"):
            initFunc=he_uniform(seed=0)
        else:
            raise AssertionError("Only glorot_uniform or he_uniform are accepted as weight initializers")
        
        if(paddType not in ("same","valid")):
            raise AssertionError("Only same (pad) or valid (do not pad) are accepted as padding types")
        
        #Implementa convolucion 3D si las dimensiones son apropiadas   
        
        #Compacta codigo
        convArgs=dict(filters=nFilters, kernel_size=filterSize, 
                     strides=strides, padding=paddType, 
                     kernel_initializer=initFunc)
        
        if(len(filterSize)==2):
            x=Conv2D(**convArgs)(inputLayer)
        elif(len(filterSize)==3):
            x=Conv3D(**convArgs)(inputLayer)
    else:
        raise AssertionError("In LeNet block: filterSize, poolReduction and strides length must be the same, and either 2 or 3")
    
    x=Activation(actiFunc)(x)
    
    if(batchNorm==True):
        x=BatchNormalization(axis=chanDim)(x)
    
    if(len(poolReduction)==2):
        x=AveragePooling2D(pool_size=poolReduction)(x)
    elif(len(poolReduction)==3):
        x=AveragePooling3D(pool_size=poolReduction)(x)
    
    return(x)

def CNN_LeNet(width, height, depth, classes=0, regression=False,
              nInitFilters=32, filterSize=(5,5), strides=(1,1),
              hiddenUnits=(120,84), poolReduction=(2,2), batchNorm=True, 
              dropout=0, kernelInit="he_uniform", actiFunc="relu", paddType="same"):

    #Establece formato de definicion de imagen
    if(K.image_data_format()=="channels_first"):
        if(len(filterSize)==2):
            input_layer=Input(shape=(depth, width, height))
            chanDim=1
        elif(len(filterSize)==3):
            input_layer=Input(shape=(depth, width, height,1))
            chanDim=1
    else:
        if(len(filterSize)==2):
            input_layer=Input(shape=(width, height, depth))
            chanDim=-1
        elif(len(filterSize)==3):
            input_layer=Input(shape=(width, height, depth,1))
            chanDim=-1
    
    #Argumentos principales del bloque
    blockArgs=dict(inputLayer=input_layer, nFilters=nInitFilters,
                   filterSize=filterSize, strides=strides, 
                   paddType=paddType, poolReduction=poolReduction,
                   batchNorm=batchNorm, chanDim=chanDim,
                   kernelInit=kernelInit, actiFunc=actiFunc)
    
    
    #Bloque 1. 1 convolucion=>Relu=>Normalizacion=>Reduccion    
    x=LeNet_Block(**blockArgs)
    x=Dropout(dropout)(x)
    #Bloque2.
    blockArgs.update(inputLayer=x, nFilters=nInitFilters*2)
    x=LeNet_Block(**blockArgs)
    x=Dropout(dropout)(x)
    
    #Bloque3
    blockArgs.update(inputLayer=x, nFilters=nInitFilters*4)
    x=LeNet_Block(**blockArgs)
    x=Dropout(dropout)(x)
    
    #Bloque4: MLP. Compresion=>(CapaOculta=>Relu=>Norm)x2=>Salida
    if (chanDim == 1):
       x=Flatten(data_format="channels_first")(x)
    else:
       x=Flatten(data_format="channels_last")(x)
    
    if(kernelInit=="he_uniform"):
        initFunc=he_uniform(seed=0)
    elif(kernelInit=="glorot_uniform"):
        initFunc=glorot_uniform(seed=0)
    else:
        raise AssertionError("Only he_uniform or glorot_uniform are accepted as weight initializers")
    
    x=Dense(units=hiddenUnits[0], activation=actiFunc, 
            kernel_initializer=initFunc)(x)
    if(batchNorm==True):
        x=BatchNormalization(axis=chanDim)(x)
    x=Dense(units=hiddenUnits[1], activation="relu",
            kernel_initializer=initFunc)(x)
    if(batchNorm==True):
        x=BatchNormalization(axis=chanDim)(x)
    x=Dropout(dropout)(x)
    
    if(regression==False and classes>2):
        output_layer=Dense(units=classes, activation="softmax")(x)
    elif(regression==False and classes==2):
        output_layer=Dense(units=classes, activation="sigmoid")(x)
    else:
        output_layer=Dense(units=1, activation="linear")(x)
    
    model=Model(inputs=input_layer, outputs=output_layer)
    return(model)
#%% Basado en VGG16. Un bloque de convolucion más, pero una capa oculta menos

def VGG_Block(inputLayer, nFilters, blockSize, filterSize, strides, poolReduction,
              batchNorm=False, chanDim=-1, paddType="same",
              kernelInit="glorot_uniform", actiFunc="relu"):
    #Comprueba la coherencia de los tamaños
    sizes=(len(filterSize), len(strides), len(poolReduction))
    if(all(s==2 for s in sizes) or all(s==3 for s in sizes)): #Maxima eficiencia
        x=inputLayer #Para poder iterar facilmente
        
        if(kernelInit=="glorot_uniform"):#Inicializador de los pesos
            initFunc=glorot_uniform(seed=0)#Asegura la reproducibilidad
        elif(kernelInit=="he_uniform"):
            initFunc=he_uniform(seed=0)
        else:
            raise AssertionError("Only glorot_uniform or he_uniform are accepted as weight initializers")
            
        if(paddType not in ("same","valid")):
            raise AssertionError("Only same (pad) or valid (do not pad) are accepted as padding types")
        
        #compacta codigo
        convArgs=dict(filters=nFilters, kernel_size=filterSize,
                         strides=strides, padding=paddType, 
                         kernel_initializer=initFunc)
        
        for i in range(0, blockSize):#Repite tantas veces como se especifique
            if(len(filterSize)==2):#Conv2D. Ya se ha comrpobado que filterSize sea un vector de 2
                x=Conv2D(**convArgs)(x)
            elif(len(filterSize)==3):#Conv3D Idem filterSize longitud 3
                x=Conv3D(**convArgs)(x)
            
            x=Activation(actiFunc)(x)
            
            if(batchNorm==True):
                x=BatchNormalization(axis=chanDim)(x)
            
        if(len(poolReduction)==2):#Misma idea de comprobaciones redundantes
            x=MaxPooling2D(pool_size=poolReduction)(x)
        elif(len(poolReduction)==3):
            x=MaxPooling3D(pool_size=poolReduction)(x)
        
        return(x)
    else:
        raise AssertionError("In VGG block: filterSize, poolReduction and strides length must be the same, and either 2 or 3")

def CNN_VGG(width, height, depth, classes=0, regression=False, 
            nInitFilter=32, filterSize=(3,3), strides=(1,1), initBlockSize=1,
            hiddenUnits=512, poolReduction=(2,2), batchNorm=True,
            dropout=0, kernelInit="he_uniform", actiFunc="relu", paddType="same"):
    
    #Establece formato de definicion de imagen
    if(K.image_data_format()=="channels_first"):
        if(len(filterSize)==2):
            input_layer=Input(shape=(depth, width, height))
            chanDim=1
        elif(len(filterSize)==3):
            input_layer=Input(shape=(depth, width, height,1))
            chanDim=1
    else:
        if(len(filterSize)==2):
            input_layer=Input(shape=(width, height, depth))
            chanDim=-1
        elif(len(filterSize)==3):
            input_layer=Input(shape=(width, height, depth,1))
            chanDim=-1
    
    #Toma argumentos principales del bloque
    blockArgs=dict(inputLayer=input_layer, nFilters=nInitFilter,
                   blockSize=initBlockSize, filterSize=filterSize, 
                   strides=strides, poolReduction=poolReduction, 
                   batchNorm=batchNorm,chanDim=chanDim, 
                   kernelInit=kernelInit,actiFunc=actiFunc,
                   paddType=paddType)
    
    #Bloque1. Conv=>Relu=>Norm=>Reduc
    x=VGG_Block(**blockArgs)
    x=Dropout(dropout)(x)
    
    #Bloque2. (Conv=>Relu=>Norm)x2=>Reduc
    blockArgs.update(inputLayer=x, nFilters=nInitFilter*2,
                     blockSize=initBlockSize+1)
    x=VGG_Block(**blockArgs)
    x=Dropout(dropout)(x)
    
    #Bloque3. (Conv=>Relu=>Norm)x3=>Reduc
    blockArgs.update(inputLayer=x, nFilters=nInitFilter*4,
                     blockSize=initBlockSize+2)
    x=VGG_Block(**blockArgs)
    x=Dropout(dropout)(x)
    
    #MLP Compresion=>CapaOculta=>Relu=>Norm=>Salida
    if (chanDim == 1):
       x=Flatten(data_format="channels_first")(x)
    else:
       x=Flatten(data_format="channels_last")(x)
    
    if(kernelInit=="he_uniform"):
        initFunc=he_uniform(seed=0)
    elif(kernelInit=="glorot_uniform"):
        initFunc=glorot_uniform(seed=0)
    else:
        raise AssertionError("Only he_uniform or glorot_uniform are accepted as weight initializers")
    
    x=Dense(units=hiddenUnits, activation="relu", 
            kernel_initializer=initFunc)(x)
    if(batchNorm==True):
        x=BatchNormalization(axis=chanDim)(x)
    x=Dropout(dropout)(x)
    
    if(regression==False and classes>2):
        output_layer=Dense(units=classes, activation="softmax")(x)
    elif(regression==False and classes==2):
        output_layer=Dense(units=classes, activation="sigmoid")(x)
    else:
        output_layer=Dense(units=1, activation="linear")(x)
    model=Model(inputs=input_layer, outputs=output_layer)
    return(model)
    
#%% Basado en ResNet50. Igual que el original, pero más configurable

def ResNet_IdentBlock(inputLayer, nFilters, midFilterSize, chanDim, 
                      batchNorm=True, strides=(1,1), paddIndex=2,
                      kernelInit="glorot_uniform", actiFunc="relu"):
    if(len(nFilters)!=3):
        raise AssertionError("Expected 3 filter numbers for Identity block")
        
    #Bloque al que rellenar
    if(type(paddIndex)==int):
        if(paddIndex<1 or paddIndex>3):
            raise AssertionError ("The convolution to pad in identity block must either 1,2 or 3, as there are only 3 conv layers")
    else:
        raise AssertionError("paddIndex must be an integer number")
    
    #Comprueba coherencia de los tamaños
    sizes=(len(midFilterSize), len(strides))
    if(all(s==2 for s in sizes) or all(s==3 for s in sizes)):
        
        #Inicializacion de los pesos muy importante
        if(kernelInit=="glorot_uniform"):
            initFunc=glorot_uniform(seed=0)
        elif(kernelInit=="he_uniform"):
            initFunc=he_uniform(seed=0)
        else:
            raise AssertionError("Only he_uniform or glorot_uniform are accepted as weight initializers")
        
        #Para la conexion puente 
        input_shortcut=inputLayer
        #Para iterar facilmente
        x=inputLayer
        
        #Camino normal
        for i in range(0,3):
            if(len(midFilterSize)==2):#Decide 2D o 3D segun el tamaño del filtro del medio
                x=Conv2D(filters=nFilters[i], 
                         kernel_size=(1,1) if(i!=1) else midFilterSize,
                         padding="valid" if(i!=(paddIndex-1)) else "same",#Rellena la capa especificada
                         strides=strides,#Ya se ha comprobado que tenga la misma longitud que el filtro del medio
                         kernel_initializer=initFunc)(x)
            elif(len(midFilterSize)==3):
                x=Conv3D(filters=nFilters[i], 
                         kernel_size=(1,1,1) if(i!=1) else midFilterSize,
                         padding="valid" if(i!=(paddIndex-1)) else "same",
                         strides=strides,
                         kernel_initializer=initFunc)(x)
            #Normalizacion primero por motivos de orden de las capas en este bloque
            if(batchNorm==True):
                x=BatchNormalization(axis=chanDim)(x)
            
            if(i!=2): #La ultima activacion viene DESPUES de la conexion puente
                x=Activation(actiFunc)(x)
        
        #Camino puente
        x=Add()([x, input_shortcut])
        x=Activation(actiFunc)(x)
        return(x)
    else:
        raise AssertionError("In ResNet Identity block: midFilterSize, and strides length must be the same, and either 2 or 3")

def ResNet_ConvBlock(inputLayer, nFilters, midFilterSize, chanDim, 
                      batchNorm=True, strides=(1,1), paddIndex=2,
                      dimReducFactor=2, reducIndex=1,
                      kernelInit="glorot_uniform", actiFunc="relu"):
    if(len(nFilters)!=3):
        raise AssertionError("Expected 3 filter numbers for Convolution block")
        
    #Bloque al que rellenar
    if(type(paddIndex)==int):
        if(paddIndex<1 or paddIndex>3):
            raise AssertionError ("The convolution to pad in convolution block must either 1,2 or 3, as there are only 3 conv layers")
    else:
        print(paddIndex)
        print(type(paddIndex))
        raise AssertionError("paddIndex must be an integer number")
    
    #Bloque que reducir    
    if(type(reducIndex)==int):
        if(reducIndex<1 or reducIndex>3):
            raise AssertionError("The convolution features tensor to reduce dimensions with bigger strides must be either 1,2 or 3; as there are only 3 conv layers")
    else:
        raise AssertionError("reducIndex must be an integer number")
    
    #Comprueba coherencia de los tamaños
    sizes=(len(midFilterSize), len(strides))
    if(all(s==2 for s in sizes) or all(s==3 for s in sizes)):
        
        #Inicializacion de los pesos muy importante
        if(kernelInit=="glorot_uniform"):
            initFunc=glorot_uniform(seed=0)
        elif(kernelInit=="he_uniform"):
            initFunc=he_uniform(seed=0)
        else:
            raise AssertionError("Only he_uniform or glorot_uniform are accepted as weight initializers")
        
        #Para la conexion puente
        input_shortcut=inputLayer
        #Para iterar comodamente
        x=inputLayer
        
        #Zancadas aumentadas para reducir dimensiones 
        bigStrides=[s*dimReducFactor for s in list(strides)]
        
        #Camino normal
        for i in range(0,3):
            if(len(midFilterSize)==2):
                x=Conv2D(filters=nFilters[i],
                         kernel_size=(1,1) if(i!=1) else midFilterSize,
                         strides=strides if(i!=(reducIndex-1)) else bigStrides,
                         padding="valid" if(i!=(paddIndex-1)) else "same",
                         kernel_initializer=initFunc)(x)
            elif(len(midFilterSize)==3):
                x=Conv3D(filters=nFilters[i],
                         kernel_size=(1,1,1) if(i!=1) else midFilterSize,
                         strides=strides if(i!=(reducIndex-1)) else bigStrides,
                         padding="valid" if(i!=(paddIndex-1)) else "same",
                         kernel_initializer=initFunc)(x)
            
            if(batchNorm==True):
                x=BatchNormalization(axis=chanDim)(x)
            
            if(i!=2):
                x=Activation(actiFunc)(x)
        
        #Camino de conexion puente. Lleva una convolucion con zancadas grandes
        if(len(midFilterSize)==2):
            input_shortcut=Conv2D(filters=nFilters[2], kernel_size=(1,1),
                     strides=bigStrides, padding="valid", #No se rellena
                     kernel_initializer=initFunc)(input_shortcut)
        elif(len(midFilterSize)==3):
            input_shortcut=Conv3D(filters=nFilters[2], kernel_size=(1,1,1),
                     strides=bigStrides, padding="valid",
                     kernel_initializer=initFunc)(input_shortcut)
        
        if(batchNorm==True):
            input_shortcut=BatchNormalization(axis=chanDim)(input_shortcut)
        
        x=Add()([x, input_shortcut])
        x=Activation(actiFunc)(x)
        
        return(x)
    else:
        raise AssertionError("In ResNet Convolution block: midFilterSize, and strides length must be the same, and either 2 or 3")
            
def CNN_ResNet(width, height, depth, classes=0, regression=False,
               nInitFilters=32, filterSize=(7,7), batchNorm=True, 
               poolReduction=(2,2), strides=(1,1), initPad=(3,3), 
               paddIndex=2, midFilterSize=(3,3), reducIndex=1, 
               dimReducFactor=2, kernelInit="he_uniform",
               actiFunc="relu", dropout=0):
    
    #Establece formato de definicion de imagen
    if(K.image_data_format()=="channels_first"):
        if(len(filterSize)==2):
            input_layer=Input(shape=(depth, width, height))
            chanDim=1
        elif(len(filterSize)==3):
            input_layer=Input(shape=(depth, width, height,1))
            chanDim=1
    else:
        if(len(filterSize)==2):
            input_layer=Input(shape=(width, height, depth))
            chanDim=-1
        elif(len(filterSize)==3):
            input_layer=Input(shape=(width, height, depth,1))
            chanDim=-1
        
    #Toma argumentos principales del bloque
    identArgs=dict(inputLayer=input_layer, midFilterSize=midFilterSize,
                   chanDim=chanDim, batchNorm=batchNorm, 
                   strides=strides, paddIndex=paddIndex, 
                   kernelInit=kernelInit, actiFunc=actiFunc)
    
    convArgs=dict(inputLayer=input_layer, midFilterSize=midFilterSize,
                   chanDim=chanDim, batchNorm=batchNorm, 
                   strides=strides, paddIndex=paddIndex, 
                   dimReducFactor=dimReducFactor, reducIndex=reducIndex,
                   kernelInit=kernelInit, actiFunc=actiFunc)

    #Etapa 1. Relleno con 0=>Conv(zancada grande)=>Relu=>Norm=>Reduc
    #Es necesario controlar aquí las entradas
    sizes=(len(filterSize), len(initPad), len(poolReduction), len(strides))
    if(all(s==2 for s in sizes) or all(s==3 for s in sizes)):
        if(kernelInit=="he_uniform"):
            initFunc=he_uniform(seed=0)
        elif(kernelInit=="glorot_uniform"):
            initFunc=glorot_uniform(seed=0)
        else:
            raise AssertionError("Only he_uniform or glorot_uniform are accepted as weight initializers")
        #Zancada grande
        bigStride=[s*dimReducFactor for s in strides]
        #Compacta codigo
        convLayerArgs=dict(filters=nInitFilters, kernel_size=filterSize, 
                     strides=bigStride, kernel_initializer=initFunc)
        #Rellleno con ceros => Convolucion
        if(len(filterSize)==2):
            x=ZeroPadding2D(padding=initPad)(input_layer)
            x=Conv2D(**convLayerArgs)(x)
        elif(len(filterSize)==3):
            x=ZeroPadding3D(padding=initPad)(input_layer)
            x=Conv3D(**convLayerArgs)(x)
        
        x=Activation(actiFunc)(x)
        
        if(batchNorm==True):
            x=BatchNormalization(axis=chanDim)(x)
        
        if(len(poolReduction)==2):
            x=MaxPooling2D(pool_size=poolReduction)(x)
        elif(len(poolReduction)==3):
            x=MaxPooling3D(pool_size=poolReduction)(x)
    else:
        raise AssertionError("In ResNet Main: filterSize, initPad, poolReduction, and strides length must be the same, and either 2 or 3")
    
    #Etapa2 Conv=>(Ident)x2
    convArgs['nFilters']=(nInitFilters, nInitFilters, nInitFilters*4)
    identArgs['nFilters']=(nInitFilters, nInitFilters, nInitFilters*4)
    x=ResNet_ConvBlock(**convArgs)
    for i in range(0,2):
        identArgs['inputLayer']=x
        x=ResNet_IdentBlock(**identArgs)
    x=Dropout(dropout)(x)
        
    #Etapa3 Conv=>(Ident)x3
    convArgs.update(inputLayer=x, nFilters=(nInitFilters*2, nInitFilters*2, nInitFilters*8))
    identArgs['nFilters']=(nInitFilters*2, nInitFilters*2, nInitFilters*8)
    x=ResNet_ConvBlock(**convArgs)
    for i in range(0,3):
        identArgs['inputLayer']=x
        x=ResNet_IdentBlock(**identArgs)
    x=Dropout(dropout)(x)
   
    #Etapa4 Conv=>(Ident)x5
    convArgs.update(inputLayer=x, nFilters=(nInitFilters*4, nInitFilters*4, nInitFilters*16))
    identArgs['nFilters']=(nInitFilters*4, nInitFilters*4, nInitFilters*16)
    x=ResNet_ConvBlock(**convArgs)
    for i in range(0,5):
        identArgs['inputLayer']=x
        x=ResNet_IdentBlock(**identArgs)
    x=Dropout(dropout)(x)
    
    #Etapa5 Conv=>(Ident)x2
    convArgs.update(inputLayer=x, nFilters=(nInitFilters*8, nInitFilters*8, nInitFilters*32))
    identArgs['nFilters']=(nInitFilters*8, nInitFilters*8, nInitFilters*32)
    x=ResNet_ConvBlock(**convArgs)
    for i in range(0,2):
        identArgs['inputLayer']=x
        x=ResNet_IdentBlock(**identArgs)
    x=Dropout(dropout)(x)
    
    #Etapa6 Reduc=>Compresion=>Salida
    if(len(poolReduction)==2):
        x=AveragePooling2D(pool_size=poolReduction)(x)
    elif(len(poolReduction)==3):
        x=AveragePooling3D(pool_size=poolReduction)(x)
    
    if (chanDim == 1):
       x=Flatten(data_format="channels_first")(x)
    else:
       x=Flatten(data_format="channels_last")(x)
    
    if(regression==False and classes>2):
        output_layer=Dense(units=classes, activation="softmax")(x)
    elif(regression==False and classes==2):
        output_layer=Dense(units=classes, activation="sigmoid")(x)
    else:
        output_layer=Dense(units=1, activation="linear")(x)
        
    model=Model(inputs=input_layer, outputs=output_layer)
    return(model)
#%% Basado en Inception V1 pero más configurable

def Inception_MiniBlock(layer, nFilters, filterSize, paddType, initFunc,
                        actiFunc, batchNorm, chanDim):
    #Conv=>Relu=>Norm
    #¡¡NO se hacen comprobaciones aquí!!
    
    #compacta codigo
    convArgs=dict(filters=nFilters, kernel_size=filterSize, padding=paddType,
                kernel_initializer=initFunc)
    
    if(len(filterSize)==2):
        x=Conv2D(**convArgs)(layer)
    elif(len(filterSize)==3):
        x=Conv3D(**convArgs)(layer)
        
    x=Activation(actiFunc)(x)
    
    if(batchNorm==True):
        x=BatchNormalization(axis=chanDim)(x)
    
    return(x)
    
def Inception_WideBlock(inputLayer, nFilters, filterSizes, 
                    chanDim, poolReduction, batchNorm=False, 
                    paddType="same",kernelInit="he_uniform", 
                    actiFunc="relu"):
    #Tiene que haber 6 numeros de filtros, y tienen que estar en orden de uso
    if(len(nFilters)!=6):
        raise AssertionError("In Inception block: 6 filter numbers are expected")
    #Tiene que haber dos tamaños de filtro, y tiene que estar en orden de uso
    if(len(filterSizes)!=2):
        raise AssertionError("In Inception block: 2 filter sizes are expected")
    #Comprueba coherencia de tamaños
    sizes=(len(filterSizes[0]), len(filterSizes[1]), len(poolReduction))
    if(all(s==2 for s in sizes) or all(s==3 for s in sizes)):
        
        if(kernelInit=="he_uniform"):
            initFunc=he_uniform(seed=0)
        elif(kernelInit=="glorot_uniform"):
            initFunc=glorot_uniform(seed=0)
        else:
            raise AssertionError("Only he_uniform or glorot_uniform are accepted as weight initializers")
        
        
        miniblockArgs=dict(layer=inputLayer,paddType=paddType,
                           initFunc=initFunc,actiFunc=actiFunc,
                           batchNorm=batchNorm,chanDim=chanDim,
                           nFilters=nFilters[0],
                           filterSize=(1,1) if(len(filterSizes[0])==2) else (1,1,1))
    
        #De menor a mayor tamaño de kernel
        #Rama1 Conv 1x1
        branch1=Inception_MiniBlock(**miniblockArgs)
        
        #Rama2 Conv1x1=>Conv f1xf1
        miniblockArgs['nFilters']=nFilters[1]
        branch2=Inception_MiniBlock(**miniblockArgs)
        miniblockArgs.update(nFilters=nFilters[2],filterSize=filterSizes[0],
                             layer=branch2)
        branch2=Inception_MiniBlock(**miniblockArgs)
        
        #Rama3 Conv1x1=>Convf2xf2
        miniblockArgs.update(nFilters=nFilters[3], layer=inputLayer,
                             filterSize=(1,1) if(len(filterSizes[0])==2) else (1,1,1))
        branch3=Inception_MiniBlock(**miniblockArgs)
        miniblockArgs.update(nFilters=nFilters[4], layer=branch3,
                             filterSize=filterSizes[1])
        branch3=Inception_MiniBlock(**miniblockArgs)
        
        #Rama4 Reduc=>Conv1x1
        if(len(poolReduction)==2):
            branch4=MaxPooling2D(pool_size=poolReduction, padding="same",
                                 strides=(1,1))(inputLayer)
        elif(len(poolReduction)==3):
            branch4=MaxPooling3D(pool_size=poolReduction, padding="same",
                                 strides=(1,1,1))(inputLayer)
        
        miniblockArgs.update(nFilters=nFilters[5], layer=branch4,
                             filterSize=(1,1) if(len(filterSizes[0])==2) else (1,1,1))
        branch4=Inception_MiniBlock(**miniblockArgs)
        
        outputLayer=concatenate([branch1, branch2, branch3, branch4], axis=chanDim)
        
        return(outputLayer)
    else:
       raise AssertionError("In Inception WideBlock filterSizes, and poolReduction length must be the same, and either 2 or 3")

def Inception_AuxBlock(inputLayer, nFilters, poolReduction,
                       strides, hiddenUnits, kernelInit, paddType, chanDim,
                       actiFunc, outputLayerName, classes=0, regression=False, batchNorm=False,
                       dropout=0):
    #Comprueba coherencia de tamaños
    sizes=(len(poolReduction), len(strides))
    if(all(s==2 for s in sizes) or all(s==3 for s in sizes)):
        
        if(kernelInit=="he_uniform"):
            initFunc=he_uniform(seed=0)
        elif(kernelInit=="glorot_uniform"):
            initFunc=glorot_uniform(seed=0)
        else:
            raise AssertionError("Only he_uniform or glorot_uniform are accepted as weight initializers")
        #Reduc=>Conv=>Relu=>Norm
        
        #compacta codigo
        convArgs=dict(filters=nFilters, kernel_size=(1,1), 
                     kernel_initializer=initFunc, padding=paddType)
        
        
        if(len(poolReduction)==2):
            x=AveragePooling2D(pool_size=poolReduction, strides=strides)(inputLayer)
            x=Conv2D(**convArgs)(x)
        elif(len(poolReduction)==3):
            x=AveragePooling3D(pool_size=poolReduction, strides=strides)(inputLayer)
            convArgs['kernel_size']=(1,1,1)
            x=Conv3D(**convArgs)(x)
       
        x=Activation(actiFunc)(x)
        
        if(batchNorm==True):
            x=BatchNormalization(axis=chanDim)(x)
        
        #Compr=>CapaOculta=>Salida
        if(chanDim==1):
            x=Flatten(data_format="channels_first")(x)
        else:
            x=Flatten(data_format="channels_last")(x)
        
        x=Dense(units=hiddenUnits, kernel_initializer=initFunc, 
                activation=actiFunc)(x)
        x=Dropout(dropout)(x)
        if(regression==False and classes>2):
            outputLayer=Dense(units=classes, activation="softmax",
                              name=outputLayerName)(x)
        elif(regression==False and classes==2):
             outputLayer=Dense(units=classes, activation="sigmoid", 
                               name=outputLayerName)(x)
        else:
            outputLayer=Dense(units=1, activation="linear", 
                              name=outputLayerName)(x)
        return(outputLayer)
    else:
        raise AssertionError("In Inception AuxBlock poolReduction and stride length must be the same, and either 2 or 3")
        
def CNN_Inception(width, height, depth, classes=0, regression=False,
               customFilters=None, wideBlockFilterSizes=((3,3),(5,5)),
               batchNorm=True, poolReductions=((3,3),(5,5),(7,7)), 
               wideBlockPoolReduc=(3,3), auxStrides=(1,1), 
               initFilterSizes=((7,7),(1,1),(3,3)),  hiddenUnits=1024, 
               kernelInit="he_uniform", actiFunc="relu", 
               paddType="same", dropout=0):
    
    #Establece formato de definicion de imagen
    if(K.image_data_format()=="channels_first"):
        if(len(wideBlockFilterSizes[0])==2):
            input_layer=Input(shape=(depth, width, height))
            chanDim=1
        elif(len(wideBlockFilterSizes[0])==3):
            input_layer=Input(shape=(depth, width, height,1))
            chanDim=1
    else:
        if(len(wideBlockFilterSizes[0])==2):
            input_layer=Input(shape=(width, height, depth))
            chanDim=-1
        elif(len(wideBlockFilterSizes[0])==3):
            input_layer=Input(shape=(width, height, depth,1))
            chanDim=-1
    
    #Controla coherencia de tamaños
    if(len(poolReductions)!=3):
        raise AssertionError("In Inception Main: 3 pool reductions are expected, for straight path, aux outputs and final pooling")
    aux_error=0 #Por comodidad controla los errores con un codigo
    sizes=(len(poolReductions[0]), len(poolReductions[1]), 
           len(poolReductions[2]))
    if(all(s==2 for s in sizes) or all(s==3 for s in sizes)):
        aux_error+=1
    
    if(len(initFilterSizes)!=3):
        raise AssertionError("In Inception Main: 3 filter sizes for initial convolutions are expected")
    sizes=(len(initFilterSizes[0]), len(initFilterSizes[1]), 
           len(initFilterSizes[2]))
    if(all(s==2 for s in sizes) or all(s==3 for s in sizes)):
        aux_error+=1
    
    if(aux_error==2):
        #Inicializacion de los pesos. Muy importante
        if(kernelInit=="he_uniform"):
            initFunc=he_uniform(seed=0)
        elif(kernelInit=="glorot_uniform"):
            initFunc=glorot_uniform(seed=0)
        else:
            raise AssertionError("Only he_uniform or glorot_uniform are accepted as weight initializers ")
        
        #Filtros de convolucion. ¿Elegidos al azar?
        if(type(customFilters)==dict):
            convFilters=customFilters
        elif(customFilters==None): 
            convFilters=dict(stem=(64,64,192),
                         incept1=(64,96,128,16,32,32),
                         incept2=(128,128,192,32,96,64),
                         incept3=(192,96,208,16,48,64),
                         incept4=(160,112,224,24,64,64),
                         incept5=(128,128,256,24,64,64),
                         incept6=(112,144,288,32,64,64),
                         incept7=(256,160,320,32,128,128),
                         incept8=(256,160,320,32,128,128),
                         incept9=(384,192,384,48,128,128),
                         aux=128)
        else:
            raise AssertionError("In Inception Main: Custom filters must be a dictionary with 11 keys: stem, incept1..9 and aux")
        
        #Toma argumentos para minibloque, para la etapa 1
        miniblockArgs=dict(layer=input_layer, 
                           nFilters=convFilters['stem'][0], 
                           filterSize=initFilterSizes[0],
                           paddType=paddType, initFunc=initFunc,
                           actiFunc=actiFunc, batchNorm=batchNorm,
                           chanDim=chanDim)
        
        #Etapa 1: Conv7x7=>Reduc3x3=>Conv1x1=>Conv3x3=>Reduc3x3
        x=Inception_MiniBlock(**miniblockArgs)
        
        if(len(poolReductions[0])==2): #La reduccion tiene el tamaño 1, del camino recto
            x=MaxPooling2D(pool_size=poolReductions[0])(x)
        elif(len(poolReductions[0])==3):
            x=MaxPooling3D(pool_size=poolReductions[0])(x)
        
        for i in range(1,3):#Compacta
            #Se cambian el numero y tamaño de los filtros iniciales
            miniblockArgs.update(layer=x,nFilters=convFilters['stem'][i],
                             filterSize=initFilterSizes[i])
            x=Inception_MiniBlock(**miniblockArgs)
        
        if(len(poolReductions[0])==2):#Sigeu siendo camino recto
            x=MaxPooling2D(pool_size=poolReductions[0])(x)
        elif(len(poolReductions[0])==3):
            x=MaxPooling3D(pool_size=poolReductions[0])(x)
        
        #Etapa 2: Inceptionx2=>Reduc
        #Toma argumentos para bloques anchos
        wideBlockArgs=dict(inputLayer=x, 
                        nFilters=convFilters['incept1'],
                        filterSizes=wideBlockFilterSizes, 
                        chanDim=chanDim, poolReduction=wideBlockPoolReduc,
                        batchNorm=batchNorm, paddType=paddType, 
                        kernelInit=kernelInit, actiFunc=actiFunc)
        x=Inception_WideBlock(**wideBlockArgs)
        
        #Solo cambian la capa de entrada y el numero de filtros
        wideBlockArgs.update(inputLayer=x, nFilters=convFilters['incept2'])
        x=Inception_WideBlock(**wideBlockArgs)
        
        if(len(poolReductions[0])==2):#Sigue siendo camino recto
            x=MaxPooling2D(pool_size=poolReductions[0])(x)
        elif(len(poolReductions[0])==3):
            x=MaxPooling3D(pool_size=poolReductions[0])(x)
        
        #Etapa 3 Incep=>Auxiliar;(Incep)x3=>Auxiliar;Incep=>Reduc
        wideBlockArgs.update(inputLayer=x, nFilters=convFilters['incept3'])
        x=Inception_WideBlock(**wideBlockArgs)
        
        #Los caminos auxiliares llevan la reduccion 2
        auxBlockArgs=dict(inputLayer=x, nFilters=convFilters['aux'],
                          poolReduction=poolReductions[1],
                          strides=auxStrides, hiddenUnits=hiddenUnits, 
                          kernelInit=kernelInit, paddType=paddType, 
                          chanDim=chanDim, actiFunc=actiFunc,
                          classes=classes, regression=regression,
                          batchNorm=batchNorm, dropout=dropout, 
                          outputLayerName="AuxOuput1")
        output_layer1=Inception_AuxBlock(**auxBlockArgs)
        
        #Para compactar los tres incept seguidos
        inceptKeys=('incept4', 'incept5', 'incept6')
        for i in range(0,3):
            wideBlockArgs.update(inputLayer=x,
                                 nFilters=convFilters[inceptKeys[i]])
            x=Inception_WideBlock(**wideBlockArgs)
        
        #Segundo auxiliar. Solo cambia la capa de entrada y el nombre
        auxBlockArgs.update(inputLayer=x, outputLayerName="AuxOutput2")
        output_layer2=Inception_AuxBlock(**auxBlockArgs)
        
        wideBlockArgs.update(inputLayer=x, nFilters=convFilters['incept7'])
        x=Inception_WideBlock(**wideBlockArgs)
        
        if(len(poolReductions[0])==2):#Sigue siendo camino recto
            x=MaxPooling2D(pool_size=poolReductions[0])(x)
        elif(len(poolReductions[0])==3):
            x=MaxPooling3D(pool_size=poolReductions[0])(x)
        x=Dropout(dropout)(x)    
        
        #Etapa 4 (Incep)x2=>Reduc=>Compr=>CapaOculta=>Salida
        inceptKeys=('incept8', 'incept9')#compacta
        for i in range(0,2):
            wideBlockArgs.update(inputLayer=x,
                                 nFilters=convFilters[inceptKeys[i]])
            x=Inception_WideBlock(**wideBlockArgs)
        
        #La reduccion final lleva tamaño 3
        if(len(poolReductions[2])==2):
            x=AveragePooling2D(pool_size=poolReductions[2])(x)
        elif(len(poolReductions[2])==3):
            x=MaxPooling3D(pool_size=poolReductions[2])(x)
        
        #Compresion
        if(chanDim==1):
            x=Flatten(data_format="channels_first")(x)
        else:
            x=Flatten(data_format="channels_last")(x)
        
        #MLP
        x=Dense(units=hiddenUnits, activation=actiFunc,
                kernel_initializer=initFunc)(x)
        x=Dropout(dropout)(x)
        if(regression==False and classes>2):
            output_layer3=Dense(units=classes,activation="softmax", name="MainOutput")(x)
        elif(regression==False and classes==2):
            output_layer3=Dense(units=classes,activation="sigmoid", name="MainOutput")(x)
        else:
            output_layer3=Dense(units=1, activation="linear", name="MainOutput")(x)
        
        output_layer=[output_layer1,output_layer2,
                                 output_layer3]
        #MUY IMPORTANTE, PONERLE PESOS 1 0.3 0.3 AL COMPILAR
        #PONER LA MISMA FUNCION DE ERROR A LAS TRES SALIDAS
        model=Model(inputs=input_layer, outputs=output_layer)
        return(model)
    elif(aux_error==1):
        raise AssertionError("In Inception Main poolReductions length must be the same, and either 2 or 3")
    else:
        raise AssertionError("In Inception Main initFilterSizes length must be the same, and either 2 or 3")

#%% Basado en UNet pero más configurable                  
def UNet_ConvBlock(inputLayer, nFilters, filterSize, nLayers,
                   chanDim, batchNorm=True, maxPool=True,
                   poolReduction=(2,2), actiFunc="relu",
                   paddType="same", kernelInit="he_uniform"):
    if(kernelInit=="he_uniform"):
        initFunc=he_uniform(seed=0)
    elif(kernelInit=="glorot_uniform"):
        initFunc=glorot_uniform(seed=0)
    else:
        raise AssertionError("Only he_uniform or glorot_uniform are accepted as weight initializers")
    
    #compacta codigo
    convArgs=dict(filters=nFilters, kernel_size=filterSize,
                  padding=paddType, kernel_initializer=initFunc)
    
    for i in range(0, (nLayers-1)):
        if(len(filterSize)==2):
            conv=Conv2D(**convArgs)(inputLayer)
        elif(len(filterSize)==3):
            conv=Conv3D(**convArgs)(inputLayer)
        else:
            raise AssertionError("In UNet ConvBlock: the filter size must be either 2 or 3")
        
        conv=Activation(actiFunc)(conv)
        
        if(batchNorm==True):
            conv=BatchNormalization(axis=chanDim)(conv)
        
    if(maxPool==True):
        if(len(poolReduction)!=len(filterSize)):
            raise AssertionError ("In UNet ConvBlock: the filterSize and poolReduction lengths must be the same")
        
        if(len(poolReduction)==2):
            pool=MaxPooling2D(pool_size=poolReduction)(conv)
        elif(len(poolReduction)==3):
            pool=MaxPooling3D(pool_size=poolReduction)(conv)
        else:
            raise AssertionError("In UNet ConvBlock: the poolReduction size must be either 2 or 3")
        return(conv, pool)
    else:
        return(conv)

def CNN_UNet(width, height, depth, classes, initFilters=64, filterSize=(3,3),
             nLayersinBlock=2, actiFunc="relu", batchNorm=True, poolVariation=(2,2),
             kernelInit="he_uniform", paddType="same", dropout=0):
    
    #Establece formato de definicion de imagen
    if(K.image_data_format()=="channels_first"):
        input_layer=Input(shape=(depth, width, height))
        chanDim=1
    else:
        input_layer=Input(shape=(width, height, depth))
        chanDim=-1
    
    blockArgs=dict(inputLayer=input_layer, nFilters=initFilters, 
                   filterSize=filterSize, nLayers=nLayersinBlock,
                   chanDim=chanDim, batchNorm=batchNorm, maxPool=True,
                   poolReduction=poolVariation, actiFunc=actiFunc,
                   paddType=paddType, kernelInit=kernelInit)
    
    #Etapa 1: 5 bloques de convolucion y reduccion
    (conv1, pool1)=UNet_ConvBlock(**blockArgs)
    conv1=Dropout(dropout)(conv1)
        
    blockArgs.update(inputLayer=pool1, nFilters=initFilters*2)
    (conv2, pool2)=UNet_ConvBlock(**blockArgs)
    conv2=Dropout(dropout)(conv2)
        
    blockArgs.update(inputLayer=pool2, nFilters=initFilters*4)
    (conv3, pool3)=UNet_ConvBlock(**blockArgs)
    conv3=Dropout(dropout)(conv3)
        
    blockArgs.update(inputLayer=pool3, nFilters=initFilters*8)
    (conv4, pool4)=UNet_ConvBlock(**blockArgs)
    conv4=Dropout(dropout)(conv4)
    
    blockArgs.update(inputLayer=pool4, nFilters=initFilters*16)
    (conv5, pool5)=UNet_ConvBlock(**blockArgs)
    conv5=Dropout(dropout)(conv5)
    
    #Nexo: 1 bloques de convolucion sin reduccion
    blockArgs.update(inputLayer=pool5, nFilters=initFilters*32, maxPool=False)
    conv6=UNet_ConvBlock(**blockArgs)
    conv6=Dropout(dropout)(conv6)
    
    #Etapa2: 5 bloques de deconvolucion con conexiones puente
    up1=UpSampling2D(size=poolVariation)(conv6)
    merge1=concatenate([up1,conv5], axis=chanDim)
    
    blockArgs.update(inputLayer=merge1, nFilters=initFilters*16)#maxPool=False
    conv7=UNet_ConvBlock(**blockArgs)
    conv7=Dropout(dropout)(conv7)
    
    up2=UpSampling2D(size=poolVariation)(conv7)
    merge2=concatenate([up2, conv4], axis=chanDim)
    
    blockArgs.update(inputLayer=merge2, nFilters=initFilters*8)
    conv8=UNet_ConvBlock(**blockArgs)
    conv8=Dropout(dropout)(conv8)
    
    up3=UpSampling2D(size=poolVariation)(conv8)
    merge3=concatenate([up3, conv3], axis=chanDim)
    
    blockArgs.update(inputLayer=merge3, nFilters=initFilters*4)
    conv9=UNet_ConvBlock(**blockArgs)
    conv9=Dropout(dropout)(conv9)
    
    up4=UpSampling2D(size=poolVariation)(conv9)
    merge4=concatenate([up4, conv2], axis=chanDim)
    
    blockArgs.update(inputLayer=merge4, nFilters=initFilters*2)
    conv10=UNet_ConvBlock(**blockArgs)
    conv10=Dropout(dropout)(conv10)
    
    up5=UpSampling2D(size=poolVariation)(conv10)
    merge5=concatenate([up5, conv1], axis=chanDim)
    
    blockArgs.update(inputLayer=merge5, nFilters=initFilters)
    conv11=UNet_ConvBlock(**blockArgs)
    conv11=Dropout(dropout)(conv11)
    
    #Salida
    if(classes>2):
        output_layer=Conv2D(filters=classes, kernel_size=(1,1), 
                            activation="softmax", padding="same")(conv11)
    else:
        output_layer=Conv2D(filters=classes, kernel_size=(1,1), 
                            activation="sigmoid", padding="same")(conv11)
    
    model=Model(inputs=input_layer, outputs=output_layer)
    return(model)

#%% Modelos mixtos con imágenes y covariables
def MLP_AuxBranch(nVars, hiddenNodes, classes=0, regression=False, 
                  actiFunc="relu"):
    #Una sola capa oculta
    if(hiddenNodes>=nVars):
        raise AssertionError("In MLP_AuxBranch, number of nodes in hidden layer must not exceed number of initial variables")
    input_layer=Input(shape=(nVars,))
    x=Dense(units=hiddenNodes, activation=actiFunc)(input_layer)
    if(regression==False and classes>2):
        output_layer=Dense(units=classes, activation="softmax")(x)
    elif(regression==False and classes==2):
        output_layer=Dense(units=classes, activation="sigmoid")(x)
    else:
        output_layer=Dense(units=1, activation="linear")(x)
    model=Model(inputs=input_layer, outputs=output_layer)
    return(model)

def MixedModel(CNN_model, MLP, hiddenNodes, classes=0, regression=False, 
               actiFunc="relu"):
    input_layer=concatenate([CNN_model.output, MLP.output])
    x=Dense(units=hiddenNodes, activation=actiFunc)(input_layer)
    if(regression==False and classes>2):
        output_layer=Dense(units=classes, activation="softmax")(x)
    elif(regression==False and classes==2):
        output_layer=Dense(units=classes, activation="sigmoid")(x)
    else:
        output_layer=Dense(units=1, activation="linear")(x)
    model=Model(inputs=[CNN_model.input, MLP.input], outputs=output_layer)
    return(model)

#%% R^2 como métrica de regresión
    
def detCoefficient(yReal, yPredicted):
    sumSquared_resid=K.sum(K.square(yReal-yPredicted), axis=-1)
    sumSquared_total=K.sum(K.square(yReal-K.mean(yReal, axis=-1)))
    R=1-(sumSquared_resid/(sumSquared_total+K.epsilon()))
    return(R)


    
    