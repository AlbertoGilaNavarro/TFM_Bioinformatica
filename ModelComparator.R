library(ggplot2)
library(reshape2)
library(PMCMR)
library(gridExtra)
#Carga de datos####
setwd("/home/albert/AlbertData/Bioinfo/TFM/NuevoBorradorPrograma/Data/")
data_mse<-read.table("MSE.tsv", header = TRUE)
data_rsquared<-read.table("Rsquared.tsv", header=TRUE)

#Reconstruye la tabla de metadatos originales
oriMetadata<-readLines("MetadataAntirrhinum.txt")
badHeaders<-oriMetadata[1]
oriMetadata<-lapply(2:length(oriMetadata), function(x){
  values<-unlist(strsplit(oriMetadata[x], split = "\t"))
  return(values)
})
oriMetadata<-as.data.frame(do.call(rbind, oriMetadata))  
colnames(oriMetadata)<-unlist(strsplit(badHeaders, split = "\t"))
#Ajusta formatos de variables
oriMetadata$`Flor nº`<-as.character(oriMetadata$`Flor nº`)
for (i in 2:ncol(oriMetadata)){
  oriMetadata[,i]<-as.numeric(as.character(oriMetadata[,i]))
}

#Separa los datos de validacion y entrenamiento
dataMSE_train<-data_mse[,which(grepl("_train", colnames(data_mse))==TRUE)]
dataMSE_test<-data_mse[,which(grepl("_vali", colnames(data_mse))==TRUE)]

dataRsq_train<-data_rsquared[,which(grepl("_train", colnames(data_rsquared))==TRUE)]
dataRsq_test<-data_rsquared[,which(grepl("_vali", colnames(data_rsquared))==TRUE)]

####Analisis estadistico de las metricas de evaluacion final####
###MSE####
#Extrae los datos 
data_FinalMSE<-melt(dataMSE_test[40,], value.name = "MSE")
data_FinalMSE$Algorithm<-gsub("(.+)_Aug.+", "\\1", data_FinalMSE$variable, 
                              perl = TRUE)
data_FinalMSE$Dataset<-gsub(".+_(Aug\\w+)_vali$", "\\1", data_FinalMSE$variable,
                            perl=TRUE)
data_FinalMSE<-data_FinalMSE[,-1]
#Calcula RMSE a partir de MSE
data_FinalMSE$RMSE<-sqrt(data_FinalMSE$MSE)
#Desnormaliza el RMSE
data_FinalMSE$UnNorm_RMSE<-data_FinalMSE$RMSE*max(oriMetadata$`Cantidad antocianina`)


#Boxplot
colors<-topo.colors(n=4)
names(colors)<-unique(data_FinalMSE$Algorithm)
ggplot(data_FinalMSE, aes(x=Algorithm, y=UnNorm_RMSE, fill=Algorithm))+
  geom_boxplot()+
  ggtitle("Final RMSE unnormalized")+
  scale_fill_manual(name="DL_algorithms", values=colors)+
  theme_minimal()

#Normalidad
tapply(data_FinalMSE$UnNorm_RMSE, data_FinalMSE$Algorithm, function(x){#Por grupo
  return(shapiro.test(x))
})
shapiro.test(data_FinalMSE$UnNorm_RMSE)#Todos juntos
#No se cumple normalidad

#Esfericidad
data_FinalMSE_cast<-dcast(data_FinalMSE, Dataset~Algorithm, value.var = "UnNorm_RMSE")
mauchlyLM<-lm(as.matrix(data_FinalMSE_cast[,-1])~1)
mauchly.test(mauchlyLM, X=~1)
#No se cumple esfericidad

#Test de Friedman
friedman.test(UnNorm_RMSE~Algorithm | Dataset, data=data_FinalMSE)
#Todos son iguales

#Test de comparaciones multiples
posthoc.friedman.nemenyi.test(UnNorm_RMSE~Algorithm | Dataset, data=data_FinalMSE)
###Rsq####
data_FinalRsq<-melt(dataRsq_test[40,], value.name = "Rsq")
data_FinalRsq$Algorithm<-gsub("(.+)_Aug.+", "\\1", data_FinalRsq$variable, 
                              perl = TRUE)
data_FinalRsq$Dataset<-gsub(".+_(Aug\\w+)_vali$", "\\1", data_FinalRsq$variable,
                            perl=TRUE)
data_FinalRsq<-data_FinalRsq[,-1]

#Boxplot
ggplot(data_FinalRsq, aes(x=Algorithm, y=Rsq, fill=Algorithm))+
  geom_boxplot()+
  ggtitle("Final R squared")+
  scale_fill_manual(name="DL_algorithms", values=colors)+
  theme_minimal()

#Normalidad
tapply(data_FinalRsq$Rsq, data_FinalRsq$Algorithm, function(x){#Por grupo
  return(shapiro.test(x))
})
shapiro.test(data_FinalRsq$Rsq)#Todos juntos
#No se cumple normalidad

#Esfericidad(Por si se cumple normalidad)
data_FinalRsq_cast<-dcast(data_FinalRsq, Dataset~Algorithm, value.var = "Rsq")
mauchlyLM<-lm(as.matrix(data_FinalRsq_cast[,-1])~1)
mauchly.test(mauchlyLM, X=~1)
#No se cumple esfericidad

#Test de Friedman
friedman.test(Rsq~Algorithm | Dataset, data=data_FinalRsq)
#Todos son iguales

#Test de comparaciones multiples
posthoc.friedman.nemenyi.test(Rsq~Algorithm | Dataset, data=data_FinalRsq)
#Grafico de historial de metricas durante el ajuste####
#Calcula media por etapa para cada algoritmo entre todos los dataset
#MSE####
algorithms<-unique(data_FinalMSE$Algorithm)
meanMSE_T<-lapply(algorithms, function(x){
  subVector<-sapply(1:nrow(dataMSE_train), function(y){
    values<-as.numeric(dataMSE_train[y,grep(x, colnames(dataMSE_train))])
    return(mean(values))
  })
  return(subVector)
})
names(meanMSE_T)<-algorithms
meanMSE_T<-as.data.frame(do.call(cbind, meanMSE_T))
meanMSE_T$Epoch<-1:40

meanMSE_T_rsh<-melt(meanMSE_T, value.name = "MeanMSE", id.vars = "Epoch",
                    variable.name = "Algorithm")
meanMSE_T_rsh$meanRMSE<-sqrt(meanMSE_T_rsh$MeanMSE)
meanMSE_T_rsh$Subset<-rep("Train", nrow(meanMSE_T_rsh))

#Vali
meanMSE_V<-lapply(algorithms, function(x){
  subVector<-sapply(1:nrow(dataMSE_test), function(y){
    values<-as.numeric(dataMSE_test[y,grep(x, colnames(dataMSE_test))])
    return(mean(values))
  })
  return(subVector)
})
names(meanMSE_V)<-algorithms
meanMSE_V<-as.data.frame(do.call(cbind, meanMSE_V))
meanMSE_V$Epoch<-1:40

meanMSE_V_rsh<-melt(meanMSE_V, value.name = "MeanMSE", id.vars = "Epoch",
                    variable.name = "Algorithm")
meanMSE_V_rsh$meanRMSE<-sqrt(meanMSE_V_rsh$MeanMSE)
meanMSE_V_rsh$Subset<-rep("Test", nrow(meanMSE_V_rsh))

#Combina y representa
meanMSE_Ready<-rbind(meanMSE_T_rsh, meanMSE_V_rsh)
subset_colors<-c("navyblue","red4")
names(subset_colors)<-c("Train","Test")
ggplot(meanMSE_Ready, aes(x=Epoch, y=meanRMSE, color=Subset))+
  geom_line(size=0.5, aes(linetype=Subset))+
  facet_wrap(vars(Algorithm), scales = "free")+
  ggtitle("RMSE history")+
  scale_color_manual(name="Subset", values=subset_colors)+
  theme_minimal()
#Rsq####
meanRsq_T<-lapply(algorithms, function(x){
  subVector<-sapply(1:nrow(dataRsq_train), function(y){
    values<-as.numeric(dataRsq_train[y,grep(x, colnames(dataRsq_train))])
    return(mean(values))
  })
  return(subVector)
})
names(meanRsq_T)<-algorithms
meanRsq_T<-as.data.frame(do.call(cbind, meanRsq_T))
meanRsq_T$Epoch<-1:40

meanRsq_T_rsh<-melt(meanRsq_T, value.name = "MeanRsq", id.vars = "Epoch",
                    variable.name = "Algorithm")
meanRsq_T_rsh$Subset<-rep("Train", nrow(meanRsq_T_rsh))

#Vali
meanRsq_V<-lapply(algorithms, function(x){
  subVector<-sapply(1:nrow(dataRsq_test), function(y){
    values<-as.numeric(dataRsq_test[y,grep(x, colnames(dataRsq_test))])
    return(mean(values))
  })
  return(subVector)
})
names(meanRsq_V)<-algorithms
meanRsq_V<-as.data.frame(do.call(cbind, meanRsq_V))
meanRsq_V$Epoch<-1:40

meanRsq_V_rsh<-melt(meanRsq_V, value.name = "MeanRsq", id.vars = "Epoch",
                    variable.name = "Algorithm")
meanRsq_V_rsh$Subset<-rep("Test", nrow(meanRsq_V_rsh))

#Combina y representa
meanRsq_Ready<-rbind(meanRsq_T_rsh, meanRsq_V_rsh)
subset_colors<-c("navyblue","red4")
names(subset_colors)<-c("Train","Test")
ggplot(meanRsq_Ready, aes(x=Epoch, y=MeanRsq, color=Subset))+
  geom_line(size=0.5, aes(linetype=Subset))+
  facet_wrap(vars(Algorithm), scales = "free")+
  ggtitle("R squared history")+
  scale_color_manual(name="Subset", values=subset_colors)+
  theme_minimal()

#Evalua el RMSE con respecto a la variable de salida####
Ymean=mean(oriMetadata$`Cantidad antocianina`)
Ysd=sd(oriMetadata$`Cantidad antocianina`)
Yrange=max(oriMetadata$`Cantidad antocianina`)-min(oriMetadata$`Cantidad antocianina`)

for(i in 2:5){
  rmse=round(mean(data_FinalMSE_cast[,i]),3)
  ratioM=round(rmse/Ymean*100,3)
  ratioS=round(rmse/Ysd*100,3)
  ratioR=round(rmse/Yrange*100,3)
  print(paste(colnames(data_FinalMSE_cast)[i],": mean rmse=", rmse,", ",
              ratioM,"% of mean of Y ;",
              #ratioS, "% of sd of Y ;",
              ratioR, "% of range of Y ",
              sep = "" ))
}








#Graficas de Rsq con las predicciones de los datos originales####
#Carga los ficheros
fileNames<-list.files("Predictions", recursive = TRUE, pattern = ".txt")
predFiles<-lapply(fileNames, function(x){
  path<-paste(getwd(),"Predictions",x, sep = "/")
  df<-read.table(path, header = TRUE, sep = "\t")
  df[,1]<-gsub("(^.+)\\.tif", "\\1", df[,1], perl = TRUE) #Quita el .tif
  return(df)
})
fileNames<-strsplit(fileNames, split = "/")
names(predFiles)<-sapply(1:length(fileNames), function(x){
  name<-paste(fileNames[[x]][1], fileNames[[x]][2], sep = "_")
  return(name)
})
#Ajusta un modelo lineal con cada fichero de predicciones y haz los graficos
#Para ordenarlos de igual forma
match_obj<-match(oriMetadata$`Flor nº`, predFiles$LeNet_Covars_Model1$Image)

linearModels<-lapply(names(predFiles), function(x){
  #Ordena de igual forma
  df<-predFiles[[x]][match_obj,]
  df<-data.frame(Real=oriMetadata$`Cantidad antocianina`, 
                 Predicted=df$Prediction)
  #Ajusta el modelo
  model<-lm(Real~Predicted, data=df)
  rsq<-round(summary(model)[["adj.r.squared"]],4)
  #Graficos exploratorio
  g<-ggplot(data=df, aes(x=Predicted, y=Real))+
    geom_point(color="Blue")+
    geom_abline(slope=model$coefficients[2], 
                intercept = model$coefficients[1], color="Red")+
    annotate("text", label=paste("R^2=",rsq, sep = ""),
             x=min(df$Predicted)+0.15*diff(range(df$Predicted)),
             y=min(df$Real)+0.6*diff(range(df$Real)),
             size=4)+
    ggtitle(x)+ylab("Real")+xlab("Predicted")+
    theme(plot.title =element_text(hjust = 0.5, size=12),
          panel.background = element_rect(fill="white", colour = "white"),
          panel.grid = element_line(colour="lightgray"))
  print(g)
  #Devuelve
  l<-list(model, rsq, g)
  names(l)<-c("Model","Rsq", "Plot")
  return(l)
})
names(linearModels)<-names(predFiles)

#Crea un multiplot con el mejor modelo de cada algoritmo
#Identifica los mejores modelos
bestModels<-lapply(algorithms, function(x){
  models<-which(grepl(x,names(linearModels))==TRUE)
  rsqs<-sapply(models, function(y){#Extrae los Rsq de cada algoritmo
    return(linearModels[[y]][["Rsq"]])
  })
  bestRsq<-which(rsqs==max(rsqs))#Identifica el mejor
  return(linearModels[[models[bestRsq]]][["Plot"]])#Extrae el plot
  })

grid.arrange(grobs=bestModels, ncol=2, top="Best model of each DL algorithm")
#Graficas de Rsq con las predicciones medias de los datos originales####
meanPred<-lapply(algorithms, function(x){
  models<-which(grepl(x, names(predFiles))==TRUE)
  subdf<-lapply(models, function(y){#Junta los valores por algoritmo
    return(predFiles[[y]]$Prediction)
  })
  subdf<-as.data.frame(do.call(cbind, subdf))
  #Aplica la media por fila
  return(as.numeric(apply(subdf, 1, mean)))
})
meanPred<-as.data.frame(do.call(cbind, meanPred))
rownames(meanPred)<-predFiles[[1]]$Image
colnames(meanPred)<-algorithms
#Ordena 
meanPred<-meanPred[match_obj,]
#Ajusta un modelo con cada algoritmo y grafica
linearModels_mean<-lapply(1:ncol(meanPred), function(x){
  df<-data.frame(Real=oriMetadata$`Cantidad antocianina`, 
                 Predicted=meanPred[,x])
  #Ajusta el modelo
  model<-lm(Real~Predicted, data=df)
  rsq<-round(summary(model)[["adj.r.squared"]],4)
  #Graficos exploratorio
  g<-ggplot(data=df, aes(x=Predicted, y=Real))+
    geom_point(color="Purple")+
    geom_abline(slope=model$coefficients[2], 
                intercept = model$coefficients[1], color="Red")+
    annotate("text", label=paste("R^2=",rsq, sep = ""),
             x=min(df$Predicted)+0.15*diff(range(df$Predicted)),
             y=min(df$Real)+0.6*diff(range(df$Real)),
             size=4)+
    ggtitle(colnames(meanPred)[x])+ylab("Real")+xlab("Predicted mean")+
    theme(plot.title =element_text(hjust = 0.5, size=12),
          panel.background = element_rect(fill="white", colour = "white"),
          panel.grid = element_line(colour="lightgray"))
  print(g)
  #Devuelve
  l<-list(model, rsq, g)
  names(l)<-c("Model","Rsq", "Plot")
  return(l)
})
names(linearModels_mean)<-colnames(meanPred)
plotlist<-lapply(names(linearModels_mean), function(x){
  return(linearModels_mean[[x]][["Plot"]])
})
grid.arrange(grobs=plotlist,ncol=2, 
             top="Models mean prediction of each DL algorithm")
