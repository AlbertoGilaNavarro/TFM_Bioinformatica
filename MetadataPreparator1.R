library(readxl)
library(ggplot2)
setwd("/home/albert/AlbertData/Bioinfo/TFM/NuevoBorradorPrograma/Data")
data<-read_excel("/home/albert/AlbertData/Bioinfo/TFM/NuevoBorradorPrograma/Data/Antirrhinum_datos_hiperespectral.xlsx")
###Datos ausentes por variable####
for (i in 1:ncol(data)){
  nas<-length(which(is.na(data[,i])))
  print(paste("Column", colnames(data)[i], "has:", nas, "NAs", 
              round(nas/nrow(data),3), "%", sep=" "))
}
#Imputa la variable longitud b por la media
for (i in which(is.na(data$`Longitud b`))){
  data$`Longitud b`[i]<-mean(data$`Longitud b`, na.rm = TRUE)
}

#Idem con la variable peso
for (i in which(is.na(data$Peso))){
  data$Peso[i]<-mean(data$Peso, na.rm = TRUE)
}

###Instancias a eliminar####
#Elimnación de las flores de la 1 a la 8, no tienen imágenes
data<-data[-c(1:8),]
###Variables a eliminar####
#La variable dia no aporta nada, se elimina
data<-data[,-1]
####Variables a renombrar####
colnames(data)[2:4]<-c("Altura", "Anchura1", "Anchura2")
###Ruido de las variables####
data<-as.data.frame(data)
l<-lapply(colnames(data), function(x){
  g<-ggplot(data=data, aes(x=get(x)))+geom_density()+
    xlab(x)+ylab("ProbDensity")
  print(g)
  return(g)
})
#Hay un valor anómalo en Absorbancia 530nm y otro en Cantidad de antocianina
#Parece que la coma está mal colocada en la fila 15. Dividelo por 1000
data$`Absorbancia 530 nm`[15]<-data$`Absorbancia 530 nm`[15]/1000
data$`Cantidad antocianina`[15]<-data$`Cantidad antocianina`[15]/1000
l<-lapply(colnames(data), function(x){
  g<-ggplot(data=data, aes(x=get(x)))+geom_density()+
    xlab(x)+ylab("ProbDensity")
  print(g)
  return(g)
}) #Ahora sí
###Guarda metadatos####
dataReady<-data
florID<-gsub("(\\d+)", "f\\1ab", data$`Flor nº`)
dataReady[,1]<-florID
write.table(dataReady, file="MetadataAntirrhinum.txt", sep = "\t", quote = FALSE, row.names = FALSE)

multivar<-sapply(1:nrow(data), function(x){
  return(paste(unname(data[x,]), collapse = "_"))
})
AugData<-data.frame(Flor=florID, Multivar=multivar)
write.table(AugData, file="MetadataAntirrhinum_Aug.txt", sep = "\t", quote = FALSE, row.names = FALSE)


