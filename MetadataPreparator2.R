###Reconstruye metadatos de conjuntos aumentados###
setwd("/home/albert/AlbertData/Bioinfo/TFM/NuevoBorradorPrograma/Data")
AugData<-read.table("Aug_HipercubosReconstruidos/AugLabels.txt", header = FALSE)
#Reconstruye un dataframe con el numero de columnas original
covars<-lapply(1:nrow(AugData), function(x){
  return(unlist(strsplit(as.character(AugData$V2[x]), split = "_")))
})
AugData_remade<-do.call(rbind, covars)
AugData_remade<-as.data.frame(AugData_remade)
colnames(AugData_remade)<-colnames(data)
AugData_remade$AugImage_ID<-AugData$V1
#Marca las filas que van a cada subconjunto
cte<-as.integer(3000/106)+1
subsets<-sapply(1:nrow(AugData_remade), function(x){
  batch<-unlist(strsplit(as.character(AugData_remade$AugImage_ID[x]),
                         split = "_"))[1]
  batch<-as.numeric(sub("Batch(\\d+)", replacement = "\\1", batch, perl=TRUE))
  if(batch<=cte){
    return(1)
  }else if(batch>cte && batch <=cte*2){
    return(2)
  }else if(batch>cte*2 && batch<=cte*3){
    return(3)
  }else if(batch>cte*3 && batch<=cte*4){
    return(4)
  }else{
    return(5)
  }
})
#Extrae filas y guarda sub dataframes
AugData_remade<-AugData_remade[,c(10, 2:9)]
AugData_remade$AugImage_ID<-gsub("(.+)\\.tif", "\\1",AugData_remade$AugImage_ID)
for (s in subsets){
  subAugData_remade<-AugData_remade[which(subsets==s),]
  filename<-paste("MetadataAugDataset", s,".txt", sep = "")#Con covariables
  write.table(subAugData_remade, file=filename, sep = "\t", quote = FALSE, 
              row.names = FALSE)
  filename<-paste("MetadataAugDataset",s,"_NoCovars.txt", sep = "")#Sin covariables
  write.table(subAugData_remade[,c(1,9)], file=filename, sep = "\t", quote=FALSE,
              row.names = FALSE)
}