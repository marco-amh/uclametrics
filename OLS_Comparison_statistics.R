# Comparison stats models python vs lm in r

#-----------------------------------	
# (0) Limpia memoria y llama paquetes	
#-----------------------------------	

rm(list=ls())  	#Limpia las variables
cat("\014")	    #Limpia la consola

setwd("C://Users//marco//Desktop//Projects//")	

library(lubridate)	
library(seasonal)	
library(fpp2)	
library(xlsx)    	

#-----------------------------------	
# (1) Prepara los datos	
#-----------------------------------	

#Importa los datos con frecuencia diaria	
data <- read.csv("Data.csv", header=T);	

model_01 <-lm(y~b0+b1+b2-1, data=data)
summary(model_01)
