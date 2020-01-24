library (PivotalR) #deals with PostgreSQL or Pivotal databases
library (RPostgreSQL) #access PostgreSQL databases
library (DBI) #interface between R and relational DBMS
library (data.table)
library (randomForest)  # tree based model - bagging
library (nnet)  		# neural network
library (Matrix)
library (foreach) 
library (glmnet)		# linear model
library (brnn)
library (lattice)
library(ggplot2)
library (caret)
library (RRF)
library (dummies)
library (gbm)
library (xgboost)
library (LiblineaR)
library (nnls)
library(dplyr)
library(caret)
library(caretEnsemble)
library(mice)
library(doParallel)
library(car)
library(mlbench)
library(ggpubr)
library(Hmisc)
library(corrr)
library(xlsx)
library(UBL)
library(e1071)

training_data <- read.csv("Training.csv");

for(i in names(training_data))
{
  if(typeof(training_data[[i]]) == "integer")
  {
    x<-which(training_data[[i]] == 9999);
    x1 <- count(as.data.frame(x));
    if(x1>= 25000)
    {
      index = grep(i, colnames(training_data));
      training_data<-training_data[,-index];
    }
  }
}

for(i in names(training_data))
{
  total <- sum(is.na(training_data[[i]]))
  if(total>= 5000)
  {
    index = grep(i, colnames(training_data));
    training_data<-training_data[,-index];
  }
}


for(i in names(training_data))
{
  if(typeof(training_data[[i]]) == "integer" && !startsWith(i,"target"))
  {
    x<-which(training_data[[i]] == 0);
    x1 <- count(as.data.frame(x));
    if(x1>= 30000)
    {
      index = grep(i, colnames(training_data));
      training_data<-training_data[,-index];
    }
  }
}

training_data$db_industry <- NULL
training_data$idc_verticals <- NULL
training_data$db_audience <- NULL

balancedata<-NULL

#Smote encoding
balancedata <- SmoteClassif(target~., training_data, C.perc="balance", k=5, dist="HEOM", p=2)
target <- training_data$target
training_data$target <- as.factor(training_data$target)

#Random forest on training data
model_rf<-randomForest(target~.,data = training_data,importance=TRUE,ntree=100,mtry=20)

for(i in names(balancedata))
{
  if(nlevels(balancedata[[i]]) > 53)
  {
    index = grep(i, colnames(balancedata));
    balancedata<-balancedata[,-index];
  }
}

for(i in names(training_data))
{
  if(is.factor(training_data[[i]]) && nlevels(training_data[[i]])>53)
  {
    index = grep(i, colnames(training_data));
    training_data<-training_data[,-index];
  }
}

training_data$Training_final_265.target <- NULL
training_data$db_country <- NULL
training_data$db_state <- NULL
training_data$db_audience <- NULL
validation_data<- read.csv("Validation.csv", header=TRUE)
validation_data$target <- as.factor(validation_data$target)
validation_Rf <-validation_data

#Calculating confusion matrix
prediction_train_rf<-predict(model_rf,validation_Rf,type = "class")
str(validation_Rf$target)
confusionMatrix_RF <- confusionMatrix(prediction_train_rf,validation_Rf$target)
confusionMatrix_RF


#Linear regression
library(LiblineaR)
BalanceData_LogReg <-balancedata

#BalanceData_LL$target <-BalanceData_LL$Training_final_265.target
BalanceData_LogReg$Training_final_265.target <- NULL
BalanceData_LogReg$db_country <- NULL
BalanceData_LogReg$db_state <- NULL
BalanceData_LogReg$db_audience <- NULL

#remove all the factors from balancedData_Logreg
for(i in names(BalanceData_LogReg))
{
  if(is.factor(BalanceData_LogReg[[i]]))
  {
    index = grep(i, colnames(BalanceData_LogReg));
    BalanceData_LogReg<-BalanceData_LogReg[,-index];
  }
}

s<-scale(BalanceData_LogReg,center = T,scale = T)

yTrain<-balancedata$Training_final_265.target  #target


test_logReg<- read.csv("Validation.csv", header=TRUE)
test_logReg$target <- as.factor(test_logReg$target)
test_logReg_data <- test_logReg[,colnames[c(1:91)]]
test_logReg_data$target <- test_logReg$target

test_logReg_data$db_country <- NULL
test_logReg_data$db_state <- NULL
test_logReg_data$db_audience <- NULL

str(test_logReg_data)
test_logReg_data1=test_logReg_data
test_logReg_data1$target <- NULL
#TestData <-Test
xtest<-test_logReg_data1
ytest<-test_logReg_data$target

Logistic_Regression<-LiblineaR(data=s,target=yTrain,type=0,cost=0.8,cross=5)

# Find the best model with the best cost parameter via 10-fold cross-validations
tryTypes <- c(0:7)
tryCosts <- c(1000,1,0.001)
bestCost <- NA
bestAcc <- 0
bestType <- NA

for(ty in tryTypes){
  for(co in tryCosts){
    acc <- LiblineaR(data=s, target=yTrain, type=ty, cost=co, cross=5, verbose=FALSE)
    cat("Results for C=",co," : ",acc," accuracy.\n",sep="")
    if(acc>bestAcc){
      bestCost <- co
      bestAcc <- acc
      bestType <- ty
    }
  }
}

bestCost
cat("Best model type is:",bestType,"\n")
cat("Best cost is:",bestCost,"\n")
cat("Best accuracy is:",bestAcc,"\n")

str(yTrain)
m <- LiblineaR(data=s,target=yTrain,type=bestType,cost=bestCost)
s2 <- scale(xtest,attr(s,"scaled:center"),attr(s,"scaled:scale"))
p <- predict(m,s2)

res <- table(p$predictions,ytest)
confusionMatrix_LR <- confusionMatrix(p,ytest)
confusionMatrix_LR



print(res)

acc<-sum(diag(res))/sum(res)
acc



#--------------L1 Norm----------------------------------------------------------------------------
set.seed(555)
m1<-LiblineaR(data=s,target=yTrain,type=6,cost=bestCost)
p1<-predict(m1,s2)
res1 <- table(p1$predictions,ytest)
acc1<-sum(diag(res1))/sum(res1)
acc1
write.csv(training_data,file = "training_datax.csv")

View(training_data)