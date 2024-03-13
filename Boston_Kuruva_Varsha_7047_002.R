#Individual Boston Housing Case Study
#Author: Varsha Kuruva

# Exploratory data analysis ----------------------------------------------
#Individual Boston Housing Case Study

#EDA----------------------------------------------------------------------------

library(MASS)
data(Boston)
attach(Boston)

head(Boston)
str(Boston)
# n: number of rows ; p: number of predictors
n <- dim(Boston)[1]
p <- dim(Boston)[2] - 1

summary(Boston)

#Checking for missing values
sum(is.na(Boston)) #no missing values

#Check for duplicated values
sum(duplicated(Boston)) #no duplicate values

#Train-Test split
set.seed(15128170)
sample_index <- sample(nrow(Boston),nrow(Boston)*0.80)
Boston_train <- Boston[sample_index,]
Boston_test <- Boston[-sample_index,]

# Variable Selection------------------------------------------------------------

# Checking correlations
library(corrplot)
corrplot(cor(Boston), method = "square", order = "FPC", type = "lower", diag = TRUE)

library(leaps)
var_fit <- regsubsets(medv~., data = Boston_train, nbest = 1, nvmax = 13)
summary(var_fit)



#Fit a linear regression model using only the selected variables
Boston_lm <- lm(medv ~ chas+nox+rm+dis+ptratio+black+lstat, data = Boston_train)
summary(Boston_lm)

#Use predict() to make predictions on the train/test set
yhat.train <- predict(Boston_lm, newdata = Boston_train) 
yhat.test <- predict(Boston_lm, newdata = Boston_test)

#ASE (in-sample)
ase <- sum((Boston_train$medv - yhat.train)^2)/(dim(Boston_train)[1])
ase #24.49
mse <- sum((Boston_train$medv - yhat.train)^2)/((dim(Boston_train)[1])-p-1)
mse #23.95946
#MSPE (out-of-sample)
mspe <- sum((Boston_test$medv - yhat.test)^2)/dim(Boston_test)[1]
mspe #20.63

# Regression tree with pruning -------------------------------------------
library(rpart)

#Fit a regression tree with Pruning
Boston_largetree <- rpart(formula = medv ~ ., data = Boston_train, cp = 0.001)

#To look at the error vs size of tree 
printcp(Boston_largetree)
prp(Boston_largetree)
#Use predict() to make predictions on the train/test set
yhat.train <- predict(Boston_largetree, newdata = Boston_train) 
yhat.test <- predict(Boston_largetree, newdata = Boston_test)

#ASE (in-sample)
ase <- sum((Boston_train$medv - yhat.train)^2)/(dim(Boston_train)[1])
#11.43869
#MSPE (out-of-sample)
mspe <- sum((Boston_test$medv - yhat.test)^2)/dim(Boston_test)[1]
#20.40240


# k-nearest neighbor (k-NN) with optimal choice of k ----------------------
Boston.norm <- Boston
train.norm <- Boston_train
test.norm <- Boston_test

#Scale data
Boston.norm[,1:p] <- scale(Boston[,1:p])

cols <- colnames(train.norm[, -(p + 1)]) #scaling only on p=13 predictors X
for (j in cols) {
  train.norm[[j]] <- (train.norm[[j]] - min(Boston_train[[j]])) / (max(Boston_train[[j]]) - min(Boston_train[[j]]))
  test.norm[[j]] <- (test.norm[[j]] - min(Boston_train[[j]])) / (max(Boston_train[[j]]) - min(Boston_train[[j]]))
}

#Fit a K-nn model
library(FNN)

Boston.knn.reg1 <- knn.reg(train = Boston.norm[, 1:p], 
                          test = Boston.norm[, 1:p], 
                          y = Boston.norm$medv, 
                          k = 2)

Boston.knn.reg2 <- knn.reg(train = train.norm[, 1:p], 
                          test = test.norm[, 1:p], 
                          y = train.norm$medv, 
                          k = 2)

#Compile the actual and predicted values
Boston_train_results <- data.frame(cbind(pred = Boston.knn.reg1$pred, actual = Boston.norm$medv))
Boston_test_results <- data.frame(cbind(pred = Boston.knn.reg2$pred, actual = Boston_test$medv))

#ASE (in-sample)
ase <- sum((Boston.norm$medv - Boston_train_results$pred)^2)/length(Boston.norm$medv)
ase
#4.8540
#MSPE (out-of-sample)
mspe <- sum((Boston_test$medv - Boston_test_results$pred)^2)/length(Boston_test$medv)
#15.0241
# Random forests -----------------------------------------------
library(randomForest)

#Fit a random forest model
Boston_rf <- randomForest(medv~., data = Boston_train, importance = TRUE)
Boston_rf

#Use predict() to make predictions on the train/test set
yhat.train <- predict(Boston_rf, newdata = Boston_train) 
yhat.test <- predict(Boston_rf, newdata = Boston_test)

#ASE (in-sample)
ase <- sum((Boston_train$medv - yhat.train)^2)/(dim(Boston_train)[1])
#2.30378
#MSPE (out-of-sample)
mspe <- sum((Boston_test$medv - yhat.test)^2)/dim(Boston_test)[1]
mspe
#6.46

# Boosting trees ----------------------------------------------------------------
library(gbm)

#Boosting for regression trees
Boston_boost <- gbm(formula = medv~., 
                    data = Boston_train, 
                    distribution = "gaussian", #Gaussian for regression tree, Bernoulli for binary classification
                    n.trees = 10000,  #choose this parameter carefully because it may results in overfitting if the number is too large
                    shrinkage = 0.01, #a tuning parameter that controls how much contribution each tree makes
                    interaction.depth = 8) #interaction.depth is how many splits of each tree we want 
summary(Boston_boost)

#Use predict() to make predictions on the train/test set
yhat.train <- predict(Boston_boost, newdata = Boston_train, n.trees = 10000) 
yhat.test <- predict(Boston_boost, newdata = Boston_test, n.trees = 10000)

#ASE (in-sample)
ase <- sum((Boston_train$medv - yhat.train)^2)/(dim(Boston_train)[1])
#0.0137
#MSPE (out-of-sample)
mspe <- sum((Boston_test$medv - yhat.test)^2)/dim(Boston_test)[1]
#5.286
# Generalized additive model (GAM) ----------------------------------------
#Fit a GAM model
library(mgcv)

Boston_gam <- gam(medv ~ s(crim)+s(indus)+s(nox)+s(rm)+s(ptratio)
                  +s(dis)+s(tax)+s(black)+s(lstat), data = Boston_train)

#Use predict() to make predictions on the train/test set
yhat.train <- predict(Boston_gam, newdata = Boston_train) 
yhat.test <- predict(Boston_gam, newdata = Boston_test)

#ASE (in-sample)
ase <- sum((Boston_train$medv - yhat.train)^2)/(dim(Boston_train)[1])
#8.3920

#MSPE (out-of-sample)
mspe <- sum((Boston_test$medv - yhat.test)^2)/dim(Boston_test)[1]
#12.9680

# Neural networks ---------------------------------------------------------
#Data scale
train.norm <- Boston_train
test.norm <- Boston_test

#Normalize all numerical variables (X&Y) to 0-1 scale, range [0,1]-standardization
cols <- colnames(train.norm[, ]) #scaling both X and Y
for (j in cols) {
  train.norm[[j]] <- (train.norm[[j]] - min(Boston_train[[j]])) / (max(Boston_train[[j]]) - min(Boston_train[[j]]))
  test.norm[[j]] <- (test.norm[[j]] - min(Boston_train[[j]])) / (max(Boston_train[[j]]) - min(Boston_train[[j]]))
}

#Fit a neural networks on (scaled) Training data 
library(neuralnet)
library(dplyr)
f <- as.formula("medv ~ .")
nn <- neuralnet(f, data = train.norm, hidden = c(5,3), linear.output = T)

yhat.train <- predict(object = nn, newdata = train.norm) 
library(dplyr)
#Recover the predicted value back to the original response scale
pr_nn <- neuralnet::compute(nn, train.norm[,1:p])

## recover the predicted value back to the original response scale ## 
pr_nn_org <- pr_nn$net.result*(max(Boston_train$medv) - min(Boston_train$medv)) + min(Boston_train$medv)
train_r <- (train.norm$medv)*(max(Boston_train$medv) - min(Boston_train$medv)) + min(Boston_train$medv)

#ASE (dividing by the number of observations n)
ase.train <- sum((train_r - pr_nn_org)^2)/nrow(train.norm)#5.45
#4.6452
#MSPE of the above neural network model for Testing data
pr_nn <-neuralnet::compute(nn, test.norm[,1:p])

#recover the predicted value back to the original response scale ## 
pr_nn_org <- pr_nn$net.result*(max(Boston_train$medv) - min(Boston_train$medv)) + min(Boston_train$medv)
test_r <- (test.norm$medv)*(max(Boston_train$medv) - min(Boston_train$medv)) + min(Boston_train$medv)

#MSPE of testing set
MSPE_nn <- sum((test_r - pr_nn_org)^2)/nrow(test.norm)
#15.2041

plot(nn)

# Figures for Boosting Model ----------------------------------------------
#Boosting for regression trees
library(gbm)

Boston_boost <- gbm(formula = medv~., 
                    data = Boston_train, 
                    distribution = "gaussian", #Gaussian for regression tree, Bernoulli for binary classification
                    n.trees = 10000,  #choose this parameter carefully because it may results in overfitting if the number is too large
                    shrinkage = 0.01, #a tuning parameter that controls how much contribution each tree makes
                    interaction.depth = 8) #interaction.depth is how many splits of each tree we want 
summary(Boston_boost)


#The fitted boosted tree also gives the relation between response and each predictor.
par(mfrow = c(1,2))
plot(Boston_boost, i = "lstat")
plot(Boston_boost, i = "rm")

#We can investigate how the testing error changes with different number of trees.
ntree <- seq(100, 10000, 100)
test.err <- rep(0, 13)

predmat <- predict(Boston_boost, newdata = Boston_test, n.trees = ntree)
err <- apply((predmat - Boston_test$medv)^2, 2, mean)
plot(ntree, err, type = 'l', col = 2, lwd = 2, xlab = "n.trees", ylab = "Test MSE")
abline(h = min(test.err), lty = 2)

