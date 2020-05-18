#Read csv document into R
Seattle=read.csv("Seattle AirBNB Listings.csv",header=T)
#Delete any rows with missing rows (big dataset)
Seattle=na.omit(Seattle)
#Variables in dataset
names(Seattle)
attach(Seattle)
#Exploratory Data Analysis: price, bedrooms, bathrooms & reviews
cor(Seattle[,c(5,8,9,10)]) #correlation matrix
pairs(Seattle[,c(5,8,9,10)]) #scatterplots of variables
#Turning room_type into a categorical variable
room_type=as.factor(room_type)
contrasts(room_type)
#Multiple linear regression
Rate=lm(price~poly(reviews,2)+bedrooms+bathrooms+room_type,data=Seattle)
summary(Rate)
library(car)
vif(Rate) #multicollinearity
par(mfrow=c(2,2))
plot(Rate) #diagnostic plots: not normal distribution, so making another model
Rate2=lm(log(price)~poly(reviews,2)+bedrooms+bathrooms+room_type,data=Seattle)
summary(Rate2)
vif(Rate2)
plot(Rate2) #better diagnostic plots compared to previous model
#Ridge regression
x=model.matrix(price~reviews+bedrooms+bathrooms+room_type,Seattle)
y=Seattle$price
library(glmnet)
grid=10^seq(10,-2,length=100)
set.seed(1) #setting seed for reproducible results
train=sample(1:nrow(x),nrow(x)/2) #train set
test=(-train) #test
y.test=y[test]
ridge.mod=glmnet(x[train,],y[train],alpha=0,lambda=grid)
cv.ridge=cv.glmnet(x[train,],y[train],alpha=0) #cross-validation
par(mfrow=c(1,1))
plot(cv.ridge)
bestlam.ridge=cv.ridge$bestlam
bestlam.ridge #lowest lambda
ridge.pred=predict(ridge.mod,s=bestlam.ridge,newx=x[test,])
mean((ridge.pred-y.test)^2) #test MSE for ridge regression
out.ridge=glmnet(x,y,alpha=0)
predict(out.ridge,type="coefficients",s=bestlam.ridge) #coefficients of ridge regression
#lasso
lasso.mod=glmnet(x[train,],y[train],alpha=1,lambda=grid)
plot(lasso.mod)
set.seed(1)
cv.lasso=cv.glmnet(x[train,],y[train],alpha=1)
plot(cv.lasso)
bestlam.lasso=cv.lasso$lambda.min
bestlam.lasso
lasso.pred=predict(lasso.mod,s=bestlam.lasso,newx=x[test,])
mean((lasso.pred-y.test)^2)
out.lasso=glmnet(x,y,alpha=1,lambda=grid)
lasso.coef=predict(out.lasso,type="coefficients",s=bestlam.lasso)
lasso.coef
#Random Forest
library(randomForest)
set.seed(1)
train.rf=sample(1:nrow(Seattle),nrow(Seattle)/2)
rf.seattle=randomForest(price~reviews+bedrooms+bathrooms+room_type,data=Seattle,subset=train.rf,importance=TRUE)
seattle.test=Seattle[-train,"price"]
yhat.rf=predict(rf.seattle,newdata=Seattle[-train,])
mean((yhat.rf-seattle.test)^2)
importance(rf.seattle) #info for most importance variables in random forest
varImpPlot(rf.seattle)
#Boosting
library(gbm)
set.seed(1)
boost.seattle=gbm(price~bedrooms+bathrooms+reviews+room_type,data=Seattle[train,],distribution="gaussian",n.trees=5000,interaction.depth=4)
summary(boost.seattle) #most importance variables
yhat.boost=predict(boost.seattle,newdata=Seattle[-train,],n.trees=5000)
mean((yhat.boost-seattle.test)^2)
