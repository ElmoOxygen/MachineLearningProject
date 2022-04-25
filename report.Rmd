---
title: "Practical Machine Learning Assignment"
output: html_document
author: "Adam Ross"
---

# Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 

# Data
In this project, I will use data from accelerometers on the belt, forearm, arm, and dumbbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 

* Class A: User performed exactly according to the specification.
* Class B: User threw their elbows to the front.
* Class C: User lifted the dumbbell only halfway.
* Class D: User lowered the dumbbell only halfway.
* Class E: User threw their hips to the front.

Read more: http://groupware.les.inf.puc-rio.br/har#ixzz3s021ksx8
More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

# Prepare Workspace
To make our workspace ready to make predictions I will load the necessary libraries. I also set the seed for reproducible. 
```{r, message = FALSE}
library(caret)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(randomForest)
library(corrplot)
library(e1071)

set.seed(39021)
```

# Load Data
Now we can load our data and take a look at it.
```{r}
training <- read.csv("training.csv", na.strings = c("", "NA", "#DIV/0!"))
testing <- read.csv("testing.csv", na.strings = c("", "NA", "#DIV/0!"))
```

# Clean Data
The first seven features hold information about users, timestamps, and time windows. These won't be used to make our predictions, and so will be dropped.
```{r}
training <- training[,-c(1:7)]
testing <- testing[,-c(1:7)]
```

There are many features that are only used to store summary statistics for each time frame window, with the rest of the observations being filled with NA's. We aren't concerned with the recorded time windows, and so we won't be using these summary statistics.
```{r}
NAcount <- vector()
for (i in 1:length(names(training))) {
     NAcount[i] <- sum(is.na(training[,i]))
}
training <- training[,-which(NAcount > 1000)]
testing <- testing[,-which(NAcount > 1000)]
dim(training)[2]
```
We are left with 53 features, including our response.

The data needs to be numeric, lets make sure.
```{r}
table(sapply(training, class))
```
There is 1 character class and all other variables are numeric and can be used for prediction.

```{r}
class(training$classe)
```
Our predictor classe is a character and needs to be a factor.
```{r}
training$classe <- factor(training$classe)
```

# Data Partitioning
Now I will partition our set into training and test sets for cross-validation at a 60% split.
```{r}
part <- createDataPartition(training$classe, p = .6, list = FALSE)
trainingfit <- training[part,]
testingfit <- training[-part,]
```

# Model Fitting

My first model will use a decision tree.
```{r}
fitrpart <- rpart(classe~., data=trainingfit, method=c("class"))
fancyRpartPlot(fitrpart)
predictionrpart <- predict(fitrpart, testingfit[,-53], type="class")
confusionMatrix(testingfit$classe, predictionrpart)
```
The decision tree doesn't offer great predictions, but given closer inspection may lead to some interesting insights into our data.

Random Forest will likely give more accurate predictions.
```{r}
time <- proc.time()
(fitRF <- randomForest(classe ~ ., data=trainingfit, importance = TRUE))
proc.time()-time
```

```{r}
predictionRF <- predict(fitRF, testingfit[,-53], type="response")
confusionMatrix(testingfit$classe, predictionRF)
```
The random forest model produces some great predictions. Still, the processing time is substantial. This model will work fine for our needs, but I would like to see if the processing time could be lowered without increasing the error rate too much.

The random forest model also provides an estimation on the importance of the various features.
```{r}
varImpPlot(fitRF, cex = .7)
```

Lets take the most important features and build a model from those. I'll use the top ten selections according to the Mean Decrease Accuracy and Mean Decrease Gini estimations.
```{r}
imptop <- vector()
imptop <- unique(names(c(sort(fitRF$importance[,6], decreasing=TRUE)[1:10],
                         sort(fitRF$importance[,7], decreasing=TRUE)[1:10])))
paste(imptop, collapse = "+")
```

Now I'll train a new model using only our selected features. I've also lowered the number of trees to 150. This should speed things up quite a bit.
```{r}
time <- proc.time()
(fitRFimp <- randomForest(trainingfit$classe ~ 
                               roll_forearm+
                               roll_belt+
                               magnet_dumbbell_y+
                               yaw_belt+
                               magnet_dumbbell_z+
                               magnet_dumbbell_x+
                               pitch_belt+
                               pitch_forearm+
                               roll_dumbbell+
                               accel_dumbbell_y+
                               accel_belt_z,
                          data=trainingfit, ntree=150, importance=TRUE))
proc.time() - time
```
Processing time has been substantially lowered. The OOB is higher, but still reasonably low.
```{r}
predictionRFimp <- predict(fitRFimp, testingfit[,-53], type="response")
confusionMatrix(testingfit$classe, predictionRFimp)
```
The overall accuracy drops by about 1%. This makes the out-of-sample error approximately .0167

Though I have a more accurate model built, I do believe that the faster model is accurate enough for the small testing set I'll be applying it to.

# Results
Providing output for the predictions on the test data set.
```{r}
predictionRFimp <- predict(fitRFimp, testing[,-53])
predictionRFimp
```