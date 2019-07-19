### Rodrigo Rogel
### 07/06/2019
### Practical Machine Learning
### Course Project

### Project data source -> http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har

# Six young health participants were asked to perform one set of 10 repetitions of the Unilateral 
# Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A),
# throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering 
# the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

# Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes. 
# Participants were supervised by an experienced weight lifter to make sure the execution complied to the manner they were 
# supposed to simulate. The exercises were performed by six male participants aged between 20-28 years, with little weight lifting 
# experience. We made sure that all participants could easily simulate the mistakes in a safe and controlled manner by using a 
# relatively light dumbbell (1.25kg).

library(caret)
library(skimr)
library(dplyr)
library(mlbench)
# Parallel processing libraries
library(parallel)
library(doParallel)
# Parallel processing setup
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)

# Download training data
traindat <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", header = T)
# Download testing data
testdat <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", header = T)

# Return class type of each column
cls <- sapply(traindat, class)
# Return columns which contain integer values in training set
integer_traindat <- traindat %>% select(which(cls == "integer" | cls == "numeric"))
# Look at summary of dataset composed of columns of class factor
summary(traindat[,cls == "factor"])
# Return column which contain integer or numeric values in testing set
integer_testdat <- testdat %>% select(which(cls == "integer" | cls == "numeric"))
# Count number of missing values in each column in training set
na_counts_train <- sapply(integer_traindat, function(x) sum(is.na(x)))
# Count number of missing values in each column in testing set
na_counts_test <- sapply(integer_testdat, function(x) sum(is.na(x)))
# Remove columns that are missing more than 50% of values in training set
integer_traindat <- integer_traindat %>% select(which(na_counts_train < nrow(integer_traindat)/2))
# Remove columns that are missing more than 50% of values in testing set
integer_testdat <- integer_testdat %>% select(which(na_counts_test < nrow(integer_testdat)/2))
# Remove first 4 columns because they are not helpful in predicting 
integer_traindat <- integer_traindat[, -c(1:4)]
# Remove first 4 columns because they are not helpful in predicting 
integer_testdat <- integer_testdat[, -c(1:4)]
# Skim training dataset
skimmed_train <- skim_to_wide(integer_traindat)
# Skim testing dataset
skimmed_test <- skim_to_wide(integer_testdat)
# Return dataset showing number of missing values per column and the destribution of values
print(skimmed_train, n = nrow(skimmed_train))
# Return dataset showing number of missing values per column and the destribution of values
print(skimmed_test, n = nrow(skimmed_test))
# Create histograms for each attribute
for (i in 1:ncol(integer_traindat)) {
     hist(integer_traindat[,i], main = names(integer_traindat)[i])
}
# Plot response variables versus each predictor
for (i in 1:ncol(integer_traindat)) {
     plot(traindat$classe, integer_traindat[, i], main = paste(paste(i, sep = " - ", names(integer_traindat)[i]), sep = " vs. ", "Class"))
}
# Generate correlation matrix of predictor variables
cor_matrix <- cor(integer_traindat)
# Show all correlations greater than .75 and less than 1 to help avoid multicollinearity
which(abs(cor_matrix) > .75 & cor_matrix < 1)
# Distribution of class variable
cbind(freq = table(traindat[, "classe"]), percentage = prop.table(table(traindat[, "classe"])) * 100)
# Calculate the pre-process parameters from the dataset
preprocessParams <- preProcess(integer_traindat, method = c("center", "scale", "pca"))
# Transform training dataset using the parameters
transformed_train <- predict(preprocessParams, integer_traindat)
# Transform testing dataset using the parameters
transformed_test <- predict(preprocessParams, integer_testdat)
# Add classification column
transformed_train[, "classe"] <- traindat[, "classe"]
# Set seed for reproducibility
set.seed(42)
# Set tuning parameters. We use repeated cross-validation for more accurary at the expense of run-time.
train_control <- trainControl(method = "repeatedcv", 
                              number = 10, 
                              repeats = 3,
                              allowParallel = TRUE)
# Fit Random Forest Model.
# RMSE not applicable as this is a classification problem. Accuracy and Kappa are more appropriate
rfmodel <- train(classe ~ ., 
                 transformed_train,
                 method = "rf",
                 metric = "Accuracy",
                 trControl = train_control)
# We explicitly shut down the cluster
stopCluster(cluster)
# Required to force R to return to single threaded processing
registerDoSEQ()
# Predict values in the test data set
rfpredictions <- predict(rfmodel,transformed_test)