# import libraries 
library(glmnet)
library(reshape)
library(reshape2)
library(gtools)
library(ROCR)
library(factoextra)
library(stringr)
library(reshape2)
library(ggplot2)
library(forcats)
library(magic)
library(leaps)
library(MASS)
memory.limit(size = 30000)

# import data
raw_data <- read.csv("~/GitHub/Insights-Databowl/data/raw_data.csv")
head(raw_data)

#### Work with raw data to set baseline accuracy
set.seed(100)

# set necessary variables as factors
factor_ixs = c(1:3, 12, 14, 16:19, 21, 25, 26, 28:31, 33, 35:39, 41:45, 49)
for (ix in factor_ixs){
  raw_data[,ix] = as.factor(raw_data[,ix])
}

raw_data = raw_data[complete.cases(raw_data), ] # get rid of ~3000 plays with NA values
raw_data = raw_data[which(raw_data$NflId==raw_data$NflIdRusher),] # only keep observations of the rusher to maintain (at least a relative sense of) independence needed for regression.

# note that doing so did not greatly affect our accuracy, but lowered our AIC by a factor of nearly 2.

factor_ixs = c(11, 24) # set player ids as factors
for (ix in factor_ixs){
  raw_data[,ix] = as.factor(raw_data[,ix])
}

# divide data into 80% training, 20% testing
plays = raw_data[,2]

'%ni%' <- Negate('%in%')
train_plays = sample(x = plays, size = .80*length(plays), replace = F)
test_plays = plays[plays %ni% train_plays]

train_data = raw_data[which(plays %in% train_plays),]
test_data = raw_data[which(plays %in% test_plays),]

# fit basic linear regression model using raw data. note that we excluded some variables with too many factors, unstandardized variables, and really messy variables (i.e., winds entered in  too many different ways)
basic_model = lm(Yards ~ Team + X + Y + S + A + Dis + Orientation + Dir + JerseyNumber + Season + YardLine + Quarter + Down + Distance + HomeScoreBeforePlay + VisitorScoreBeforePlay + OffenseFormation + PlayDirection + DefendersInTheBox + PlayerHeight + PlayerWeight + Position + Week + StadiumType + GameWeather + Temperature + Humidity, data = train_data)

summary(basic_model)

AIC(basic_model)
BIC(basic_model)

testing = predict(basic_model, test_data) # predict yards on test data using our model
actuals_preds = data.frame(cbind(actuals = test_data$Yards, predicteds = testing)) # actual versus predicted values
summary(actuals_preds$actuals)
summary(actuals_preds$predicteds)

### model does a good job capturing the center of the data but fares poorly in attempting to identify big gains or losses.

correlation_accuracy = cor(actuals_preds)
correlation_accuracy[2]

actuals_preds1 = actuals_preds[-which(apply(actuals_preds, 1, max) == 0),]
min_max_accuracy = mean(apply(actuals_preds1, 1, min) / apply(actuals_preds1, 1, max))  

min_max_accuracy

# take 1000 samples

correlations = numeric(1000)
accuracies = numeric(1000)

for (trial in 1:1000){
  train_plays = sample(x = plays, size = .80*length(plays), replace = F)
  test_plays = plays[plays %ni% train_plays]
  train_data = raw_data[which(plays %in% train_plays),]
  test_data = raw_data[which(plays %in% test_plays),]
  
  testing = predict(basic_model, test_data) # predict yards on test data using our model
  actuals_preds = data.frame(cbind(actuals = test_data$Yards, predicteds = testing)) # actual versus predicted values
  
  correlation_accuracy = cor(actuals_preds)
  correlations[trial]=correlation_accuracy[2]
  
  actuals_preds1 = actuals_preds[-which(apply(actuals_preds, 1, max) == 0),]
  min_max_accuracy = mean(apply(actuals_preds1, 1, min) / apply(actuals_preds1, 1, max))  
  
  accuracies[trial]=min_max_accuracy
}

summary(correlations)
summary(accuracies)
sd(correlations)
sd(accuracies)


## baseline model has consistent correlation and min-max accuracy markers (standard deviation of correlations is approximately 0.012, accuracies is approximately 0.026)
## mean correlation for our baseline model is approx. 0.18 and mean min-max accuracy is approx. 0.35

