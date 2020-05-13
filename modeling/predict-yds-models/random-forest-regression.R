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
library(randomForest)
library(ranger)
library(caret)
library(h2o)
library(party)

# read in data
memory.limit(size = 30000)
clean_data <- read.csv("~/GitHub/Insights-Databowl/data/clean_data.csv")
clean_data = clean_data[,-1]
# sam_data <- read.csv("~/GitHub/Insights-Databowl/data/fe_data.csv")

set.seed(100) # set seed

# set necessary variables as factors
factor_ixs = c(1:4, 12, 14, 16, 19, 21, 24, 26, 27, 31:40, 44, 46:56, 58, 60)
for (ix in factor_ixs){
  clean_data[,ix] = as.factor(clean_data[,ix])
}

clean_data = clean_data[complete.cases(clean_data), ] # get rid of ~3000 plays with NA values
clean_data = clean_data[which(clean_data$IsRusher=='True'),] # only keep observations of the rusher

# divide data into 80% training, 20% testing
plays = clean_data[,3]
'%ni%' <- Negate('%in%')
train_plays = sample(x = plays, size = .80*length(plays), replace = F)
test_plays = plays[plays %ni% train_plays]
train_data = clean_data[which(plays %in% train_plays),]
test_data = clean_data[which(plays %in% test_plays),]

# feature selection from ...? improve later. for now, take from lm (stepwise backwardsreg
# minimize aic)

# run random forest model
# rf <- randomForest(Yards ~ X + S + A + Dir + JerseyNumber + Season + GameClock + Distance + VisitorScoreBeforePlay + Formation_I_FORM + Formation_JUMBO + Formation_SHOTGUN + Formation_SINGLEBACK + PlayDirection + Position + Temperature + Humidity + WindSpeed + DefendersInTheBox + Field_eq_Possession + YardsLeft, data=train)

# for now, let's just create a subset data frame with only the important features
imp_features = c('X', 'S', 'A', 'Dir', 'JerseyNumber', 'Season', 'GameClock', 'Distance', 'VisitorScoreBeforePlay', 'PlayDirection', 'Position', 'Temperature', 'Humidity', 'WindSpeed', 'DefendersInTheBox', 'Field_eq_Possession', 'YardsLeft', 'Formation_I_FORM', 'Formation_JUMBO', 'Formation_SHOTGUN', 'Formation_SINGLEBACK')

train = train_data[,which(colnames(train_data) %in% c(imp_features,'Yards'))]
test = test_data[,which(colnames(test_data) %in% c(imp_features,'Yards'))]


# run random forest model using h2o pakcage in R
h2o.init(max_mem_size = "8g")

train.h2o <- as.h2o(train)
test.h2o <- as.h2o(test)
  

# hyperparameter grid. these features have been tuned
hyper_grid.h2o <- list(
  ntrees      = seq(200, 500, by = 150),
  mtries      = seq(5, 15, by = 5),
  max_depth   = seq(20, 40, by = 5),
  min_rows    = seq(3, 7, by = 2),
  nbins       = seq(10, 30, by = 5),
  sample_rate = c(.55, .632, .75)
)

# random grid search criteria
search_criteria <- list(
  strategy = "RandomDiscrete",
  stopping_metric = "mse",
  stopping_tolerance = 0.005,
  stopping_rounds = 10,
  max_runtime_secs = 30*60
)

# build grid search 
random_grid <- h2o.grid(
  algorithm = "randomForest",
  grid_id = "rf_grid2",
  x = imp_features, 
  y = 'Yards', 
  training_frame = train.h2o,
  hyper_params = hyper_grid.h2o,
  search_criteria = search_criteria
)

# collect the results and sort by our model performance metric of choice
grid_perf2 <- h2o.getGrid(
  grid_id = "rf_grid2", 
  sort_by = "mse", 
  decreasing = FALSE
)
print(grid_perf2)


# best model
best_model_id <- grid_perf2@model_ids[[1]] # chosen by minimizing MSE
best_model <- h2o.getModel(best_model_id)
best_model_perf <- h2o.performance(model = best_model, newdata = test.h2o)
best_model@model$model_summary
# best_model@model$training_metrics
best_model_perf

### MAE and RMSE of best model
# interpret MAE: average difference between yards predicted and the actual yards
# measured in yds
# best_model@model$training_metrics@metrics$mae
best_model_perf@metrics$mae # 3.405712

# interpret RMSE: sq root of avg of squared differences between actual and predicted values
# measured in yds
# NOTE: Since errors are squared before averaged, RMSE gives a relatively high weight to 
# large errors -> RMSE more useful when large errors are particularly undesirable
# ALSO NOTE RMSE result always larger or equal to the MAE
# best_model@model$training_metrics@metrics$RMSE
best_model_perf@metrics$RMSE # penalizes large errors more (sensitive to outliers) #5.486868

# plot feature importance in random forest
# WARNING: CAN BE INACCURATE DO TO MULTICOLLINEARITY
df <- data.frame(var=best_model@model$variable_importances$variable, nums=best_model@model$variable_importances$scaled_importance)
ggplot(df, aes(x=var, y=nums, fill='blue')) + geom_bar(stat = "identity") + coord_flip() + labs(title="Feature importance in random forest model", y='Scaled Importance Score', x = 'Model Feature') + theme_classic() + theme(legend.position="none")


# prediction
pred_h2o <- predict(best_model, test.h2o)
head(pred_h2o)

pred_vec = as.numeric((as.data.frame(pred_h2o))[,1])
actuals_preds = data.frame(cbind(actuals = test$Yards, predicteds = pred_vec)) # actual versus predicted values

summary(actuals_preds$actuals)
summary(actuals_preds$predicteds)

# best model evalution
correlation_accuracy = cor(actuals_preds)
correlation_accuracy[2] #0.2933567

actuals_preds1 = actuals_preds[-which(apply(actuals_preds, 1, max) == 0),]
min_max_accuracy = mean(apply(actuals_preds1, 1, min) / apply(actuals_preds1, 1, max))  
min_max_accuracy #0.4112127(before tuning hyperparamters) -- > 0.3536511(now)



###########################################################################################3

set.seed(100)

# lets try log transforming yards
train$Yards = log(train$Yards-(min(clean_data$Yards)-1))
test$Yards = log(test$Yards-(min(clean_data$Yards)-1))

h2o.init(max_mem_size = "8g")
train.h2o <- as.h2o(train)
test.h2o <- as.h2o(test)


# hyperparameter grid. these features have been tuned
hyper_grid.h2o <- list(
  ntrees      = seq(200, 500, by = 150),
  mtries      = seq(5, 15, by = 5),
  max_depth   = seq(20, 40, by = 5),
  min_rows    = seq(3, 7, by = 2),
  nbins       = seq(10, 30, by = 5),
  sample_rate = c(.55, .632, .75)
)

# random grid search criteria
search_criteria <- list(
  strategy = "RandomDiscrete",
  stopping_metric = "mse",
  stopping_tolerance = 0.005,
  stopping_rounds = 10,
  max_runtime_secs = 30*60
)

# build grid search 
random_grid <- h2o.grid(
  algorithm = "randomForest",
  grid_id = "rf_grid2",
  x = imp_features, 
  y = 'Yards', 
  training_frame = train.h2o,
  hyper_params = hyper_grid.h2o,
  search_criteria = search_criteria
)

# collect the results and sort by our model performance metric of choice
grid_perf2 <- h2o.getGrid(
  grid_id = "rf_grid2", 
  sort_by = "mse", 
  decreasing = FALSE
)
print(grid_perf2)


# best model
best_model_id <- grid_perf2@model_ids[[1]] # chosen by minimizing MSE
best_model <- h2o.getModel(best_model_id)
best_model_perf <- h2o.performance(model = best_model, newdata = test.h2o)
best_model@model$model_summary
# best_model@model$training_metrics
best_model_perf


# plot feature importance in random forest
# WARNING: CAN BE INACCURATE DO TO MULTICOLLINEARITY
df <- data.frame(var=best_model@model$variable_importances$variable, nums=best_model@model$variable_importances$scaled_importance)
ggplot(df, aes(x=var, y=nums, fill='blue')) + geom_bar(stat = "identity") + coord_flip() + labs(title="Feature importance in random forest model", y='Scaled Importance Score', x = 'Model Feature') + theme_classic() + theme(legend.position="none")


# prediction
pred_h2o <- predict(best_model, test.h2o)
head(pred_h2o)

pred_vec = as.numeric((as.data.frame(pred_h2o))[,1])
pred_vec = exp(pred_vec)-15
test = test_data[,which(colnames(test_data) %in% c(imp_features,'Yards'))]
actuals_preds = data.frame(cbind(actuals = test$Yards, predicteds = pred_vec)) # actual versus predicted values

summary(actuals_preds$actuals)
summary(actuals_preds$predicteds)

# best model evalution
correlation_accuracy = cor(actuals_preds)
correlation_accuracy[2] # 0.297138

actuals_preds1 = actuals_preds[-which(apply(actuals_preds, 1, max) == 0),]
min_max_accuracy = mean(apply(actuals_preds1, 1, min) / apply(actuals_preds1, 1, max))  
min_max_accuracy # 0.480215


### MAE and RMSE of best model
rmse <- function(error){
  sqrt(mean(error^2))
}
mae <- function(error){
  mean(abs(error))
}

# MAE
best_model_perf@metrics$mae # 3.295917

# RMSE
best_model_perf@metrics$RMSE # 5.520828

# compared to polynomial model w same data and features included, correlation improved by ~5%,
# minmax improved > 10%, rmse dec by 0.3 yards, mae dec by 0.19 yards
