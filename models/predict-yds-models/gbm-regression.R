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
final_fe_data <- read.csv("~/final_fe_data.csv")
final_fe_data = final_fe_data[,-1]

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


# use all features
imp_features = colnames(final_fe_data)[which(colnames(final_fe_data)!='Yards')]

train = train_data[,which(colnames(train_data) %in% c(imp_features,'Yards'))]
test = test_data[,which(colnames(test_data) %in% c(imp_features,'Yards'))]


# run gbm using h2o pakcage in R
h2o.init(max_mem_size = "9g")

train.h2o <- as.h2o(train)
test.h2o <- as.h2o(test)


# create training & validation sets
split <- h2o.splitFrame(train.h2o, ratios = 0.75)
train <- split[[1]]
valid <- split[[2]]


# create hyperparameter grid
hyper_grid <- list(
  max_depth = c(3, 5),
  min_rows = c(10, 15, 20),
  learn_rate = c(0.05, 0.1),
  learn_rate_annealing = c(.99, 1),
  sample_rate = c(.5, .75, 1),
  col_sample_rate = c(.8, .9, 1)
)

# random grid search criteria
search_criteria <- list(
  strategy = "RandomDiscrete",
  stopping_metric = "mse",
  stopping_tolerance = 0.005,
  stopping_rounds = 10,
  max_runtime_secs = 60*60
)

# perform grid search 
grid <- h2o.grid(
  algorithm = "gbm",
  grid_id = "gbm_grid2",
  x = imp_features, 
  y = 'Yards', 
  training_frame = train,
  validation_frame = valid,
  hyper_params = hyper_grid,
  search_criteria = search_criteria, # add search criteria
  ntrees = 10000,
  stopping_rounds = 10,
  stopping_tolerance = 0,
  seed = 123
)

# collect the results and sort by our model performance metric of choice
grid_perf <- h2o.getGrid(
  grid_id = "gbm_grid2", 
  sort_by = "mse", 
  decreasing = FALSE
)
grid_perf


# Grab the model_id for the top model, chosen by validation error
best_model_id <- grid_perf@model_ids[[1]]
best_model <- h2o.getModel(best_model_id)

# Now let's get performance metrics on the best model
h2o.performance(model = best_model, valid = TRUE) # mae approx. 3.35



# train final model
h2o.final <- h2o.gbm(
  x = imp_features, 
  y = 'Yards', 
  training_frame = train.h2o,
  nfolds = 5,
  ntrees = 10000,
  learn_rate = 0.01,
  learn_rate_annealing = 1,
  max_depth = 3,
  min_rows = 10,
  sample_rate = 0.75,
  col_sample_rate = 1,
  stopping_rounds = 10,
  stopping_tolerance = 0,
  seed = 123
)

# model stopped after xx trees
h2o.final@parameters$ntrees

# cross validated RMSE
h2o.rmse(h2o.final, xval = TRUE)

h2o.varimp_plot(h2o.final, num_of_features = 10) # plot variable importance

