clean_data <- read.csv("~/GitHub/Insights-Databowl/data/clean_data.csv")
head(clean_data)

#### Work with clean data with engineered features
set.seed(100)

# set necessary variables as factors
factor_ixs = c(1:4, 12, 14, 16, 19, 21, 24, 26, 27, 31:40, 44, 46:56, 58, 60)
for (ix in factor_ixs){
  clean_data[,ix] = as.factor(clean_data[,ix])
}

clean_data = clean_data[complete.cases(clean_data), ] # get rid of ~3000 plays with NA values
clean_data = clean_data[which(clean_data$IsRusher=='True'),] # only keep observations of the rusher to maintain (at least a relative sense of) independence needed for regression.

# divide data into 80% training, 20% testing
plays = clean_data[,3]

'%ni%' <- Negate('%in%')
train_plays = sample(x = plays, size = .80*length(plays), replace = F)
test_plays = plays[plays %ni% train_plays]

train_data = clean_data[which(plays %in% train_plays),]
test_data = clean_data[which(plays %in% test_plays),]

# Run backward selection using the leaps library and BIC criterion
model= regsubsets(Yards ~ Team + X + Y + S + A + Dis + Orientation + Dir + JerseyNumber + Season + Quarter + GameClock + Down + Distance + HomeScoreBeforePlay + VisitorScoreBeforePlay + Formation_ACE + Formation_EMPTY + Formation_I_FORM + Formation_JUMBO + Formation_PISTOL + Formation_SHOTGUN + Formation_SINGLEBACK + Formation_WILDCAT + PlayDirection + Position + Week + StadiumType + Turf + GameWeather + Temperature + Humidity + WindSpeed + WindDirection + DefendersInTheBox + HomePossesion + Field_eq_Possession + HomeField + PlayerBMI + TimeDelta + PlayerAge + YardsLeft, data= train_data, nbest=3, really.big = T, nvmax = 100, method='backward')
smm = summary(model)
colnames(smm$outmat)[which(smm$outmat[which(smm$bic==min(smm$bic)),] == '*')] # print variables to use in model and minimize BIC

## BIC criterion selects an 8 predictor model. Yards ~ X + S + A + Distance + Position + DefendersInTheBox + Field_eq_Possession + YardsLeft 

full= lm(Yards ~ Team + X + Y + S + A + Dis + Orientation + Dir + JerseyNumber + Season + Quarter + GameClock + Down + Distance + HomeScoreBeforePlay + VisitorScoreBeforePlay + Formation_EMPTY + Formation_I_FORM + Formation_JUMBO + Formation_PISTOL + Formation_SHOTGUN + Formation_SINGLEBACK + Formation_WILDCAT + PlayDirection + Position + Week + StadiumType + Turf + GameWeather + Temperature + Humidity + WindSpeed + WindDirection + DefendersInTheBox + HomePossesion + Field_eq_Possession + HomeField + PlayerBMI + TimeDelta + PlayerAge + YardsLeft, data= train_data)
stp = step(full, data=train_data, direction= "backward") # minimize AIC

## AIC criterion selects the model Yards ~ X + S + A + Dir + JerseyNumber + Season + GameClock + Distance + VisitorScoreBeforePlay + Formation_I_FORM + Formation_JUMBO + Formation_SHOTGUN + Formation_SINGLEBACK + PlayDirection + Position + Temperature + Humidity + WindSpeed + DefendersInTheBox + Field_eq_Possession + YardsLeft.

# Let's see if we can drop any predictor variables from the larger model:
AIC.lm = lm(Yards ~ X + S + A + Dir + JerseyNumber + Season + GameClock + Distance + VisitorScoreBeforePlay + Formation_I_FORM + Formation_JUMBO + Formation_SHOTGUN + Formation_SINGLEBACK + PlayDirection + Position + Temperature + Humidity + WindSpeed + DefendersInTheBox + Field_eq_Possession + YardsLeft, data= train_data)
BIC.lm = lm(Yards ~ X + S + A + Distance + Position + DefendersInTheBox + Field_eq_Possession + YardsLeft, data = train_data)
summary(AIC.lm)
summary(BIC.lm)


### We will use the 8-predictor model selected using BIC criterion, since the $R^2$ values are very similar and the 8-predictor model is much simpler.


# make residual plot show uniform scatter and satisfy homoskedastic error -- we log transform the Yards:
trans.BIC.lm = lm(log(Yards+15) ~ X + S + A + Distance + Position + DefendersInTheBox + Field_eq_Possession + YardsLeft, data = train_data)

#add polynomial terms:
xsq= (train_data$X)^2
ssq= (train_data$S)^2
dissq= (train_data$A)^2
defsq= (train_data$DefendersInTheBox)^2 
ydsleftsq= (train_data$YardsLeft)^2
poly.trans.BIC.lm = lm(log(Yards+15) ~ X + S + A + Distance + Position + DefendersInTheBox + Field_eq_Possession + YardsLeft + xsq + ssq + dissq + defsq + ydsleftsq, data = train_data)
summary(poly.trans.BIC.lm)

# Some of the polynomial terms are signifcant when all of the squared terms are added to the model. I will now conduct the simultaneous test.
anova(trans.BIC.lm, poly.trans.BIC.lm) 


# I will make a model with only xsq and dissq as these have the most significant p values.
less.poly.trans.BIC.lm = lm(log(Yards+15) ~ X + S + A + Distance + Position + DefendersInTheBox + Field_eq_Possession + YardsLeft + xsq + dissq, data = train_data)
summary(less.poly.trans.BIC.lm)


# Let's confirm the other quadratic terms can be dropped:
anova(poly.trans.BIC.lm)
ts = (0.17+0.04+0.12)/3/0.0584
1-(pf(ts,3,17923))

# Yes, we can drop ssq, defsq, and ydsleftsq. Let's see how significant our predictors are now:
anova(less.poly.trans.BIC.lm)

# Good. Cubic terms?
xcub= (train_data$X)^3
discub= (train_data$A)^3
cub.trans.BIC.lm = lm(log(Yards+15) ~ X + S + A + Distance + Position + DefendersInTheBox + Field_eq_Possession + YardsLeft + xsq + dissq + xcub + discub, data = train_data)
summary(cub.trans.BIC.lm)

# Good. Quartic terms?
xquar= (train_data$X)^4
disquar= (train_data$A)^4
quar.trans.BIC.lm = lm(log(Yards+15) ~ X + S + A + Distance + Position + DefendersInTheBox + Field_eq_Possession + YardsLeft + xsq + dissq + xcub + discub + xquar + disquar, data = train_data)
summary(quar.trans.BIC.lm)

# We can go up to quartic terms for x and eighth-degree terms for dis with them showing signficance.
disfive= (train_data$A)^5
dissix= (train_data$A)^6
disseven= (train_data$A)^7
diseight= (train_data$A)^8
eight.trans.BIC.lm = lm(log(Yards+15) ~ X + S + A + Distance + Position + DefendersInTheBox + Field_eq_Possession + YardsLeft + xsq + dissq + xcub + discub + xquar + disquar + disfive + dissix + disseven + diseight, data = train_data)
final.poly.trans.BIC.lm = eight.trans.BIC.lm
summary(final.poly.trans.BIC.lm)


# Let's now add interaction terms (fitted manually, but too long to demonstrate):
interaction.trans.BIC.lm = lm(log(Yards+15) ~ X + S + A + Distance + Position + DefendersInTheBox + Field_eq_Possession + YardsLeft + xsq + dissq + xcub + discub + xquar + disquar + disfive + dissix + disseven + diseight + X:A + X:YardsLeft + X:dissq + X:discub + X:disquar + X:disfive + X:dissix + X:disseven + X:diseight + S:DefendersInTheBox + + A:xsq + A:xcub + A:xquar + Distance:YardsLeft, data = train_data)
interaction.trans.BIC.lm = lm(log(Yards+15) ~ X + S + A + Distance + Position + DefendersInTheBox + YardsLeft + I(X^2) + I(X^3) + I(X^4) + X:A + X:YardsLeft + + S:DefendersInTheBox + A:I(X^2) + A:I(X^3) + A:I(X^4) + Distance:YardsLeft, data = train_data)
summary(interaction.trans.BIC.lm)
sort(cooks.distance(interaction.trans.BIC.lm), decreasing = T)[1:10]


# Now, lets take out the outliers and repeat our procedure:
outliers = paste0(names(sort(cooks.distance(interaction.trans.BIC.lm), decreasing = T)[1:5]))
ixs = numeric(5)
for (ix in 1:5){
ixs[ix] = which(rownames(train_data)==outliers[ix])
}
train_data_new = train_data[-ixs,]
xsq= (train_data_new$X)^2
ssq= (train_data_new$S)^2
dissq= (train_data_new$A)^2
defsq= (train_data_new$DefendersInTheBox)^2 
ydsleftsq= (train_data_new$YardsLeft)^2
xcub= (train_data_new$X)^3
discub= (train_data_new$A)^3
xquar= (train_data_new$X)^4
disquar= (train_data_new$A)^4
disfive= (train_data_new$A)^5
dissix= (train_data_new$A)^6
disseven= (train_data_new$A)^7
diseight= (train_data_new$A)^8
interaction.trans.BIC.lm = lm(log(Yards+15) ~ X + S + A + Distance + Position + DefendersInTheBox + Field_eq_Possession + YardsLeft + I(X^2) + I(X^3) + I(X^4) + Distance^2 + X:A + X:YardsLeft + S:DefendersInTheBox + A:I(X^2) + A:I(X^3) + A:I(X^4) + Distance:YardsLeft, data = train_data_new)
summary(interaction.trans.BIC.lm)

#interaction.trans.BIC.lm.new = lm(log(Yards+15) ~ X + S + A + Distance + Position + DefendersInTheBox + Field_eq_Possession + YardsLeft + xsq + xcub +xquar+ dissq + X:A + X:YardsLeft + S:DefendersInTheBox + A:xsq + A:xcub + A:xquar + Distance:YardsLeft, data = train_data_new)
#summary(interaction.trans.BIC.lm.new)

AIC(interaction.trans.BIC.lm) ### lowered substantially from baseline
BIC(interaction.trans.BIC.lm)
testing = predict(interaction.trans.BIC.lm, test_data) # predict yards on test data using our model
actuals_preds = data.frame(cbind(actuals = test_data$Yards, predicteds = testing)) # actual versus predicted values
summary(actuals_preds$actuals)
summary(actuals_preds$predicteds)

## model does a good job capturing the center of the data but fares poorly in attempting to identify big gains or losses.

correlation_accuracy = cor(actuals_preds)
correlation_accuracy[2] # approx. 0.25
actuals_preds1 = actuals_preds[-which(apply(actuals_preds, 1, max) == 0),]
min_max_accuracy = mean(apply(actuals_preds, 1, min) / apply(actuals_preds, 1, max))  
min_max_accuracy

# min-max accuracy score of 0.37


# Root Mean Squared Error
rmse <- function(error)
{
  sqrt(mean(error^2))
}

# Mean Absolute Error
mae <- function(error)
{
  mean(abs(error))
}

# Calculate error
error <- actuals_preds$actuals - actuals_preds$predicteds

# invocation of functions
rmse(error)
mae(error) # approx. 3.50
