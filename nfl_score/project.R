# MAT 205 - Statistical Modeling
# Final Project
# 
# In this project, we attempt to predict the score of a football game in the
# NFL based on parameters such as the week of the season, wind speed, 
# temperature, vegas line, over under, and several other parameters which are
# averages over the last several games the team has played.
# 
# This model performed about as well as one would expect. Predicting the score
# of any sporting event is a very difficult task.
# 
# We achieved an R-squared value of about 0.16 using the model_quad, which 
# included all two-way interactions and quadratic terms.

NFL = read.csv("~/project/nfl.csv", header = TRUE)
NFL = na.omit(NFL)
View(NFL)

# Convert the week to a number (quantitative) instead of qualitative
NFL$week = as.numeric(NFL$week)

# Remove outliers that had 70 mph as wind speed
NFL = NFL[-c(2139, 4446, 6662, 8969),]

# Stepwise Regression
model0 = lm(teamScore ~ 1, data = NFL)
model1 = lm(teamScore ~ ., data = NFL)

step(model0, scope = list(lower = model0, upper = model1), direction = 'both')

# Result from stepwise regression:
step_model = lm(formula = teamScore ~ vegasLine + overUnder + windSpeed + week + oppAvgYds + oppAvgRushPerc + teamAvg3Conv + teamAvgPossTime + teamAvgNumFD + temp + oppAvg4Conv + teamAvgTO, data = NFL)
model_summary = summary(step_model)
model_summary

# All Possible Models
models = ols_all_subset(step_model)

# Find best models from all possible models
models$predictors[models$aic == min(models$aic)]
models$predictors[models$adjr == max(models$adjr)]
models$predictors[models$cp == min(models$cp)]

# Only keep the columns we want for shorthand
NFL = NFL[, c(4, 9, 10, 1, 7, 13, 19, 22, 26, 20, 8, 25, 14)]

# Calculate VIF for each variable to check if any are a linear combination of the others
VIF(lm(vegasLine ~ . - teamScore, data = NFL))
VIF(lm(overUnder ~ . - teamScore, data = NFL))
VIF(lm(windSpeed ~ . - teamScore, data = NFL))
VIF(lm(week ~ . - teamScore, data = NFL))
VIF(lm(oppAvgYds ~ . - teamScore, data = NFL))
VIF(lm(oppAvgRushPerc ~ . - teamScore, data = NFL))
VIF(lm(teamAvg3Conv ~ . - teamScore, data = NFL))
VIF(lm(teamAvgPossTime ~ . - teamScore, data = NFL))
VIF(lm(teamAvgNumFD ~ . - teamScore, data = NFL))
VIF(lm(temp ~ . - teamScore, data = NFL))
VIF(lm(oppAvg4Conv ~ . - teamScore, data = NFL))
VIF(lm(teamAvgTO ~ . - teamScore, data = NFL))
# ALL VIF values are less than 10 so we keep all of them

# Start building models
model2 = lm(teamScore ~ .^2, data = NFL) # all two way interactions
summary(model2)

model3 = lm(teamScore ~ .^3, data = NFL) # all three way interactions
summary(model3)

model4 = lm(teamScore ~ .^4, data = NFL) # all four way interactions
summary(model4)

# All two way interactions and quadratic terms
model_quad = lm(teamScore ~ .^2
                + I(vegasLine^2) + I(overUnder^2) + I(windSpeed^2) + I(week^2)
                + I(oppAvgYds^2) + I(oppAvgRushPerc^2) + I(teamAvg3Conv^2) + I(teamAvgPossTime^2) + I(teamAvgNumFD^2)
                + I(temp^2) + I(oppAvg4Conv^2) + I(teamAvgTO^2), data = NFL)
summary(model_quad)


# All two way interactions, quadratic terms, and cubic terms
model_quad_cubic = lm(teamScore ~ .^2
                + I(vegasLine^2) + I(overUnder^2) + I(windSpeed^2) + I(week^2)
                + I(oppAvgYds^2) + I(oppAvgRushPerc^2) + I(teamAvg3Conv^2) + I(teamAvgPossTime^2) + I(teamAvgNumFD^2)
                + I(temp^2) + I(oppAvg4Conv^2) + I(teamAvgTO^2)
                + I(vegasLine^3) + I(overUnder^3) + I(windSpeed^3) + I(week^3)
                + I(oppAvgYds^3) + I(oppAvgRushPerc^3) + I(teamAvg3Conv^3) + I(teamAvgPossTime^3) + I(teamAvgNumFD^3)
                + I(temp^3) + I(oppAvg4Conv^3) + I(teamAvgTO^3), data = NFL)


summary(model_quad_cubic)

# Use ANOVA to compare the models and see which we can use
anova(model4, model3) # FTR H0, use reduced (model3)
anova(model3, model2) # FTR H0, use reduced (model2)
anova(model2, step_model) # Reject H0 in favor of Ha, use two way interactions
anova(model2, model_quad) # p-value = 0.0125, use quadratic model
anova(model_quad_cubic, model_quad) # p-value = 0.082, use quadratic model

quad_summary = summary(model_quad)
quad_summary

# Ask if we need to do this for each xi variable
plot(NFL$vegasLine, quad_summary$residuals)
plot(NFL$overUnder, quad_summary$residuals)
plot(NFL$windSpeed, quad_summary$residuals) # Ask about this one, error in data?
plot(NFL$week, quad_summary$residuals)
plot(NFL$oppAvgYds, quad_summary$residuals)
plot(NFL$oppAvgRushPerc, quad_summary$residuals)
plot(NFL$teamAvg3Conv, quad_summary$residuals)
plot(NFL$teamAvgPossTime, quad_summary$residuals)
plot(NFL$teamAvgNumFD, quad_summary$residuals)
plot(NFL$temp, quad_summary$residuals) # line in the middle
plot(NFL$oppAvg4Conv, quad_summary$residuals)
plot(NFL$teamAvgTO, quad_summary$residuals)

# Take the first 100 data points and store them in their own data frame
NFL_Test = NFL[c(1:100),]
# Remove these data points from the original data frame
NFL = NFL[-c(1:100),]

# Rebuild quadratic model on the slightly reduced data
model_quad = lm(teamScore ~ .^2
                + I(vegasLine^2) + I(overUnder^2) + I(windSpeed^2) + I(week^2)
                + I(oppAvgYds^2) + I(oppAvgRushPerc^2) + I(teamAvg3Conv^2) + I(teamAvgPossTime^2) + I(teamAvgNumFD^2)
                + I(temp^2) + I(oppAvg4Conv^2) + I(teamAvgTO^2), data = NFL)
summary(model_quad)

# Find the prediction interval for each of these games
predictions = predict(model_quad, NFL_Test, interval = 'prediction', level = 0.95)

# Go through each game and check if it is in the interval or not
prediction_range_correct = 0
for(i in 1:100) {
  realScore = NFL_Test$teamScore[i]
  max = predictions[i + 200]
  min = predictions[i + 100]
  if (realScore >= min && realScore <= max) {
    prediction_range_correct = prediction_range_correct + 1
  }
}

prediction_range_correct
