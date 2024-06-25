#___________________________________
# Loading Libraries
#___________________________________

install.packages("ggplot2")
library(ggplot2)

install.packages("magrittr")
library(magrittr)

install.packages("dplyr")
library(dplyr)

#___________________________________
# Importing Dataset
#___________________________________

setwd("/Users/charithadanda/Documents/Linear Regression Models (Stats)")
bike_data <- read.csv("Bike Sharing Demand Prediction.csv",check.names = F)

#___________________________________
# setting new column names for easy access
#___________________________________

bike_data <- setNames(bike_data, c("date","bike_count","hour","temp","humidity","wind_speed","visibility",
                                   "dew_point_temp","solar_radiation","rainfall","snowfall","seasons",
                                   "holiday","functioning_day"))

#_______________________________________
#Setting categorical values and changing categorical values to numerical values
#_______________________________________

bike_data$hour <- factor(bike_data$hour)
bike_data$seasons <- factor(bike_data$seasons)
bike_data$holiday <- factor(bike_data$holiday)
bike_data$functioning_day <- factor(bike_data$functioning_day)

#____________________________
# Studying Data
#____________________________
head(bike_data)
str(bike_data)
summary(bike_data)

#__________________
# Data Analysis
#_________________

summary(bike_data$functioning_day)

# According to the summary statistics of functioning_day, there are 295 non-functioning days in the data set. 
#These days will have zero bike counts.

fig1 <- ggplot(bike_data, aes(x = as.factor(functioning_day), y = bike_count)) + 
  geom_boxplot(fill="slateblue", alpha=0.2) + 
  ggtitle("Bike Demand by functioning day") +  
  xlab("Functioning Day") + 
  ylab("Bike Count") +
  theme(plot.title = element_text(size=14, face="bold", hjust=0.5))
fig1

#The bikes will not be available to rent during non functioning days and therefore, bike count will be zero on non functioning day as no one will be able to rent the bicycles as seen in Fig 1. 
#Therefore, non-functioning day data points are deleted from the dataset as this is not relevant to our purpose. 
#The data set is now left with 8465 data points.

# Deleting rows when it is non-functioning day
bike_data <-bike_data[!(bike_data$functioning_day=="No"),]

# removing unused columns
bike_data <- subset(bike_data, select = - c(functioning_day))

summary(bike_data)
#Summary statistics of the dataset shows general idea of how the data is without non-functioning days look like. 
#This will be further analysed with bar plots and histograms.

fig2<- ggplot(bike_data, aes(x = as.factor(hour) , y = bike_count)) + geom_boxplot(fill="slateblue", alpha=0.2) + 
  ggtitle("Hourly Bike Count Bar Plot") +  xlab("Hour") + ylab("Bike Count") +
  theme(plot.title = element_text(size=14, face="bold", hjust=0.5))
fig2

#Fig 2. Hourly Bike Count Bar Plot indicates the numbers of bikes rented by hour. The mean demand is lower than 500 bike counts for hours between 1 am 
#and 6 am. We can see huge spike in boxplot at 8 which is 8 am in the morning, and we can see slowly increasing pattern in boxplot throughout daytime 
#reaching huge peak at 18 which is 6pm. we can see bike renting starts to decrease after 18 and reached lowest at 4. All the am hours except 8 am and 7 
#am has demand of lower than 1000 bike counts.Therefore, we take am hours as low demand hours and pm hours as high demand hours.

fig3 <- ggplot(bike_data, aes(x = as.factor(seasons), y = bike_count)) + geom_boxplot(fill="slateblue", alpha=0.2) +
  ggtitle("Seasonal Bike Count Bar Plot") +  xlab("Seasons") + ylab("Bike Count") +
  theme(plot.title = element_text(size=14, face="bold", hjust=0.5))
fig3

#From Fig 3. Seasonal Bike Count Bar Plot, we can see the comparisons of bike renting by seasons. we can easily infer that during better weather 
#times such as summer, people usually prefer to cycle more and in winters bikes are rented lowest with well lower than 500 bike counts in demand. 
#This may be due to cold weather as well as snow during winter. There are a few outliers with higher density than the rest of the seasons where bike 
#demand in Winter is over 500. This may be some underlying reasons such as the day having better weather than other days in winter. In Autumn, bike 
#demand is higher than Spring though we were expecting the demand to be higher in Spring than in Autumn. 
#This may be due to having more rainy days in Spring than in Autumn.

fig4 <- ggplot(bike_data, aes(x = as.factor(holiday), y = bike_count)) + geom_boxplot(fill="slateblue", alpha=0.2) + 
  ggtitle("Holiday Bike Count Bar Plot") +  xlab("Holiday") + ylab("Bike Count") +
  theme(plot.title = element_text(size=14, face="bold", hjust=0.5))
fig4

#According to figure 4, we can determine that the demand for bike is higher on non holiday days, which means mostly our users might be renting bikes 
#for other reasons than for recreational purposes.

#_________________________________
# Correlation matrix of dataset
#________________________________

# We are performing correlation matrix to understand relationship between our variables which is explained in greater detail below.

cor_df <- data.frame(bike_data$bike_count, as.integer(bike_data$hour), bike_data$temp, bike_data$humidity, 
                     bike_data$wind_speed, bike_data$visibility, bike_data$dew_point, bike_data$solar_radiation, 
                     bike_data$rainfall, bike_data$snowfall)

cor_df <- data.frame(cor(cor_df))

cor_df[order(-cor_df[,1]), ]

library(corrplot)
library(ggplot2)
install.packages("reshape2")
library(reshape2)

# Creating the correlation matrix
cor_matrix <- cor(cor_df)

# Converting the correlation matrix to long format
cor_melted <- melt(cor_matrix)

# Plotting the heatmap with values
ggplot(cor_melted, aes(x = Var1, y = Var2, fill = value, label = round(value, 2))) +
  geom_tile(color = "white") +
  geom_text(color = "black") +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", 
                       midpoint = 0, limits = c(-1, 1), name = "Correlation") +
  labs(title = "Correlation Matrix", x = "", y = "") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(hjust = 0.5, face = "bold")) +
  coord_fixed()


#_______________________
#Histograms
#_______________________


# Defining a function to create histograms for numeric variables
multi.hist <- function(bike_data) {
  par(mfrow=c(ceiling(sqrt(ncol(bike_data))), ceiling(sqrt(ncol(bike_data)))))
  for (i in 1:ncol(bike_data)) {
    hist(bike_data[,i], main=names(bike_data)[i])
  }
  par(mfrow=c(1,1))
}

# Calling the function with your numeric data
multi.hist(bike_data[,sapply(bike_data, is.numeric)])

#Above are the histograms of continuous data to determine the shape of the variables. It can be seen that bike_count values are right-skewed. 
#There are a lot of hours with temperature below zero, which we expect these hours to have lower demand for bike rentals. The humidity histogram shows 
#most of the days. Dew point temperature shows similar distribution to temperature. Solar radiation is 0 on most hours with very few hours with extrememe 
#values. Rainfall and snowfall data is extremely skewed to the right with very few hours with high levels of rain and snow. 
#The temperature shows signs of having a bimodal distribution.

#________________________
# Modelling
#_________________________

# Model 1
#Model 1 is a multiple linear regression model predicting bike rental demand for every hour using hourly weather conditions. For this model, 
#the dataset is used as it is without changing anything. 
#Data is splitted into 80-20 to build the model.

# Data spliting
# Data is splitted to 80% train set and 20% test set. This is to minimise overfitting. Data is splitted by random selection using sample() method.

# Randomizing dataset

## 80% of the sample size
smp_size <- floor(0.80 * nrow(bike_data))

## Setting the seed to make your partition reproducible
set.seed(123)
trainIndex <- sample(seq_len(nrow(bike_data)), size = smp_size)

model1_prediction <- bike_data[ -trainIndex,]
model1 <- bike_data[ trainIndex,]

# After splitting the data, train dataset is fitted to the model.

# Regression model with all columns original data
row.names(model1) <- NULL

original_full = lm(bike_count ~ hour+temp+humidity+wind_speed+visibility+dew_point_temp+
                     solar_radiation+rainfall+snowfall+seasons+holiday, data=model1)

#Summary
summary(original_full)

# The stepwise AIC is used to calculate an optimum model for prediction.

# Possible subsets regression of original data
library(MASS)
stepReg=MASS::stepAIC(original_full, direction="both")

stepReg$anova

# Multicollinearity test
install.packages("car")
library(car)
car::vif(original_full)

# Stepwise method produced a final regression model without visibility, and windspeed. These will be removed from the final model. 
# Dew_point_temp’s GVIF value is greater than 10 and showing presence of multicollinearity. This will be removed from the model.

# Removing columns with multicollinearity and stepwise
original_final = lm(bike_count ~ hour + temp + humidity + solar_radiation + 
                      rainfall + snowfall + seasons + holiday, data=model1)

#summary
summary(original_final)

# T-value for snowfall, hour 9 and hour 16 show that they are greater than 0.05 therefore, the predictions for these variables are not statistically 
# significant. However, the final model shows that it can explain 65.87% of the dataset and it is statistically significant as p-value is less than 0.05.

# Residual Plots
par(mfrow=c(2,2))
plot(original_final , which=1:4)

# Cook’s distance shows that there is an outlier. This will be removed from the model.

# Evaluate homoscedasticity
# non-constant error variance test
ncvTest(original_final)

# P-value is less than 0.05, implying that the constant error variance assumption is violated. 
# Therefore, box-cox transformation will be performed on the model.

# Test for Autocorrelated Errors
durbinWatsonTest(original_final)

# P-value is more than 0.05 and therefore, uncorrelated error assumption is not violated.

# BoxCox transformation

# removing One outlier
model1_BC <- model1[-c(6040), ] # removing Outlier


for (i in colnames(model1_BC)){
  if(class(model1_BC[[i]]) == "integer" | class(model1_BC[[i]]) == "numeric"){
    model1_BC[i] <- model1_BC[i] + (abs(min(bike_data[i])))+1 #adding minimum value to make dataset positive
  }
}

bc1 <-boxTidwell(bike_count ~ ( temp + humidity), data = model1_BC)

lambda_val=bc1$result[1]

transform_boxcox <- function(data)
{
  # box-cox transformation
  data = (data^lambda_val)-1/lambda_val
  return(data)
}

for (i in colnames(model1_BC)){
  if(class(model1_BC[[i]]) == "integer" | class(model1_BC[[i]]) == "numeric"){
    model1_BC[i] <- transform_boxcox(model1_BC[i])
  }
}

# Final Model 1

# fitting new model
model1_BC_fit = lm(bike_count ~ hour + temp + humidity + solar_radiation + 
                     rainfall + snowfall + seasons + holiday, data=model1_BC
)

summary(model1_BC_fit)

# The final model can explain 71.55% of the data and it is statistically significant as the p-value of the model is less than 0.05. 
# None of the variables’s p-values are greater than 0.05 and therefore, statistically significant. However, after transformation, factors such as 
# snowfall, hour 9 and hour 16 are showing that they’re not statistically significant.

# Checking multicollinearity in the final Model
car::vif(model1_BC_fit)

# Since all the GVIF values are less than 10, it can be assumed that there is no multicollinearity for the final Model.

# Residual Plots
par(mfrow=c(2,2))
plot(model1_BC_fit , which=1:4)

# Evaluate homoscedasticity
# non-constant error variance test
ncvTest(model1_BC_fit)

#P-value is less than 0.05, implying that the constant error variance assumption is still violated after box-cox transformation.

# Test for Autocorrelated Errors
durbinWatsonTest(model1_BC_fit)

# P-value is greater than 0.05 and therefore, we cannot reject H0. The uncorrelated error assumption is not violated.

# Model 1 did not turn out to pass homoscedasticity test and therefore, deemed to not be able to predict the bike demand accurately. 
# We will now test if Model 2 can predict it better.

#__________________________________
# Model 2
#___________________________________

# Model 2 is a multiple linear regression model predicting bike rental demand using average daily weather conditions. For this model, bike demand for 
# each day and average daily weather conditions are calculated. Data will be splitted to 80-20 train and test dataset to build the model.

# creating dataframe for bike count by date
bike_by_day <- bike_data
levels(bike_by_day$holiday) <- c("No Holiday" = 1, "Holiday"= 2)
levels(bike_by_day$seasons) <- c("Winter" = 1, "Spring"= 2, "Summer" = 3, "Autumn" = 4)


bike_by_day <- bike_by_day %>% group_by(date) %>% summarise(sum_bike_count = sum(bike_count), mean_temp = mean(temp), mean_humidity = mean(humidity), 
                                                            mean_wind_speed = mean(wind_speed), mean_visibility = mean(visibility), 
                                                            mean_dew_point_temp = mean(dew_point_temp), mean_solar_radiation = mean(solar_radiation), 
                                                            mean_rainfall = mean(rainfall), mean_snowfall = mean(snowfall), 
                                                            seasons = mean(as.numeric(seasons)), holiday = mean(as.numeric(holiday)))

#Changing the factors back to original values 
bike_by_day$holiday <- as.factor(bike_by_day$holiday)
bike_by_day$seasons <- as.factor(bike_by_day$seasons)
levels(bike_by_day$holiday) <- c('1'= 'No Holiday', '2'='Holiday')
levels(bike_by_day$seasons) <- c('1'= 'Winter' , '2'='Spring', '3'= 'Summer', '4'='Autumn')
head(bike_by_day)

# Data Splitting

# Data is splitted to 80% train set and 20% test set. This is to minimise overfitting. Data is splitted by random selection using sample() method.

# Randomising our dataset

## 75% of the sample size
smp_size <- floor(0.80 * nrow(bike_by_day))

## set the seed to make partition reproducible
set.seed(123)
trainIndex <- sample(seq_len(nrow(bike_by_day)), size = smp_size)

model2_prediction <- bike_by_day[ -trainIndex,]
model2 <- bike_by_day[ trainIndex,]

# After splitting the data, train dataset is fitted to the model.

# regression model by day 
bike_by_day_model = lm(sum_bike_count ~ mean_temp + mean_humidity + mean_wind_speed + mean_visibility + mean_dew_point_temp + mean_solar_radiation + 
                         mean_rainfall + mean_snowfall + seasons + holiday, data=model2)

#summary
summary(bike_by_day_model)

# This model can explain 0.8434 and is statistically significant as p-value is < 0.05. However, it shows that there are some factors that are not 
# statistically significant. Stepwise AIC is used to calculate an optimum model for prediction.

# possible subsets regression 
stepReg=MASS::stepAIC(bike_by_day_model, direction="both")

stepReg$anova

# multicollinearity
car::vif(bike_by_day_model)

# mean_temperature and mean_dew_point_temp GVIF values are greater than 10 and therefore, there is multicollinearity. 
# The stepwise AIC removed visibility and snowfall in the final model. These variables will be removed from the final model.

# Removing columns with multicollinearity and stepwise
bike_by_day_model_final = lm(sum_bike_count ~ mean_temp + mean_humidity + mean_wind_speed + mean_solar_radiation + mean_rainfall + seasons + holiday, data=model2)

#summary
summary(bike_by_day_model_final)

# multicollinearity
car::vif(bike_by_day_model_final)

# The final model can explain 84.18% of the data and is statistically significant at 95% confidence interval as p-value < 0.05. 
# There is no multicollinearity in the final model. The p-value of mean_humidity is greater than 0.5 and therefore, is not statistically significant.


# Residual Plots
par(mfrow=c(2,2))
plot(bike_by_day_model_final , which=1:4)

# There are no outliers in the dataset. But the resituals shows a bit of hetroscedasticity.

# Evaluate homoscedasticity
# non-constant error variance test
ncvTest(bike_by_day_model_final)


# P-value is less than 0.05, implying that the constant error variance assumption is violated.

# Test for Autocorrelated Errors
durbinWatsonTest(bike_by_day_model_final)

# P-value is greater than 0.05 and therefore, we cannot reject H0. The uncorrelated error assumption is not violated.

# Test for Normally Distributed Errors
shapiro.test(bike_by_day_model_final$residuals)

# P-value of the Shapiro-Wilk is greater than 0.05 therefore, it can be concluded that normality error assumption is not violated. 
# Data will be transformed using box-cox transformation as it did not pass all the tests.

# BoxCox transformation

# Making all the values positive
bike_by_day_BC <- bike_by_day

for (i in colnames(bike_by_day_BC)){
  if(class(bike_by_day_BC[[i]]) == "integer" | class(bike_by_day_BC[[i]]) == "numeric"){
    bike_by_day_BC[i] <- round(bike_by_day_BC[i] + (abs(min(bike_by_day[i])))+1 , digits=3) #adding minimum value to make dataset positive
  }
}
bc1 <-boxTidwell(sum_bike_count ~ mean_temp +
                   mean_humidity + 
                   mean_solar_radiation 
                 ,data = bike_by_day_BC)

lambda_val=bc1$result[1]
transform_boxcox <- function(data)
{
  # box-cox transformation
  data = (data^lambda_val)-1/lambda_val
  return(data)
}

for (i in colnames(bike_by_day_BC)){
  if(class(bike_by_day_BC[[i]]) == "integer" | class(bike_by_day_BC[[i]]) == "numeric"){
    bike_by_day_BC[i] <- transform_boxcox(bike_by_day_BC[i])
  }
}

# Final Model2

# fitting new model
bike_by_day_BC_fit = lm(sum_bike_count ~ mean_temp + mean_humidity + mean_wind_speed + 
                          mean_solar_radiation + mean_rainfall + seasons + holiday, 
                        data = bike_by_day_BC)

summary(bike_by_day_BC_fit)


# The final model can explain 0.86% of the data and it is statistically significant as the p-value of the model is less than 0.05. 
# None of the variables’s p-values are greater than 0.05 and therefore, statistically significant.

# Checking multicollinearity in the final Model
car::vif(bike_by_day_BC_fit)

# Since all the GVIF values are less than 10, it can be assumed that there is no multicollinearity for the final Model.

# Residual Plots
par(mfrow=c(2,2))
plot(bike_by_day_BC_fit , which=1:4)


# Evaluate homoscedasticity
# non-constant error variance test
ncvTest(bike_by_day_BC_fit)

# P-value is still less than 0.05, implying that the constant error variance assumption is violated even after transformation.

# Test for Autocorrelated Errors
durbinWatsonTest(bike_by_day_BC_fit)

# P-value is greater than 0.05 and therefore, we cannot reject H0. The uncorrelated error assumption is not violated.

shapiro.test(bike_by_day_BC_fit$residuals)

# P-value is less than 0.05 therefore, it can be assumed that the residuals are not normally distributed for this model.

#_________________________________
# Model 3
#_________________________________

# Model 3 is a multiple linear regression model predicting bike rental demand during am hours and pm hours. 
# For this, sum of bike counts for am hours and pm hours are calculated and average values of the am and pm weather data.

# Dividing to am and pm
bike_AMPM <- bike_data
bike_AMPM$hour <- as.numeric(bike_AMPM$hour)

bike_AMPM['hour'] [bike_AMPM['hour'] <=12] <- -1
bike_AMPM['hour'] [bike_AMPM['hour'] >12] <- 1
bike_AMPM$hour <- factor(bike_AMPM$hour)


head(bike_AMPM)

# creating dataframe for bike count by am pm
levels(bike_AMPM$holiday) <- c("No Holiday" = 1, "Holiday"= 2)
levels(bike_AMPM$seasons) <- c("Winter" = 1, "Spring"= 2, "Summer" = 3, "Autumn" = 4)

bike_AMPM <- bike_AMPM %>% group_by(date,hour) %>% summarise(bike_count = sum(bike_count), temp = mean(temp), humidity = mean(humidity), 
                                                             wind_speed = mean(wind_speed), visibility = mean(visibility), 
                                                             dew_point_temp = mean(dew_point_temp), solar_radiation = mean(solar_radiation), 
                                                             rainfall = mean(rainfall), snowfall = mean(snowfall), 
                                                             seasons = mean(as.numeric(seasons)), holiday = mean(as.numeric(holiday)))

#Changing the factors back to original values 
bike_AMPM$holiday <- as.factor(bike_AMPM$holiday)
bike_AMPM$seasons <- as.factor(bike_AMPM$seasons)
levels(bike_AMPM$holiday) <- c('1'= 'No Holiday', '2'='Holiday')
levels(bike_AMPM$seasons) <- c('1'= 'Winter' , '2'='Spring', '3'= 'Summer', '4'='Autumn')

#Drop date column
bike_AMPM <- subset(bike_AMPM, select = -c(date))

head(bike_AMPM)

# Model 3 Data Splitting

#Randomise our dataset

## 80% of the sample size
smp_size <- floor(0.80 * nrow(bike_AMPM))

## set the seed to make your partition reproducible
set.seed(123)
trainIndex <- sample(seq_len(nrow(bike_AMPM)), size = smp_size)

model3_prediction <- bike_AMPM[ -trainIndex,]
model3 <- bike_AMPM[ trainIndex,]

# Regression model with all columns except functioning day
am_pm_model = lm(bike_count ~ hour+temp+humidity+wind_speed+visibility+dew_point_temp+solar_radiation+rainfall+snowfall+seasons+holiday, data=model3)

#summary
summary(am_pm_model)

#Building ANOVA table with full dataset (all columns) against intercept only model
anova(am_pm_model)

# possible subsets regression
library(MASS)
stepReg=MASS::stepAIC(am_pm_model, direction="both")

stepReg$anova

# multicollinearity
car::vif(am_pm_model)

# Stepwise method produced a final regression model without temperature, snowfall and visibility. dew_point_temp and temp GVIF values are greater 
# than 10 and therefore, there is multicollinearity. Therefore, temperature visibility and snowfall will be removed from the final model.

# Removing columns with multicollinearity and stepwise
am_pm_model_final = lm(bike_count ~ hour + humidity + wind_speed + dew_point_temp + 
                         solar_radiation + rainfall + seasons + holiday, data=model3)

#summary
summary(am_pm_model_final)

# The final model can explain 80.49% of the data and it is statistically significant as the p-value of the model is less than 0.05. 
# None of the variables’s p-values are greater than 0.05 and therefore, statistically significant.

# Checking multicollinearity in the final Model
car::vif(am_pm_model_final)

# Since all the GVIF values are less than 10, it can be assumed that there is no multicollinearity for the final Model.

# Residual plots
par(mfrow=c(2,2))
plot(am_pm_model_final , which=1:4)

# Evaluate homoscedasticity
# non-constant error variance test
library(car)
ncvTest(am_pm_model_final)

# P-value is less than 0.05, implying that the constant error variance assumption is violated.

# Test for Autocorrelated Errors
durbinWatsonTest(am_pm_model_final)


# P-value is greater than 0.05 and therefore, we cannot reject H0. The uncorrelated error assumption is not violated.

shapiro.test(am_pm_model_final$residuals)

# P-value is less than 0.05 therefore, it can be assumed that the residuals are not normally distributed for this model. 
# Therefore, box-cox transformation is performed again on the model.

# BoxCox transformation

bike_AMPM_BC <- model3[-c(185), ]

for (i in colnames(bike_AMPM_BC)){
  if(class(bike_AMPM_BC[[i]]) == "integer" | class(bike_AMPM_BC[[i]]) == "numeric"){
    bike_AMPM_BC[i] <- round(bike_AMPM_BC[i] + (abs(min(bike_AMPM[i])))+1 , digits=3) # adding minimum value to make dataset positive
  }
}

bc1 <-boxTidwell(
  bike_AMPM_BC$bike_count ~ 
    bike_AMPM_BC$humidity + 
    bike_AMPM_BC$dew_point_temp + 
    bike_AMPM_BC$solar_radiation + 
    bike_AMPM_BC$rainfall
)

lambda_val=bc1$result[1]

transform_boxcox <- function(data)
{
  # box-cox transformation
  data = data^lambda_val
  return(data)
}

for (i in colnames(bike_AMPM_BC)){
  if(class(bike_AMPM_BC[[i]]) == "integer" | class(bike_AMPM_BC[[i]]) == "numeric"){
    bike_AMPM_BC[i] <- transform_boxcox(bike_AMPM_BC[i])
  }
}

# Final Model 3

# fitting new model
bike_AMPM_BC_fit = lm(bike_AMPM_BC$bike_count ~ bike_AMPM_BC$hour + 
                        bike_AMPM_BC$humidity + 
                        bike_AMPM_BC$wind_speed + 
                        bike_AMPM_BC$dew_point_temp + 
                        bike_AMPM_BC$solar_radiation + 
                        bike_AMPM_BC$rainfall + 
                        bike_AMPM_BC$seasons +
                        bike_AMPM_BC$holiday)

summary(bike_AMPM_BC_fit)

# After transforming the data, the model can explain 71.79% of the data and it is statistically significant as the p-value of the model is 
# less than 0.05. Non of the variables’s p-values are greater than 0.05 and therefore, it is statistically significant.

# Checking multicollinearity in the final Model
car::vif(bike_AMPM_BC_fit)

# Since all the GVIF values are less than 10, it can be assumed that there is no multicollinearity for the transformed model.

# Residual Plots
par(mfrow=c(2,2))
plot(bike_AMPM_BC_fit , which=1:4)

# Evaluate homoscedasticity
# non-constant error variance test

ncvTest(bike_AMPM_BC_fit)

# P-value is still less than 0.05, implying that the constant error variance assumption is violated even after transformation.

# Test for Autocorrelated Errors
durbinWatsonTest(bike_AMPM_BC_fit)

# P-value is greater than 0.05 and therefore, we cannot reject H0. The uncorrelated error assumption is not violated.

shapiro.test(bike_AMPM_BC_fit$residuals)

# P-value is less than 0.05 therefore, it can be assumed that the residuals are not normally distributed for this model. 
# Therefore, this model cannot adequately predict the bike demand.


# Model 4

# Model 4 is a multiple linear regression model predicting bike rental demand for every day with weekend column and using daily average 
# weather conditions. We expect bike demand will be higher during weekends as more people will be hiring them for recreational purposes and 
# this will be one of the factors.

#Creating weekend colum
bike_by_day$date <- as.Date(bike_by_day$date, "%d/%m/%y")
bike_by_day$weekend = chron::is.weekend(bike_by_day$date)
bike_by_day$weekend <- as.factor(bike_by_day$weekend)
bike_by_day

# Bar plot for weekend and weekday bike demand
fig5 <- ggplot(bike_by_day, aes(x = as.factor(weekend), y = sum_bike_count)) + geom_boxplot(fill="slateblue", alpha=0.2) + ggtitle("Fig 5. Weekend and Weekday Bike Count Bar Plot") +  xlab("Weekend") + ylab("Bike Count")
fig5 + theme(plot.title = element_text(size=14, face="bold", hjust=0.5))
fig5

# As expected, Fig 5. shows that bike demand is slightly higher during the weekends.

# Data spliting for model 4

#Data is splitted to 90% train set and 10% test set instead of usual 8-20 due to being smaller dataset. Data is splitted by random 
# selection using sample() method.

#Randomising our dataset

## 75% of the sample size
smp_size <- floor(0.90 * nrow(bike_by_day))

## set the seed to make your partition reproducible
set.seed(500)
trainIndex <- sample(seq_len(nrow(bike_by_day)), size = smp_size)

model4_prediction <- bike_by_day[ -trainIndex,]
model4 <- bike_by_day[ trainIndex,]

# regression model by day with Weekend
bike_weekend_model = lm(sum_bike_count ~ mean_temp + mean_humidity + mean_wind_speed + mean_visibility + mean_dew_point_temp + mean_solar_radiation + mean_rainfall + mean_snowfall + seasons + holiday+ weekend, data=model4)

#summary
summary(bike_weekend_model)

# possible subsets regression (by day and weekend)
library(MASS)
stepReg=MASS::stepAIC(bike_weekend_model, direction="both")

stepReg$anova

# multicollinearity
car::vif(bike_weekend_model)

# mean_temp and mean_dew_point_temp GVIF values are greater than 10 and therefore, there is multicollinearity. mean_temp will be taken out 
# from the final model. The final model from stepwise AIC did not include mean_visibility and mean_snowfall. These will be taken out from our 
# final Weekend Model.

# Final regression model by day with Weekend
bike_weekend_model_final = lm(sum_bike_count ~ mean_dew_point_temp + mean_humidity + mean_wind_speed + mean_solar_radiation + mean_rainfall + seasons + holiday + weekend, data=model4)

#summary
summary(bike_weekend_model_final)

# multicollinearity
car::vif(bike_weekend_model_final)

# The final model with weekend can explain 82.36% of the data and is statistically accurate at 95% with p-value < 0.05. 
# There is no mulicollinearity in the final model.

# Residual Plots
par(mfrow=c(2,2))
plot(bike_weekend_model_final , which=1:4)

# There is no outliers in the model’s residuals according to the Cook’s distance plot.

# Evaluate homoscedasticity
# non-constant error variance test
ncvTest(bike_weekend_model_final)

# P-value is less than 0.05, implying that the constant error variance assumption is violated.

# Test for Autocorrelated Errors
durbinWatsonTest(bike_weekend_model_final)

# P-value is greater than 0.05 and therefore, we cannot reject H0. The uncorrelated error assumption is not violated.

# Test for Normally Distributed Errors
shapiro.test(bike_weekend_model_final$residuals)

# P-value of the normality test is greater than 0.05 and therefore, the residuals of the data is normally distributed. 
# However, as the model failed the homoscedascity test, the model will be transformed using box-cox transformation method.

# BoxCox transformation

bike_by_dayBC <- model4

for (i in colnames(bike_by_dayBC)){
  if(class(bike_by_dayBC[[i]]) == "integer" | class(bike_by_dayBC[[i]]) == "numeric"){
    bike_by_dayBC[i] <- bike_by_dayBC[i] + (abs(min(bike_by_day[i])))+1 #adding minimum value to make dataset positive
  }
}

bc1 <-boxTidwell(
  bike_by_dayBC$sum_bike_count ~ bike_by_dayBC$mean_humidity + 
    bike_by_dayBC$mean_wind_speed + 
    bike_by_dayBC$mean_dew_point_temp + 
    bike_by_dayBC$mean_solar_radiation + 
    bike_by_dayBC$mean_rainfall
)

lambda_val=bc1$result[1]

transform_boxcox <- function(data)
{
  # box-cox transformation
  data = ((data^lambda_val)-1)/lambda_val
  return(data)
}

for (i in colnames(bike_by_dayBC)){
  if(class(bike_by_dayBC[[i]]) == "integer" | class(bike_by_dayBC[[i]]) == "numeric"){
    bike_by_dayBC[i] <- transform_boxcox(bike_by_dayBC[i])
  }
}

# Fitting new model after box-cox transformation
bike_by_dayBC_fit = lm(bike_by_dayBC$sum_bike_count ~ bike_by_dayBC$mean_humidity + 
                         bike_by_dayBC$mean_wind_speed + 
                         bike_by_dayBC$mean_dew_point_temp + 
                         bike_by_dayBC$mean_solar_radiation + 
                         bike_by_dayBC$mean_rainfall +
                         bike_by_dayBC$seasons +
                         bike_by_dayBC$holiday +
                         bike_by_dayBC$weekend)

summary(bike_by_dayBC_fit)

# multicollinearity
car::vif(bike_by_dayBC_fit)

# The final transformed model with weekend can explain 86.3% of the data and is statistically accurate at 95% with p-value < 0.05. 
# There is no mulicollinearity in the final model.

# Residual Plots
par(mfrow=c(2,2))
plot(bike_by_dayBC_fit , which=1:4)

# Cook’s distance shows that there are some outliers (152, 99 and 39) and therfore, they will be removed from the dataset before fitting it 
# into the model again.

# Evaluate homoscedasticity
# non-constant error variance test
ncvTest(bike_by_dayBC_fit)

# P-value is less than 0.05, implying that the constant error variance assumption is violated if we do not remove the outliers. 
# Therefore, these will be removed and then the model will be tested again.


# Final Model 4

bike_by_dayBC <- bike_by_dayBC[-c(152,99,39),]

# fitting new model
bike_by_dayBC_fit = lm(bike_by_dayBC$sum_bike_count ~ bike_by_dayBC$mean_humidity + 
                         bike_by_dayBC$mean_wind_speed + 
                         bike_by_dayBC$mean_dew_point_temp + 
                         bike_by_dayBC$mean_solar_radiation + 
                         bike_by_dayBC$mean_rainfall +
                         bike_by_dayBC$seasons +
                         bike_by_dayBC$holiday +
                         bike_by_dayBC$weekend)

summary(bike_by_dayBC_fit)

# The final model with weekend can explain 87.69% of the data and is statistically accurate at 95% with p-value < 0.05. 
# There is no mulicollinearity in the final model. This is by far a model with the most prediction accuracy.

# multicollinearity
car::vif(bike_by_dayBC_fit)

# Residual Plots
par(mfrow=c(2,2))
plot(bike_by_dayBC_fit , which=1:4)

# Evaluate homoscedasticity
# non-constant error variance test
ncvTest(bike_by_dayBC_fit)

# After removing outliers, the P-value of non-constant error variance test shows that it is greater than 0.05, implying that the 
# constant error variance assumption is not violated.

# Test for Autocorrelated Errors
durbinWatsonTest(bike_by_dayBC_fit)

# P-value is greater than 0.05 and therefore, we cannot reject H0. The uncorrelated error assumption is not violated.

# Test for Normally Distributed Errors
shapiro.test(bike_by_dayBC_fit$residuals)

# P-value of the normality test is less than 0.05 and therefore residuals are not normally distributed.



# Prediction

# We have selected model 4 as our best model and to predict values using this model we will be using the model 4’s prediction dataset which 
# is model4_prediction.

# we will transform this series to positive by adding absolute smallest value to whole dataset by their columns

for (i in colnames(model4_prediction)){
  if(class(model4_prediction[[i]]) == "integer" | class(model4_prediction[[i]]) == "numeric"){
    model4_prediction[i] <- model4_prediction[i] + (abs(min(bike_by_day[i])))+1
  }
}

# Box-cox transformatino using lambda
transform_boxcox <- function(data)
{
  # box-cox transformation
  data = ((data^lambda_val)-1)/lambda_val
  return(data)
}

for (i in colnames(model4_prediction)){
  if(class(model4_prediction[[i]]) == "integer" | class(model4_prediction[[i]]) == "numeric"){
    model4_prediction[i] <- transform_boxcox(model4_prediction[i])
  }
}

head(model4_prediction)

# Model 4’s equation
# lm(sum_bike_count ~ mean_humidity + mean_wind_speed + mean_dew_point_temp + mean_solar_radiation + mean_rainfall + seasons + holiday + weekend, data=bike_by_dayBC)

# y = 19.4886 - 0.8255 X mean_humidity - 1.0416 X mean_wind_speed + 1.0738 X mean_dew_point_temp + 3.4083 X mean_solar_radiation - 
             # 1.9489 X mean_rainfall - 1.0351 X seasonsSpring - 0.4055 X seasonsSummer -2.4737 X seasonsAutumn + 0.8703 X holidayHoliday + 
              # 0.2655 X weekendTRUE + ε

# From the boxcox transformation we get lambda value of 0.128451. We will use this to transforme the data from the model4_prediction.

# creating function to predict the bike counts
pred_bike_count <- function(mean_humidity, mean_wind_speed, mean_dew_point_temp,  mean_solar_radiation, mean_rainfall, seasons, holiday, weekend){
  seasonsSummer = 0
  seasonsAutumn = 0
  seasonsSpring = 0
  holidayHoliday = 0
  weekendTRUE = 0
  
  # Assinging value as 1 if they exist in data
  if(seasons == "Summer"){
    seasonsSummer = 1
  } else if(seasons == "Autumn"){
    seasonsAutumn = 1
  } else{
    seasonsSpring = 1
  }
  
  if(holiday == "Holiday"){
    holidayHoliday = 1
  }
  
  if(weekendTRUE == "TRUE"){
    weekendTRUE = 1
  }
  
  # fitted eqN of model 4
  y =  19.4886 - 0.8255 * mean_humidity - 1.0416 * mean_wind_speed + 1.0738 * mean_dew_point_temp + 3.4083 * mean_solar_radiation - 
    1.9489 * mean_rainfall - 1.0351 * seasonsSpring - 0.4055 * seasonsSummer -2.4737 * seasonsAutumn + 0.8703 * holidayHoliday + 
    0.2655 * weekendTRUE
  
  return(as.numeric(y))
}


# Predicting Values

model4_prediction[1,] # first row of model 4 prediction

prediction1 <- pred_bike_count(
  model4_prediction[1,"mean_humidity"],
  model4_prediction[1,"mean_wind_speed"],
  model4_prediction[1,"mean_dew_point_temp"],
  model4_prediction[1,"mean_solar_radiation"],
  model4_prediction[1,"mean_rainfall"],
  model4_prediction[1,"seasons"],
  model4_prediction[1,"holiday"],
  model4_prediction[1,"weekend"]
)

prediction1

model4_prediction[2,] # second row of model 4 prediction

prediction2 <- pred_bike_count(
  model4_prediction[2,"mean_humidity"],
  model4_prediction[2,"mean_wind_speed"],
  model4_prediction[2,"mean_dew_point_temp"],
  model4_prediction[2,"mean_solar_radiation"],
  model4_prediction[2,"mean_rainfall"],
  model4_prediction[2,"seasons"],
  model4_prediction[2,"holiday"],
  model4_prediction[2,"weekend"]
)

prediction2

model4_prediction[3,] # third row of model 4 prediction

prediction3 <- pred_bike_count(
  model4_prediction[3,"mean_humidity"],
  model4_prediction[3,"mean_wind_speed"],
  model4_prediction[3,"mean_dew_point_temp"],
  model4_prediction[3,"mean_solar_radiation"],
  model4_prediction[3,"mean_rainfall"],
  model4_prediction[3,"seasons"],
  model4_prediction[3,"holiday"],
  model4_prediction[3,"weekend"]
)

prediction3

model4_prediction[4,] # four row of model 4 prediction

prediction4 <- pred_bike_count(
  model4_prediction[4,"mean_humidity"],
  model4_prediction[4,"mean_wind_speed"],
  model4_prediction[4,"mean_dew_point_temp"],
  model4_prediction[4,"mean_solar_radiation"],
  model4_prediction[4,"mean_rainfall"],
  model4_prediction[4,"seasons"],
  model4_prediction[4,"holiday"],
  model4_prediction[4,"weekend"]
)

prediction4

model4_prediction[5,] # five row of model 4 prediction

prediction5 <- pred_bike_count(
  model4_prediction[5,"mean_humidity"],
  model4_prediction[5,"mean_wind_speed"],
  model4_prediction[5,"mean_dew_point_temp"],
  model4_prediction[5,"mean_solar_radiation"],
  model4_prediction[5,"mean_rainfall"],
  model4_prediction[5,"seasons"],
  model4_prediction[5,"holiday"],
  model4_prediction[5,"weekend"]
)

prediction5

# Prediction table of daily bike demand

actual_bikecount <- c(model4_prediction[1,]$sum_bike_count, model4_prediction[2,]$sum_bike_count, model4_prediction[3,]$sum_bike_count, model4_prediction[4,]$sum_bike_count, model4_prediction[5,]$sum_bike_count)
predicted_bikecount <- c(prediction1, prediction2, prediction3, prediction4, prediction5)

df_prediction <- data.frame(actual_bikecount, predicted_bikecount)

df_prediction # final prediction table











































































