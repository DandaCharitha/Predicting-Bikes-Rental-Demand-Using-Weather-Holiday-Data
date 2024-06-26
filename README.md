# Predicting-Bikes-Rental-Demand-Using-Weather-Holiday-Data

## Introduction

Bike sharing system is an innovative transportation strategy that provides individuals with bikes for their common use on a short-term basis for a price or for free. Over the last few decades, there has been a significant increase in the popularity of bike-sharing systems all over the world. This is because it is an environmentally sustainable, convenient, and economical way of improving urban mobility. Therefore, predicting the required bike count for each hour is crucial for maintaining a stable supply of rental bikes. In addition to this, this system also helps to promote healthier habits among its users and reduce fuel consumption.

## Problem Statement

With the growing demand and user base for bike-sharing system, providing the city with a stable supply of rental bikes could eventually become a challenging task. In this project, my goal is to predict bike rental demand by leveraging weather data and holiday information. By analyzing this we can see how weather conditions and holidays impact bike rental patterns, we can uncover valuable insights into the factors influencing urban mobility. The success of bike sharing system relies in ensuring that the quality of facilities provided, meets the needs and expectations of the users. Therefore, it is important to ensure that rental bikes are available and accessible to the users at right time ,as it reduces the waiting time. So, by doing this project, we can addresses this challenge by integrating weather data and holiday information into our model, aiming to provide cities with effective tools for managing their bike rental systems and enhancing urban transportation infrastructure.

## Describing DataSet

The dataset has 8760 data points and each of them are representing weather and holiday information which are possible major factors in hiring a bike. The observations in the dataset were recorded during a span of 365 days, from December 2017 to November 2018. The dataset contains 14 attributes, 10 numerical and 3 categorical and 1 time series values.

Target values: Rented Bike count - Count of bikes rented at each hour.
Feature attributes: Date: year-month-day, Rented Bike count — Count of bikes rented at each hour, Hour — Hour of the day, Temperature-Temperature in Celsius, Humidity — %, Windspeed — m/s, Visibility — 10m, Dew point temperature — Celsius, Solar radia1on — MJ/m2, Rainfall — mm, Snowfall — cm, Seasons — Winter, Spring, Summer, Autumn, Holiday — Holiday/No holiday, Functional Day — NoFunc (Non-Functional Hours), Fun(Functional hours).

## Method

The data is splitted to 80-20 to minimise overfitting and to test the prediction accuracy of the model.
In this project I have used 4 regression models to predict the bike rental demands.

Model 1: Predicting bike rental demand based on hourly weather attributes to understand their direct impact on rental patterns throughout the day.
Model 2 : To forecast bike rental demand using average daily weather conditions, including mean temperature, mean humidity, mean windspeed, mean visibility, mean dew point temperature, mean solar radia1on, mean rainfall, and mean snowfall.
Model 3 : Predicting bike rental demand separately for AM and PM hours by analyzing hourly weather condi1ons, allowing for a better understanding of rental patterns during different times of the day.
Model 4 : To forecast daily bike rental demand while considering the influence of weekend days, along with daily average weather conditions, seasons, and holiday periods, providing comprehensive insights into the factors affecting overall rental patterns.

Categorical columns, hours, seasons, holiday and functional day are set as factors before fitting into the models. All the feature attributes of the dataset, except functioning day, is used to predict bike demand in all the models.

For each of the models, the dataset is splitted and then fit into the regression model. Then the optimum model is calculated using the stepwise regression method and VIF regression to test multicollinearity. Then statistical tests such as the non-constant error variance test to check homoscedasticity of the model, Cook’s distance to check outliers, Durbin Watson test for autocorrelation and Shapiro-Wilk tests for normality, are performed to check if the model is adequate. If the model fails the tests, especially for homoscedasticity, box-cox transformation is performed on the model. Then outliers are removed and the data is then fitted to model again and tested for adequacy.

## Final Model Selection
Out of all the models, Model 4 is shown to be the most reliable and accurate at predicting bike demand by passing all the requirements except normality of residuals. As expected, ‘weekend’ plays a factor in the prediction of bike demand and Model 4: predicting bike rental demand for everyday with added ‘weekend’ feature gives the best prediction of bike demand, giving the accuracy of 87.69%. For normality assumption of regression we will consider that since the dataset is not too small, the p-value is not too much smaller than 0.05 and from QQ-plot we can see that most of the residuals are on the dotted lines. The model 4 passes the rest of the tests, we conclude that Model 4 can adequately predict the bike demand. It may also be valuable to add on other transformation methods such as min-max transformation after box-cox transformation on the model to see whether models that failed adequacy tests will give adequate models.

## Conclusion
Even though regression models could not predict hourly bike demand as accurately as other machine learning models and deep learning models, the accuracy for predicting daily bike demand is considerably strong. However, this model can only predict daily bike demand instead of hourly, which may be more valuable for the bike hire companies. This means other machine learning techniques may be more suitable for the purposes of the bike hire companies. However, predicting daily bike demand is still valuable as this can help in activities such as scheduling bike servicing scheduling, seasonal bike stockings and profit optimisation. I did not test for adding the weekend factor for the hourly bike demand prediction model. If these factors are also accounted in the model, we may be able to improve the prediction accuracy.
