### Unit Root Test ###
# ADF, PP and KPSS Tests using "urca" package
getwd()
setwd("D:/0교재 집필/R code/Ch1and2")
dir()

install.packages("readxl")
install.packages("urca")

library(readxl)
library(urca)

data <- read_excel("DefaultPremium_data.xls")
daily = matrix(data$Daily)
daily = na.omit(daily)
plot(daily, type = 'l')  

## AR 계수 추정
ar1=arima(daily, order=c(1,0,0))
ar1

ar2=arima(daily, order=c(1,0,0), xreg = 1:length(daily))
ar2

## Augmented Dickey-Fuller (ADF) Test, 
##귀무가설: Unit Root

## Model with intercept
daily_adf1=ur.df(daily, type="drift")
summary(daily_adf1)
daily_adf1@teststat
daily_adf1@cval

## Model with intercept and trend
daily_adf2=ur.df(daily, type="trend")
summary(daily_adf2)
daily_adf2@teststat
daily_adf2@cval


## Kwiatkowsk-Phillips-Schmidt-Shin (KPSS) Test 
## 귀무가설: Stationarity

# Model with intercept
daily_kpss1=ur.kpss(daily, type="mu", lags = "long")
summary(daily_kpss1)

# Model with intercept and trend
daily_kpss2=ur.kpss(daily, type="tau", lags = "long" )
summary(daily_kpss2)
