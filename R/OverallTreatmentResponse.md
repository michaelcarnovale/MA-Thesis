Prediction of Overall and Differential Treatment Response for Depression
using Personality Facets: Evaluating Multivariable Prediction Models
using Multi-Study Data
================
Michael Carnovale

This markdown file involves the following analyses steps: \* For BDI-II
and HAM-D scores separately \* Estimating predictive accuracy using
series of OLS regression models (i.e., with subsets of IVs) on test data
\* Estimating predictive accuracy using series of elastic net regression
models (i.e., with same subset of IVs as in OLS) on test data

Note: for an easier navigation of the sections in this Markdown file,
please click the ‘bullet point’ button at the top-left of the viewer to
see a table of contents.

# Load required packages

``` r
library(caret)   ## Package to help with machine learning functions
library(psych)   ## Descriptive stats and reliability statistics
library(dplyr)   ## Data management
library(car)     ## More data management
library(foreign) ## To import datasets
library(effsize) ## Effect sizes with CIs
library(visreg)  ## Visualization of regression models
library(jtools)  ## Various functions for regression models
library(glmnet)  ## Penalized regression package
library(summarytools)
library(boot)    ## Bootstrapping
```

    ##         boot summarytools       glmnet       Matrix       jtools       visreg 
    ##     "1.3-25"      "0.9.6"        "4.0"     "1.2-18"      "2.1.3"      "2.7.0" 
    ##      effsize      foreign          car      carData        dplyr        psych 
    ##      "0.8.0"     "0.8-80"      "3.0-8"      "3.0-4"      "1.0.6"  "1.9.12.31" 
    ##        caret      ggplot2      lattice 
    ##     "6.0-86"      "3.3.4"    "0.20-41"

# Data management

## Making separate datasets

## Handling missing data

# Overall treatment response

## Functions

``` r
## Best model function 
best.model = function(model) {
  best = which(rownames(model$results) == rownames(model$bestTune))
  best.result = model$results[best,]
  rownames(best.result) = NULL
  best.result # Gives you the following from the best performing model: hyperparameters, RMSE (and its SD),
              # R2 (and its SD), and MAE (and its SD)
}

## Cross-validation settings
ctrl.overall = trainControl(method = "repeatedcv", # Repeated k-fold CV
                       number = 5,                 # 5-fold CV
                       repeats = 5,               # Repeated 5 times
                       savePredictions = T,
                       verbose = F)

## Standardizing variables function
standardizing = function(data = NULL, var.info = NULL){
  processed.data = list()
  var.names = var.info[, 1]
  var.types = var.info[, 2]
  data.cont = subset(data, select = var.names[var.types=="numeric"])
  data.binary = subset(data, select = var.names[var.types=="binary"])
  
  zscoring.info = preProcess(data.cont, method = c("center", "scale")) # Getting z-scores for IVs
  numeric.z = predict(zscoring.info, data.cont)
  
  data.zscored = data
  data.zscored[, var.names[var.types=="binary"]] = data.binary
  data.zscored[, var.names[var.types=="numeric"]] = numeric.z
  
  return(data.zscored)
}

bdi.info = matrix(c(
  "NRN1.1", "numeric",
  "NRN2.1", "numeric",
  "NRN3.1", "numeric",
  "NRN4.1", "numeric",
  "NRN5.1", "numeric",
  "NRN6.1", "numeric",
  "NRE1.1", "numeric",
  "NRE2.1", "numeric", 
  "NRE3.1", "numeric",
  "NRE4.1", "numeric",
  "NRE5.1", "numeric",
  "NRE6.1", "numeric",
  "NRO1.1", "numeric", 
  "NRO2.1", "numeric", 
  "NRO3.1", "numeric",
  "NRO4.1", "numeric",
  "NRO5.1", "numeric", 
  "NRO6.1", "numeric",
  "NRA1.1", "numeric", 
  "NRA2.1", "numeric",
  "NRA3.1", "numeric", 
  "NRA4.1", "numeric",
  "NRA5.1", "numeric",
  "NRA6.1", "numeric",
  "NRC1.1", "numeric", 
  "NRC2.1", "numeric",
  "NRC3.1", "numeric", 
  "NRC4.1", "numeric",
  "NRC5.1", "numeric",
  "NRC6.1", "numeric",
  "BDI2.1", "numeric",
  "sex", "binary",
  "age", "numeric"
  
), ncol = 2, byrow = T)

hd.info = matrix(c(
  "NRN1.1", "numeric",
  "NRN2.1", "numeric",
  "NRN3.1", "numeric",
  "NRN4.1", "numeric",
  "NRN5.1", "numeric",
  "NRN6.1", "numeric",
  "NRE1.1", "numeric",
  "NRE2.1", "numeric", 
  "NRE3.1", "numeric",
  "NRE4.1", "numeric",
  "NRE5.1", "numeric",
  "NRE6.1", "numeric",
  "NRO1.1", "numeric", 
  "NRO2.1", "numeric", 
  "NRO3.1", "numeric",
  "NRO4.1", "numeric",
  "NRO5.1", "numeric", 
  "NRO6.1", "numeric",
  "NRA1.1", "numeric", 
  "NRA2.1", "numeric",
  "NRA3.1", "numeric", 
  "NRA4.1", "numeric",
  "NRA5.1", "numeric",
  "NRA6.1", "numeric",
  "NRC1.1", "numeric", 
  "NRC2.1", "numeric",
  "NRC3.1", "numeric", 
  "NRC4.1", "numeric",
  "NRC5.1", "numeric",
  "NRC6.1", "numeric",
  "hamd1", "numeric",
  "sex", "binary",
  "age", "numeric"
  
), ncol = 2, byrow = T)

iv.bdi.names = bdi.info[, 1]
iv.hd.names = hd.info[, 1]



###### Elastic net regression function #####

elastic.net = function(data, zscored.ivs, outcome){
  
  eGrid = expand.grid(.alpha = seq(0, 1, by = .25), .lambda = seq(0, 2, by = .05)) # Parameter space for possible hyperparameters
  
  
  set.seed(2020)
  model = caret::train(y ~ ., 
                       data = data,
                       method = "glmnet",
                       metric = "RMSE", # Pick the model with the lowest RMSE
                       tuneGrid = eGrid,
                       trControl = ctrl.overall)
  
  model.stats = getTrainPerf(model)
  print(model.stats)
  
  print(best.model(model))
  print(model$bestTune)
  
  final.model = glmnet::glmnet(data.matrix(zscored.ivs), as.matrix(outcome), alpha = model$bestTune["alpha"],
                               lambda = model$bestTune["lambda"]) # Fitting the ENR model with best hyperparameters
  print(coef(final.model))
  return(final.model)
}
```

## BDI-II scores

``` r
## Splitting data into training and test sets
set.seed(2020)
bdi.index = createDataPartition(neo.depr.bdi$BDI2.2, p = .7, list = F) # 70/30 split
bdi.overall.train = neo.depr.bdi[bdi.index, ]
bdi.overall.test = neo.depr.bdi[-bdi.index, ]

nrow(bdi.overall.train)
```

    ## [1] 147

``` r
nrow(bdi.overall.test)
```

    ## [1] 61

### Ordinary least squares

``` r
#### Pre-tx BDI-II only
bdi.overall.lm1 = lm(BDI2.2 ~ BDI2.1, data = bdi.overall.train)
summ(bdi.overall.lm1)
```

    ## MODEL INFO:
    ## Observations: 147
    ## Dependent Variable: BDI2.2
    ## Type: OLS linear regression 
    ## 
    ## MODEL FIT:
    ## F(1,145) = 29.60, p = 0.00
    ## R² = 0.17
    ## Adj. R² = 0.16 
    ## 
    ## Standard errors: OLS
    ## ------------------------------------------------
    ##                      Est.   S.E.   t val.      p
    ## ----------------- ------- ------ -------- ------
    ## (Intercept)         -0.18   3.18    -0.06   0.96
    ## BDI2.1               0.54   0.10     5.44   0.00
    ## ------------------------------------------------

``` r
# Test sample
pred.bdi.ov.lm1.test = predict(bdi.overall.lm1, bdi.overall.test)
postResample(pred.bdi.ov.lm1.test, bdi.overall.test$BDI2.2) 
```

    ##      RMSE  Rsquared       MAE 
    ## 9.5660167 0.1004814 8.0104360

``` r
#### Model 1: Pre-tx BDI-II and demographics
bdi.overall.lm2 = lm(BDI2.2 ~ BDI2.1 + age + sex, data = bdi.overall.train)
summ(bdi.overall.lm2)
```

    ## MODEL INFO:
    ## Observations: 147
    ## Dependent Variable: BDI2.2
    ## Type: OLS linear regression 
    ## 
    ## MODEL FIT:
    ## F(3,143) = 10.04, p = 0.00
    ## R² = 0.17
    ## Adj. R² = 0.16 
    ## 
    ## Standard errors: OLS
    ## ------------------------------------------------
    ##                      Est.   S.E.   t val.      p
    ## ----------------- ------- ------ -------- ------
    ## (Intercept)         -2.43   4.68    -0.52   0.60
    ## BDI2.1               0.54   0.10     5.39   0.00
    ## age                  0.04   0.08     0.47   0.64
    ## sex1                 1.48   1.85     0.80   0.42
    ## ------------------------------------------------

``` r
# Test sample
pred.bdi.ov.lm2.test = predict(bdi.overall.lm2, bdi.overall.test)
postResample(pred.bdi.ov.lm2.test, bdi.overall.test$BDI2.2) 
```

    ##      RMSE  Rsquared       MAE 
    ## 9.5055058 0.1117186 8.0573794

``` r
#### Model 2: FFM facets only
bdi.overall.train.facets = bdi.overall.train[,c(1:30, 32)]
bdi.overall.lm3 = lm(BDI2.2 ~ ., data = bdi.overall.train.facets)
summ(bdi.overall.lm3)
```

    ## MODEL INFO:
    ## Observations: 147
    ## Dependent Variable: BDI2.2
    ## Type: OLS linear regression 
    ## 
    ## MODEL FIT:
    ## F(30,116) = 2.65, p = 0.00
    ## R² = 0.41
    ## Adj. R² = 0.25 
    ## 
    ## Standard errors: OLS
    ## --------------------------------------------------
    ##                       Est.    S.E.   t val.      p
    ## ----------------- -------- ------- -------- ------
    ## (Intercept)         -14.24   11.50    -1.24   0.22
    ## NRN1.1                0.71    0.25     2.82   0.01
    ## NRN2.1               -0.02    0.28    -0.08   0.93
    ## NRN3.1                0.22    0.32     0.70   0.49
    ## NRN4.1                0.23    0.27     0.85   0.39
    ## NRN5.1               -0.35    0.23    -1.51   0.13
    ## NRN6.1               -0.27    0.30    -0.90   0.37
    ## NRE1.1               -0.34    0.29    -1.16   0.25
    ## NRE2.1               -0.04    0.23    -0.17   0.87
    ## NRE3.1                0.71    0.24     3.00   0.00
    ## NRE4.1                0.15    0.26     0.57   0.57
    ## NRE5.1                0.70    0.24     2.93   0.00
    ## NRE6.1               -0.55    0.25    -2.15   0.03
    ## NRO1.1                0.23    0.21     1.06   0.29
    ## NRO2.1                0.18    0.21     0.85   0.40
    ## NRO3.1                0.02    0.27     0.06   0.95
    ## NRO4.1               -0.39    0.27    -1.44   0.15
    ## NRO5.1               -0.23    0.21    -1.10   0.27
    ## NRO6.1               -0.23    0.28    -0.84   0.40
    ## NRA1.1               -0.20    0.25    -0.80   0.43
    ## NRA2.1                0.44    0.24     1.86   0.07
    ## NRA3.1                0.29    0.31     0.93   0.36
    ## NRA4.1                0.36    0.30     1.21   0.23
    ## NRA5.1                0.43    0.25     1.74   0.09
    ## NRA6.1               -0.50    0.28    -1.82   0.07
    ## NRC1.1                0.96    0.30     3.17   0.00
    ## NRC2.1               -0.02    0.24    -0.07   0.94
    ## NRC3.1               -0.16    0.27    -0.62   0.54
    ## NRC4.1                0.25    0.24     1.06   0.29
    ## NRC5.1               -0.53    0.26    -2.01   0.05
    ## NRC6.1               -0.45    0.24    -1.86   0.07
    ## --------------------------------------------------

``` r
# Test sample
bdi.overall.test.facets = bdi.overall.test[,c(1:30, 32)]
pred.bdi.ov.lm3.test = predict(bdi.overall.lm3, bdi.overall.test.facets)
postResample(pred.bdi.ov.lm3.test, bdi.overall.test.facets$BDI2.2) 
```

    ##        RMSE    Rsquared         MAE 
    ## 12.76808693  0.01094401 10.20747656

``` r
#### Model 3: Pre-tx BDI-II and FFM facets
bdi.overall.train.facets2 = bdi.overall.train[,c(1:32)]
bdi.overall.lm4 = lm(BDI2.2 ~ ., data = bdi.overall.train.facets2)
summ(bdi.overall.lm4)
```

    ## MODEL INFO:
    ## Observations: 147
    ## Dependent Variable: BDI2.2
    ## Type: OLS linear regression 
    ## 
    ## MODEL FIT:
    ## F(31,115) = 3.26, p = 0.00
    ## R² = 0.47
    ## Adj. R² = 0.32 
    ## 
    ## Standard errors: OLS
    ## --------------------------------------------------
    ##                       Est.    S.E.   t val.      p
    ## ----------------- -------- ------- -------- ------
    ## (Intercept)         -21.18   11.11    -1.91   0.06
    ## NRN1.1                0.35    0.26     1.33   0.18
    ## NRN2.1                0.17    0.27     0.65   0.52
    ## NRN3.1                0.03    0.31     0.09   0.93
    ## NRN4.1                0.19    0.25     0.77   0.45
    ## NRN5.1               -0.46    0.22    -2.07   0.04
    ## NRN6.1               -0.29    0.29    -1.00   0.32
    ## NRE1.1               -0.17    0.28    -0.59   0.55
    ## NRE2.1               -0.01    0.22    -0.05   0.96
    ## NRE3.1                0.62    0.23     2.73   0.01
    ## NRE4.1                0.07    0.25     0.30   0.77
    ## NRE5.1                0.66    0.23     2.88   0.00
    ## NRE6.1               -0.43    0.24    -1.77   0.08
    ## NRO1.1                0.22    0.20     1.08   0.28
    ## NRO2.1                0.25    0.20     1.27   0.21
    ## NRO3.1                0.04    0.25     0.17   0.86
    ## NRO4.1               -0.43    0.26    -1.67   0.10
    ## NRO5.1               -0.37    0.20    -1.86   0.07
    ## NRO6.1               -0.10    0.27    -0.37   0.71
    ## NRA1.1               -0.21    0.24    -0.88   0.38
    ## NRA2.1                0.34    0.23     1.50   0.14
    ## NRA3.1                0.13    0.30     0.45   0.65
    ## NRA4.1                0.51    0.29     1.78   0.08
    ## NRA5.1                0.37    0.24     1.55   0.12
    ## NRA6.1               -0.44    0.26    -1.68   0.10
    ## NRC1.1                0.77    0.29     2.64   0.01
    ## NRC2.1               -0.06    0.23    -0.26   0.79
    ## NRC3.1               -0.10    0.25    -0.38   0.70
    ## NRC4.1                0.27    0.23     1.17   0.24
    ## NRC5.1               -0.42    0.25    -1.67   0.10
    ## NRC6.1               -0.16    0.24    -0.65   0.52
    ## BDI2.1                0.47    0.13     3.63   0.00
    ## --------------------------------------------------

``` r
# Test sample
bdi.overall.test.facets2 = bdi.overall.test[,c(1:32)]
pred.bdi.ov.lm4.test = predict(bdi.overall.lm4, bdi.overall.test.facets2)
postResample(pred.bdi.ov.lm4.test, bdi.overall.test.facets2$BDI2.2) 
```

    ##       RMSE   Rsquared        MAE 
    ## 12.1541976  0.0225949  9.4729020

``` r
#### Model 4: Demographics and FFM facets
bdi.overall.train.facets3 = bdi.overall.train[,c(1:30, 35, 36, 32)]
bdi.overall.lm6 = lm(BDI2.2 ~ ., data = bdi.overall.train.facets3)
summ(bdi.overall.lm6)
```

    ## MODEL INFO:
    ## Observations: 147
    ## Dependent Variable: BDI2.2
    ## Type: OLS linear regression 
    ## 
    ## MODEL FIT:
    ## F(32,114) = 2.46, p = 0.00
    ## R² = 0.41
    ## Adj. R² = 0.24 
    ## 
    ## Standard errors: OLS
    ## --------------------------------------------------
    ##                       Est.    S.E.   t val.      p
    ## ----------------- -------- ------- -------- ------
    ## (Intercept)         -13.64   12.03    -1.13   0.26
    ## NRN1.1                0.71    0.26     2.74   0.01
    ## NRN2.1               -0.00    0.28    -0.01   0.99
    ## NRN3.1                0.24    0.32     0.74   0.46
    ## NRN4.1                0.21    0.27     0.78   0.44
    ## NRN5.1               -0.36    0.23    -1.55   0.13
    ## NRN6.1               -0.28    0.31    -0.92   0.36
    ## NRE1.1               -0.39    0.31    -1.28   0.20
    ## NRE2.1               -0.03    0.23    -0.14   0.89
    ## NRE3.1                0.70    0.24     2.95   0.00
    ## NRE4.1                0.15    0.27     0.55   0.58
    ## NRE5.1                0.68    0.25     2.73   0.01
    ## NRE6.1               -0.51    0.26    -1.94   0.05
    ## NRO1.1                0.23    0.22     1.07   0.29
    ## NRO2.1                0.17    0.21     0.83   0.41
    ## NRO3.1               -0.01    0.28    -0.03   0.98
    ## NRO4.1               -0.39    0.28    -1.41   0.16
    ## NRO5.1               -0.20    0.21    -0.93   0.35
    ## NRO6.1               -0.26    0.28    -0.91   0.37
    ## NRA1.1               -0.19    0.26    -0.74   0.46
    ## NRA2.1                0.44    0.24     1.83   0.07
    ## NRA3.1                0.30    0.31     0.95   0.34
    ## NRA4.1                0.36    0.30     1.20   0.23
    ## NRA5.1                0.43    0.25     1.69   0.09
    ## NRA6.1               -0.51    0.28    -1.83   0.07
    ## NRC1.1                0.96    0.31     3.13   0.00
    ## NRC2.1               -0.02    0.24    -0.10   0.92
    ## NRC3.1               -0.12    0.28    -0.44   0.66
    ## NRC4.1                0.23    0.24     0.95   0.34
    ## NRC5.1               -0.56    0.27    -2.07   0.04
    ## NRC6.1               -0.45    0.24    -1.84   0.07
    ## sex1                  1.23    2.13     0.58   0.56
    ## age                  -0.01    0.10    -0.12   0.91
    ## --------------------------------------------------

``` r
# Test sample
bdi.overall.test.facets3 = bdi.overall.test[,c(1:30, 35, 36, 32)]
pred.bdi.ov.lm6.test = predict(bdi.overall.lm6, bdi.overall.test.facets3)
postResample(pred.bdi.ov.lm6.test, bdi.overall.test.facets3$BDI2.2) 
```

    ##       RMSE   Rsquared        MAE 
    ## 12.5981925  0.0138286 10.1408289

``` r
#### Model 5: Every IV
bdi.overall.train = bdi.overall.train[,c(1:32, 35, 36)]
bdi.overall.lm5 = lm(BDI2.2 ~ ., data = bdi.overall.train)
summ(bdi.overall.lm5)
```

    ## MODEL INFO:
    ## Observations: 147
    ## Dependent Variable: BDI2.2
    ## Type: OLS linear regression 
    ## 
    ## MODEL FIT:
    ## F(33,113) = 3.02, p = 0.00
    ## R² = 0.47
    ## Adj. R² = 0.31 
    ## 
    ## Standard errors: OLS
    ## --------------------------------------------------
    ##                       Est.    S.E.   t val.      p
    ## ----------------- -------- ------- -------- ------
    ## (Intercept)         -21.67   11.67    -1.86   0.07
    ## NRN1.1                0.33    0.27     1.24   0.22
    ## NRN2.1                0.18    0.27     0.64   0.52
    ## NRN3.1                0.03    0.31     0.11   0.92
    ## NRN4.1                0.20    0.26     0.76   0.45
    ## NRN5.1               -0.47    0.22    -2.09   0.04
    ## NRN6.1               -0.29    0.29    -0.99   0.33
    ## NRE1.1               -0.20    0.30    -0.68   0.50
    ## NRE2.1               -0.00    0.22    -0.00   1.00
    ## NRE3.1                0.61    0.23     2.66   0.01
    ## NRE4.1                0.07    0.25     0.28   0.78
    ## NRE5.1                0.66    0.24     2.78   0.01
    ## NRE6.1               -0.42    0.25    -1.66   0.10
    ## NRO1.1                0.23    0.21     1.11   0.27
    ## NRO2.1                0.24    0.20     1.22   0.23
    ## NRO3.1                0.05    0.27     0.19   0.85
    ## NRO4.1               -0.44    0.26    -1.68   0.10
    ## NRO5.1               -0.35    0.21    -1.71   0.09
    ## NRO6.1               -0.11    0.27    -0.41   0.68
    ## NRA1.1               -0.22    0.25    -0.90   0.37
    ## NRA2.1                0.34    0.23     1.44   0.15
    ## NRA3.1                0.14    0.30     0.48   0.63
    ## NRA4.1                0.51    0.29     1.75   0.08
    ## NRA5.1                0.36    0.24     1.50   0.14
    ## NRA6.1               -0.44    0.27    -1.65   0.10
    ## NRC1.1                0.78    0.30     2.64   0.01
    ## NRC2.1               -0.06    0.23    -0.27   0.79
    ## NRC3.1               -0.09    0.27    -0.32   0.75
    ## NRC4.1                0.25    0.23     1.08   0.28
    ## NRC5.1               -0.43    0.26    -1.68   0.10
    ## NRC6.1               -0.16    0.25    -0.63   0.53
    ## BDI2.1                0.47    0.13     3.57   0.00
    ## sex1                  0.93    2.03     0.46   0.65
    ## age                   0.02    0.10     0.21   0.83
    ## --------------------------------------------------

``` r
# Test sample
bdi.overall.test = bdi.overall.test[,c(1:32, 35, 36)]
pred.bdi.ov.lm5.test = predict(bdi.overall.lm5, bdi.overall.test)
postResample(pred.bdi.ov.lm5.test, bdi.overall.test$BDI2.2) 
```

    ##        RMSE    Rsquared         MAE 
    ## 12.08507660  0.02454955  9.39852547

### Elastic net regression

``` r
#### Standardizing IVs

# Training set
iv.bdi.process = subset(bdi.overall.train, select = iv.bdi.names)
iv.bdi.processed = standardizing(data = iv.bdi.process, var.info = bdi.info)

bdi.overall.train.z = iv.bdi.processed
bdi.overall.train.z$y = bdi.overall.train$BDI2.2
bdi.overall.train.outcome = select(bdi.overall.train, BDI2.2)

# Test set
iv.bdi.process.test = subset(bdi.overall.test, select = iv.bdi.names)
iv.bdi.processed.test = standardizing(data = iv.bdi.process.test, var.info = bdi.info)

bdi.overall.test.z = iv.bdi.processed.test
bdi.overall.test.z$y = bdi.overall.test$BDI2.2
bdi.overall.test.outcome = select(bdi.overall.test, BDI2.2)

#### Model 1: Pre-tx BDI-II and demographics only
iv.bdi.processed1 = iv.bdi.processed[,c(31:33)]
bdi.overall.train.z1 = bdi.overall.train.z[,c(31:34)]
bdi.overall.enr1 = elastic.net(data = bdi.overall.train.z1, zscored.ivs = iv.bdi.processed1, outcome = bdi.overall.train.outcome)
```

    ##   TrainRMSE TrainRsquared TrainMAE method
    ## 1  10.91164      0.164035 8.689279 glmnet
    ##   alpha lambda     RMSE Rsquared      MAE    RMSESD RsquaredSD     MAESD
    ## 1     1    0.9 10.91164 0.164035 8.689279 0.7096093  0.1169811 0.5945251
    ##     alpha lambda
    ## 183     1    0.9
    ## 4 x 1 sparse Matrix of class "dgCMatrix"
    ##                    s0
    ## (Intercept) 16.411804
    ## BDI2.1       3.931116
    ## sex          .       
    ## age          .

``` r
# Test sample
bdi.overall.test.enr1 = bdi.overall.test.z[,c(31:33)]
pred.bdi.ov.enr1.test = predict.glmnet(bdi.overall.enr1, newx = data.matrix(bdi.overall.test.enr1))
postResample(pred.bdi.ov.enr1.test, bdi.overall.test$BDI2.2) 
```

    ##      RMSE  Rsquared       MAE 
    ## 9.5765630 0.1004814 8.0542841

``` r
#### Model 2: FFM facets only
iv.bdi.processed2 = iv.bdi.processed[,c(1:30)]
bdi.overall.train.z2 = bdi.overall.train.z[,c(1:30, 34)]
bdi.overall.enr2 = elastic.net(data = bdi.overall.train.z2, zscored.ivs = iv.bdi.processed2, outcome = bdi.overall.train.outcome)
```

    ##   TrainRMSE TrainRsquared TrainMAE method
    ## 1  11.14753      0.124387 8.876155 glmnet
    ##   alpha lambda     RMSE Rsquared      MAE   RMSESD RsquaredSD    MAESD
    ## 1  0.25    1.8 11.14753 0.124387 8.876155 1.157953 0.07933812 1.000924
    ##    alpha lambda
    ## 78  0.25    1.8
    ## 31 x 1 sparse Matrix of class "dgCMatrix"
    ##                      s0
    ## (Intercept) 16.41180370
    ## NRN1.1       2.50616243
    ## NRN2.1       .         
    ## NRN3.1       0.59288709
    ## NRN4.1       0.22793384
    ## NRN5.1      -0.41234839
    ## NRN6.1       .         
    ## NRE1.1      -0.95885780
    ## NRE2.1      -0.01041001
    ## NRE3.1       1.80435388
    ## NRE4.1       .         
    ## NRE5.1       1.23760786
    ## NRE6.1      -1.14433204
    ## NRO1.1       0.38279730
    ## NRO2.1       .         
    ## NRO3.1       .         
    ## NRO4.1      -0.29363031
    ## NRO5.1       .         
    ## NRO6.1       .         
    ## NRA1.1      -0.42720599
    ## NRA2.1       1.19525024
    ## NRA3.1       0.71657718
    ## NRA4.1       0.07243733
    ## NRA5.1       1.13219402
    ## NRA6.1      -0.80306639
    ## NRC1.1       1.91068372
    ## NRC2.1       .         
    ## NRC3.1       .         
    ## NRC4.1       0.19580979
    ## NRC5.1      -1.21835838
    ## NRC6.1       .

``` r
# Test sample
bdi.overall.test.enr2 = bdi.overall.test.z[,c(1:30)]
pred.bdi.ov.enr2.test = predict.glmnet(bdi.overall.enr2, newx = data.matrix(bdi.overall.test.enr2))
postResample(pred.bdi.ov.enr2.test, bdi.overall.test$BDI2.2) 
```

    ##       RMSE   Rsquared        MAE 
    ## 9.75443790 0.07812139 8.12081858

``` r
#### Model 3: Pre-tx BDI-II and FFM facets
iv.bdi.processed3 = iv.bdi.processed[,c(1:31)]
bdi.overall.train.z3 = bdi.overall.train.z[,c(1:31, 34)]
bdi.overall.enr3 = elastic.net(data = bdi.overall.train.z3, zscored.ivs = iv.bdi.processed3, outcome = bdi.overall.train.outcome)
```

    ##   TrainRMSE TrainRsquared TrainMAE method
    ## 1  10.64947     0.1941543 8.322074 glmnet
    ##   alpha lambda     RMSE  Rsquared      MAE    RMSESD RsquaredSD     MAESD
    ## 1  0.25   1.55 10.64947 0.1941543 8.322074 0.8277879  0.1068974 0.8147538
    ##    alpha lambda
    ## 73  0.25   1.55
    ## 32 x 1 sparse Matrix of class "dgCMatrix"
    ##                     s0
    ## (Intercept) 16.4118037
    ## NRN1.1       1.6540294
    ## NRN2.1       .        
    ## NRN3.1       .        
    ## NRN4.1       0.1324082
    ## NRN5.1      -1.0740807
    ## NRN6.1       .        
    ## NRE1.1      -0.6055750
    ## NRE2.1       .        
    ## NRE3.1       1.6075275
    ## NRE4.1       .        
    ## NRE5.1       1.3672270
    ## NRE6.1      -0.6262439
    ## NRO1.1       0.5325569
    ## NRO2.1       .        
    ## NRO3.1       .        
    ## NRO4.1      -0.6197090
    ## NRO5.1      -0.2591353
    ## NRO6.1       .        
    ## NRA1.1      -0.5097137
    ## NRA2.1       1.0382220
    ## NRA3.1       0.2088765
    ## NRA4.1       0.6173411
    ## NRA5.1       1.1578438
    ## NRA6.1      -0.7563748
    ## NRC1.1       2.1551190
    ## NRC2.1       .        
    ## NRC3.1       .        
    ## NRC4.1       0.5538772
    ## NRC5.1      -1.0104255
    ## NRC6.1       .        
    ## BDI2.1       3.6635260

``` r
# Test sample
bdi.overall.test.enr3 = bdi.overall.test.z[,c(1:31)]
pred.bdi.ov.enr3.test = predict.glmnet(bdi.overall.enr3, newx = data.matrix(bdi.overall.test.enr3))
postResample(pred.bdi.ov.enr3.test, bdi.overall.test$BDI2.2) 
```

    ##      RMSE  Rsquared       MAE 
    ## 9.5521918 0.1285848 7.7599783

``` r
#### Model 4: Demographics and FFM facets
iv.bdi.processed5 = iv.bdi.processed[,c(1:30, 32, 33)]
bdi.overall.train.z5 = bdi.overall.train.z[,c(1:30, 32, 33, 34)]
bdi.overall.enr5 = elastic.net(data = bdi.overall.train.z5, zscored.ivs = iv.bdi.processed5, outcome = bdi.overall.train.outcome)
```

    ##   TrainRMSE TrainRsquared TrainMAE method
    ## 1  11.22624     0.1129719 8.934234 glmnet
    ##   alpha lambda     RMSE  Rsquared      MAE   RMSESD RsquaredSD MAESD
    ## 1  0.25      2 11.22624 0.1129719 8.934234 1.209173 0.08145917  1.01
    ##    alpha lambda
    ## 82  0.25      2
    ## 33 x 1 sparse Matrix of class "dgCMatrix"
    ##                       s0
    ## (Intercept) 16.411803701
    ## NRN1.1       2.448173535
    ## NRN2.1       .          
    ## NRN3.1       0.562779545
    ## NRN4.1       0.186968418
    ## NRN5.1      -0.307905191
    ## NRN6.1       .          
    ## NRE1.1      -0.900577328
    ## NRE2.1       .          
    ## NRE3.1       1.658520833
    ## NRE4.1       .          
    ## NRE5.1       1.068847465
    ## NRE6.1      -1.048380591
    ## NRO1.1       0.325862726
    ## NRO2.1       .          
    ## NRO3.1       .          
    ## NRO4.1      -0.199022392
    ## NRO5.1       .          
    ## NRO6.1       .          
    ## NRA1.1      -0.354956647
    ## NRA2.1       1.119056203
    ## NRA3.1       0.659107988
    ## NRA4.1       0.002045114
    ## NRA5.1       1.074162937
    ## NRA6.1      -0.749524277
    ## NRC1.1       1.807079641
    ## NRC2.1       .          
    ## NRC3.1       .          
    ## NRC4.1       0.116201646
    ## NRC5.1      -1.054962401
    ## NRC6.1       .          
    ## sex          .          
    ## age          .

``` r
# Test sample
bdi.overall.test.enr5 = bdi.overall.test.z[,c(1:30, 32, 33)]
pred.bdi.ov.enr5.test = predict.glmnet(bdi.overall.enr5, newx = data.matrix(bdi.overall.test.enr5))
postResample(pred.bdi.ov.enr5.test, bdi.overall.test$BDI2.2) 
```

    ##       RMSE   Rsquared        MAE 
    ## 9.67447594 0.08580138 8.07032310

``` r
#### Model 5: Every IV
iv.bdi.processed4 = iv.bdi.processed[,c(1:33)]
bdi.overall.train.z4 = bdi.overall.train.z[,c(1:34)]
bdi.overall.enr4 = elastic.net(data = bdi.overall.train.z4, zscored.ivs = iv.bdi.processed4, outcome = bdi.overall.train.outcome)
```

    ##   TrainRMSE TrainRsquared TrainMAE method
    ## 1  10.72399     0.1809086 8.385054 glmnet
    ##   alpha lambda     RMSE  Rsquared      MAE    RMSESD RsquaredSD     MAESD
    ## 1  0.25    1.8 10.72399 0.1809086 8.385054 0.8393502  0.1002692 0.8221222
    ##    alpha lambda
    ## 78  0.25    1.8
    ## 34 x 1 sparse Matrix of class "dgCMatrix"
    ##                      s0
    ## (Intercept) 16.41180370
    ## NRN1.1       1.63838254
    ## NRN2.1       .         
    ## NRN3.1       .         
    ## NRN4.1       0.06089068
    ## NRN5.1      -0.90747821
    ## NRN6.1       .         
    ## NRE1.1      -0.57260675
    ## NRE2.1       .         
    ## NRE3.1       1.42306305
    ## NRE4.1       .         
    ## NRE5.1       1.14020993
    ## NRE6.1      -0.50703792
    ## NRO1.1       0.40810890
    ## NRO2.1       .         
    ## NRO3.1       .         
    ## NRO4.1      -0.50054422
    ## NRO5.1      -0.11820818
    ## NRO6.1       .         
    ## NRA1.1      -0.40483931
    ## NRA2.1       0.95306400
    ## NRA3.1       0.16184606
    ## NRA4.1       0.50577030
    ## NRA5.1       1.08616198
    ## NRA6.1      -0.70202112
    ## NRC1.1       2.00253278
    ## NRC2.1       .         
    ## NRC3.1       .         
    ## NRC4.1       0.42649459
    ## NRC5.1      -0.79477490
    ## NRC6.1       .         
    ## BDI2.1       3.57284737
    ## sex          .         
    ## age          .

``` r
# Test sample
bdi.overall.test.enr4 = bdi.overall.test.z[,c(1:33)]
pred.bdi.ov.enr4.test = predict.glmnet(bdi.overall.enr4, newx = data.matrix(bdi.overall.test.enr4))
postResample(pred.bdi.ov.enr4.test, bdi.overall.test$BDI2.2) 
```

    ##      RMSE  Rsquared       MAE 
    ## 9.4222274 0.1403064 7.7407666

## HAM-D scores

``` r
## Splitting data into training and test sets
set.seed(2020)
hd.index = createDataPartition(neo.depr.hd$hamd2, p = .7, list = F)
hd.overall.train = neo.depr.hd[hd.index, ]
hd.overall.test = neo.depr.hd[-hd.index, ]

nrow(hd.overall.train)
```

    ## [1] 229

``` r
nrow(hd.overall.test)
```

    ## [1] 96

### Ordinary least squares

``` r
#### Pre-tx HAM-D only
hd.overall.lm1 = lm(hamd2 ~ hamd1, data = hd.overall.train)
summ(hd.overall.lm1)
```

    ## MODEL INFO:
    ## Observations: 229
    ## Dependent Variable: hamd2
    ## Type: OLS linear regression 
    ## 
    ## MODEL FIT:
    ## F(1,227) = 22.12, p = 0.00
    ## R² = 0.09
    ## Adj. R² = 0.08 
    ## 
    ## Standard errors: OLS
    ## -----------------------------------------------
    ##                     Est.   S.E.   t val.      p
    ## ----------------- ------ ------ -------- ------
    ## (Intercept)         0.82   1.75     0.47   0.64
    ## hamd1               0.41   0.09     4.70   0.00
    ## -----------------------------------------------

``` r
# Test sample
pred.hd.ov.lm1.test = predict(hd.overall.lm1, hd.overall.test)
postResample(pred.hd.ov.lm1.test, hd.overall.test$hamd2) 
```

    ##      RMSE  Rsquared       MAE 
    ## 7.1632535 0.2102014 5.7046246

``` r
#### Model 1: Pre-tx HAM-D and demographics
hd.overall.lm2 = lm(hamd2 ~ hamd1 + age + sex, data = hd.overall.train)
summ(hd.overall.lm2)
```

    ## MODEL INFO:
    ## Observations: 229
    ## Dependent Variable: hamd2
    ## Type: OLS linear regression 
    ## 
    ## MODEL FIT:
    ## F(3,225) = 8.51, p = 0.00
    ## R² = 0.10
    ## Adj. R² = 0.09 
    ## 
    ## Standard errors: OLS
    ## ------------------------------------------------
    ##                      Est.   S.E.   t val.      p
    ## ----------------- ------- ------ -------- ------
    ## (Intercept)          0.43   2.33     0.19   0.85
    ## hamd1                0.39   0.09     4.47   0.00
    ## age                 -0.01   0.04    -0.18   0.86
    ## sex1                 1.67   0.93     1.80   0.07
    ## ------------------------------------------------

``` r
# Test sample
pred.hd.ov.lm2.test = predict(hd.overall.lm2, hd.overall.test)
postResample(pred.hd.ov.lm2.test, hd.overall.test$hamd2) 
```

    ##      RMSE  Rsquared       MAE 
    ## 7.2367412 0.1797363 5.8007232

``` r
#### Model 2: FFM facets only
hd.overall.train.facets = hd.overall.train[,c(1:30, 32)]
hd.overall.lm3 = lm(hamd2 ~ ., data = hd.overall.train.facets)
summ(hd.overall.lm3)
```

    ## MODEL INFO:
    ## Observations: 229
    ## Dependent Variable: hamd2
    ## Type: OLS linear regression 
    ## 
    ## MODEL FIT:
    ## F(30,198) = 2.33, p = 0.00
    ## R² = 0.26
    ## Adj. R² = 0.15 
    ## 
    ## Standard errors: OLS
    ## ------------------------------------------------
    ##                      Est.   S.E.   t val.      p
    ## ----------------- ------- ------ -------- ------
    ## (Intercept)          1.59   5.73     0.28   0.78
    ## NRN1.1               0.01   0.13     0.09   0.93
    ## NRN2.1               0.04   0.13     0.30   0.76
    ## NRN3.1              -0.13   0.16    -0.84   0.40
    ## NRN4.1               0.07   0.13     0.56   0.58
    ## NRN5.1              -0.12   0.12    -1.04   0.30
    ## NRN6.1              -0.02   0.15    -0.13   0.89
    ## NRE1.1               0.04   0.14     0.29   0.77
    ## NRE2.1              -0.05   0.11    -0.50   0.62
    ## NRE3.1               0.33   0.12     2.70   0.01
    ## NRE4.1              -0.14   0.14    -1.00   0.32
    ## NRE5.1              -0.01   0.12    -0.10   0.92
    ## NRE6.1              -0.27   0.12    -2.25   0.03
    ## NRO1.1               0.14   0.11     1.22   0.22
    ## NRO2.1              -0.05   0.11    -0.44   0.66
    ## NRO3.1              -0.14   0.13    -1.07   0.29
    ## NRO4.1              -0.18   0.14    -1.34   0.18
    ## NRO5.1              -0.06   0.10    -0.64   0.53
    ## NRO6.1              -0.11   0.14    -0.79   0.43
    ## NRA1.1              -0.28   0.11    -2.46   0.01
    ## NRA2.1               0.02   0.12     0.13   0.89
    ## NRA3.1               0.37   0.16     2.25   0.03
    ## NRA4.1               0.08   0.14     0.54   0.59
    ## NRA5.1               0.12   0.13     0.98   0.33
    ## NRA6.1               0.45   0.14     3.28   0.00
    ## NRC1.1              -0.09   0.15    -0.60   0.55
    ## NRC2.1               0.21   0.12     1.82   0.07
    ## NRC3.1              -0.01   0.14    -0.05   0.96
    ## NRC4.1               0.16   0.12     1.34   0.18
    ## NRC5.1              -0.07   0.12    -0.61   0.54
    ## NRC6.1              -0.05   0.13    -0.37   0.71
    ## ------------------------------------------------

``` r
# Test sample
hd.overall.test.facets = hd.overall.test[,c(1:30, 32)]
pred.hd.ov.lm3.test = predict(hd.overall.lm3, hd.overall.test.facets)
postResample(pred.hd.ov.lm3.test, hd.overall.test.facets$hamd2) 
```

    ##       RMSE   Rsquared        MAE 
    ## 8.30509591 0.01483147 6.69441435

``` r
#### Model 3: Pre-tx HAM-D and FFM facets
hd.overall.train.facets2 = hd.overall.train[,c(1:32)]
hd.overall.lm4 = lm(hamd2 ~ ., data = hd.overall.train.facets2)
summ(hd.overall.lm4)
```

    ## MODEL INFO:
    ## Observations: 229
    ## Dependent Variable: hamd2
    ## Type: OLS linear regression 
    ## 
    ## MODEL FIT:
    ## F(31,197) = 2.72, p = 0.00
    ## R² = 0.30
    ## Adj. R² = 0.19 
    ## 
    ## Standard errors: OLS
    ## ------------------------------------------------
    ##                      Est.   S.E.   t val.      p
    ## ----------------- ------- ------ -------- ------
    ## (Intercept)         -1.85   5.68    -0.32   0.75
    ## NRN1.1              -0.04   0.13    -0.31   0.76
    ## NRN2.1               0.07   0.13     0.51   0.61
    ## NRN3.1              -0.12   0.15    -0.80   0.42
    ## NRN4.1               0.08   0.13     0.58   0.56
    ## NRN5.1              -0.09   0.12    -0.78   0.44
    ## NRN6.1              -0.08   0.14    -0.56   0.58
    ## NRE1.1              -0.00   0.14    -0.01   0.99
    ## NRE2.1              -0.06   0.11    -0.60   0.55
    ## NRE3.1               0.28   0.12     2.34   0.02
    ## NRE4.1              -0.11   0.13    -0.83   0.41
    ## NRE5.1              -0.00   0.12    -0.00   1.00
    ## NRE6.1              -0.26   0.12    -2.21   0.03
    ## NRO1.1               0.13   0.11     1.20   0.23
    ## NRO2.1              -0.01   0.11    -0.11   0.91
    ## NRO3.1              -0.17   0.13    -1.29   0.20
    ## NRO4.1              -0.16   0.13    -1.23   0.22
    ## NRO5.1              -0.05   0.10    -0.51   0.61
    ## NRO6.1              -0.12   0.13    -0.89   0.37
    ## NRA1.1              -0.26   0.11    -2.37   0.02
    ## NRA2.1              -0.01   0.12    -0.11   0.91
    ## NRA3.1               0.36   0.16     2.26   0.03
    ## NRA4.1               0.08   0.14     0.57   0.57
    ## NRA5.1               0.08   0.12     0.66   0.51
    ## NRA6.1               0.42   0.13     3.11   0.00
    ## NRC1.1              -0.08   0.15    -0.51   0.61
    ## NRC2.1               0.17   0.11     1.52   0.13
    ## NRC3.1              -0.02   0.13    -0.18   0.85
    ## NRC4.1               0.16   0.12     1.30   0.20
    ## NRC5.1              -0.09   0.12    -0.79   0.43
    ## NRC6.1               0.05   0.13     0.42   0.68
    ## hamd1                0.32   0.10     3.32   0.00
    ## ------------------------------------------------

``` r
# Test sample
hd.overall.test.facets2 = hd.overall.test[,c(1:32)]
pred.hd.ov.lm4.test = predict(hd.overall.lm4, hd.overall.test.facets2)
postResample(pred.hd.ov.lm4.test, hd.overall.test.facets2$hamd2) 
```

    ##      RMSE  Rsquared       MAE 
    ## 7.7236973 0.0801128 6.2854568

``` r
#### Model 4: Demographics and FFM facets
hd.overall.train.facets3 = hd.overall.train[,c(1:30, 35, 36, 32)]
hd.overall.lm6 = lm(hamd2 ~ ., data = hd.overall.train.facets3)
summ(hd.overall.lm6)
```

    ## MODEL INFO:
    ## Observations: 229
    ## Dependent Variable: hamd2
    ## Type: OLS linear regression 
    ## 
    ## MODEL FIT:
    ## F(32,196) = 2.41, p = 0.00
    ## R² = 0.28
    ## Adj. R² = 0.17 
    ## 
    ## Standard errors: OLS
    ## ------------------------------------------------
    ##                      Est.   S.E.   t val.      p
    ## ----------------- ------- ------ -------- ------
    ## (Intercept)          2.06   5.89     0.35   0.73
    ## NRN1.1               0.01   0.13     0.07   0.95
    ## NRN2.1               0.04   0.13     0.32   0.75
    ## NRN3.1              -0.08   0.16    -0.54   0.59
    ## NRN4.1               0.10   0.13     0.72   0.47
    ## NRN5.1              -0.17   0.12    -1.38   0.17
    ## NRN6.1              -0.05   0.15    -0.34   0.73
    ## NRE1.1              -0.00   0.14    -0.01   0.99
    ## NRE2.1              -0.02   0.11    -0.22   0.82
    ## NRE3.1               0.35   0.12     2.88   0.00
    ## NRE4.1              -0.13   0.13    -0.97   0.33
    ## NRE5.1               0.03   0.12     0.26   0.80
    ## NRE6.1              -0.28   0.12    -2.32   0.02
    ## NRO1.1               0.15   0.11     1.35   0.18
    ## NRO2.1              -0.05   0.11    -0.48   0.63
    ## NRO3.1              -0.18   0.14    -1.35   0.18
    ## NRO4.1              -0.23   0.14    -1.66   0.10
    ## NRO5.1              -0.04   0.10    -0.45   0.65
    ## NRO6.1              -0.14   0.14    -1.01   0.31
    ## NRA1.1              -0.25   0.11    -2.20   0.03
    ## NRA2.1               0.04   0.12     0.32   0.75
    ## NRA3.1               0.37   0.16     2.28   0.02
    ## NRA4.1               0.10   0.14     0.75   0.46
    ## NRA5.1               0.08   0.13     0.60   0.55
    ## NRA6.1               0.43   0.14     3.12   0.00
    ## NRC1.1              -0.11   0.15    -0.73   0.47
    ## NRC2.1               0.22   0.12     1.87   0.06
    ## NRC3.1               0.02   0.14     0.17   0.87
    ## NRC4.1               0.15   0.12     1.19   0.23
    ## NRC5.1              -0.08   0.12    -0.69   0.49
    ## NRC6.1              -0.08   0.13    -0.59   0.55
    ## sex1                 2.50   1.05     2.39   0.02
    ## age                 -0.01   0.05    -0.24   0.81
    ## ------------------------------------------------

``` r
# Test sample
hd.overall.test.facets3 = hd.overall.test[,c(1:30, 35, 36, 32)]
pred.hd.ov.lm6.test = predict(hd.overall.lm6, hd.overall.test.facets3)
postResample(pred.hd.ov.lm6.test, hd.overall.test$hamd2) 
```

    ##        RMSE    Rsquared         MAE 
    ## 8.511704988 0.006051467 6.797800196

``` r
#### Model 5: Every IV
hd.overall.train2 = hd.overall.train[,c(1:32, 35, 36)]
hd.overall.lm5 = lm(hamd2 ~ ., data = hd.overall.train2)
summ(hd.overall.lm5)
```

    ## MODEL INFO:
    ## Observations: 229
    ## Dependent Variable: hamd2
    ## Type: OLS linear regression 
    ## 
    ## MODEL FIT:
    ## F(33,195) = 2.77, p = 0.00
    ## R² = 0.32
    ## Adj. R² = 0.20 
    ## 
    ## Standard errors: OLS
    ## ------------------------------------------------
    ##                      Est.   S.E.   t val.      p
    ## ----------------- ------- ------ -------- ------
    ## (Intercept)         -1.31   5.85    -0.22   0.82
    ## NRN1.1              -0.04   0.13    -0.33   0.75
    ## NRN2.1               0.07   0.13     0.52   0.61
    ## NRN3.1              -0.08   0.15    -0.52   0.60
    ## NRN4.1               0.09   0.13     0.73   0.47
    ## NRN5.1              -0.13   0.12    -1.12   0.26
    ## NRN6.1              -0.11   0.14    -0.74   0.46
    ## NRE1.1              -0.04   0.14    -0.30   0.77
    ## NRE2.1              -0.04   0.11    -0.33   0.74
    ## NRE3.1               0.30   0.12     2.52   0.01
    ## NRE4.1              -0.10   0.13    -0.80   0.43
    ## NRE5.1               0.04   0.12     0.34   0.74
    ## NRE6.1              -0.27   0.12    -2.28   0.02
    ## NRO1.1               0.14   0.11     1.32   0.19
    ## NRO2.1              -0.02   0.10    -0.15   0.88
    ## NRO3.1              -0.20   0.13    -1.55   0.12
    ## NRO4.1              -0.21   0.14    -1.54   0.13
    ## NRO5.1              -0.03   0.10    -0.33   0.74
    ## NRO6.1              -0.15   0.13    -1.10   0.27
    ## NRA1.1              -0.24   0.11    -2.12   0.04
    ## NRA2.1               0.01   0.12     0.07   0.94
    ## NRA3.1               0.36   0.16     2.28   0.02
    ## NRA4.1               0.10   0.14     0.76   0.45
    ## NRA5.1               0.04   0.12     0.30   0.76
    ## NRA6.1               0.40   0.13     2.97   0.00
    ## NRC1.1              -0.09   0.15    -0.64   0.52
    ## NRC2.1               0.18   0.11     1.57   0.12
    ## NRC3.1               0.00   0.13     0.03   0.98
    ## NRC4.1               0.14   0.12     1.15   0.25
    ## NRC5.1              -0.10   0.12    -0.86   0.39
    ## NRC6.1               0.02   0.13     0.18   0.86
    ## hamd1                0.31   0.10     3.24   0.00
    ## sex1                 2.36   1.03     2.30   0.02
    ## age                 -0.01   0.05    -0.22   0.82
    ## ------------------------------------------------

``` r
pred.hd.ov.lm5.train = predict(hd.overall.lm5, hd.overall.train2)
postResample(pred.hd.ov.lm5.train, hd.overall.train2$hamd2) 
```

    ##      RMSE  Rsquared       MAE 
    ## 5.8954539 0.3191315 4.8367904

``` r
# Test sample
hd.overall.test2 = hd.overall.test[,c(1:32, 35, 36)]
pred.hd.ov.lm5.test = predict(hd.overall.lm5, hd.overall.test2)
postResample(pred.hd.ov.lm5.test, hd.overall.test$hamd2)  
```

    ##       RMSE   Rsquared        MAE 
    ## 7.87687142 0.05765388 6.37990483

### Elastic net regression

``` r
#### Standardizing IVs

# Training set
iv.hd.process = subset(hd.overall.train, select = iv.hd.names)
iv.hd.processed = standardizing(data = iv.hd.process, var.info = hd.info)

hd.overall.train.z = iv.hd.processed
hd.overall.train.z$y = hd.overall.train$hamd2
hd.overall.train.outcome = select(hd.overall.train, hamd2)

# Test set
iv.hd.process.test = subset(hd.overall.test, select = iv.hd.names)
iv.hd.processed.test = standardizing(data = iv.hd.process.test, var.info = hd.info)

hd.overall.test.z = iv.hd.processed.test
hd.overall.test.z$y = hd.overall.test$hamd2
hd.overall.test.outcome = select(hd.overall.test, hamd2)

#### Model 1: Pre-tx HAM-D and demographics only
iv.hd.processed1 = iv.hd.processed[,c(31:33)]
hd.overall.train.z1 = hd.overall.train.z[,c(31:34)]
hd.overall.enr1 = elastic.net(data = hd.overall.train.z1, zscored.ivs = iv.hd.processed1, outcome = hd.overall.train.outcome)
```

    ##   TrainRMSE TrainRsquared TrainMAE method
    ## 1  6.837705     0.1012529 5.838932 glmnet
    ##   alpha lambda     RMSE  Rsquared      MAE    RMSESD RsquaredSD     MAESD
    ## 1     1    0.2 6.837705 0.1012529 5.838932 0.3547775  0.0604019 0.3243021
    ##     alpha lambda
    ## 169     1    0.2
    ## 4 x 1 sparse Matrix of class "dgCMatrix"
    ##                   s0
    ## (Intercept) 6.653765
    ## hamd1       1.856541
    ## sex         1.314136
    ## age         .

``` r
# Test sample
hd.overall.test.enr1 = hd.overall.test.z[,c(31:33)]
pred.hd.ov.enr1.test = predict.glmnet(hd.overall.enr1, newx = data.matrix(hd.overall.test.enr1))
postResample(pred.hd.ov.enr1.test, hd.overall.test$hamd2) 
```

    ##      RMSE  Rsquared       MAE 
    ## 7.1965928 0.1968098 5.7786473

``` r
#### Model 2: FFM facets only
iv.hd.processed2 = iv.hd.processed[,c(1:30)]
hd.overall.train.z2 = hd.overall.train.z[,c(1:30, 34)]
hd.overall.enr2 = elastic.net(data = hd.overall.train.z2, zscored.ivs = iv.hd.processed2, outcome = hd.overall.train.outcome)
```

    ##   TrainRMSE TrainRsquared TrainMAE method
    ## 1  6.775755     0.1175868 5.691824 glmnet
    ##   alpha lambda     RMSE  Rsquared      MAE    RMSESD RsquaredSD     MAESD
    ## 1  0.25   0.95 6.775755 0.1175868 5.691824 0.4450445 0.07397481 0.3465034
    ##    alpha lambda
    ## 61  0.25   0.95
    ## 31 x 1 sparse Matrix of class "dgCMatrix"
    ##                      s0
    ## (Intercept)  8.75982533
    ## NRN1.1       0.04838779
    ## NRN2.1       0.01802274
    ## NRN3.1       .         
    ## NRN4.1       .         
    ## NRN5.1       .         
    ## NRN6.1       .         
    ## NRE1.1       .         
    ## NRE2.1       .         
    ## NRE3.1       0.79695999
    ## NRE4.1      -0.10563219
    ## NRE5.1      -0.04801607
    ## NRE6.1      -1.08723146
    ## NRO1.1       .         
    ## NRO2.1       .         
    ## NRO3.1      -0.29884142
    ## NRO4.1      -0.86151343
    ## NRO5.1      -0.19955145
    ## NRO6.1      -0.04154196
    ## NRA1.1      -0.82451920
    ## NRA2.1       0.04684098
    ## NRA3.1       1.06018489
    ## NRA4.1       .         
    ## NRA5.1       0.35143135
    ## NRA6.1       1.13236771
    ## NRC1.1       .         
    ## NRC2.1       0.70541966
    ## NRC3.1       .         
    ## NRC4.1       0.34802425
    ## NRC5.1       .         
    ## NRC6.1       .

``` r
# Test sample
hd.overall.test.enr2 = hd.overall.test.z[,c(1:30)]
pred.hd.ov.enr2.test = predict.glmnet(hd.overall.enr2, newx = data.matrix(hd.overall.test.enr2))
postResample(pred.hd.ov.enr2.test, hd.overall.test$hamd2) 
```

    ##       RMSE   Rsquared        MAE 
    ## 7.89522026 0.01430932 6.47539330

``` r
#### Model 3: Pre-tx HAM-D and FFM facets
iv.hd.processed3 = iv.hd.processed[,c(1:31)]
hd.overall.train.z3 = hd.overall.train.z[,c(1:31, 34)]
hd.overall.enr3 = elastic.net(data = hd.overall.train.z3, zscored.ivs = iv.hd.processed3, outcome = hd.overall.train.outcome)
```

    ##   TrainRMSE TrainRsquared TrainMAE method
    ## 1  6.595289     0.1620366 5.502284 glmnet
    ##   alpha lambda     RMSE  Rsquared      MAE    RMSESD RsquaredSD     MAESD
    ## 1  0.25      1 6.595289 0.1620366 5.502284 0.4776377 0.08416047 0.4094703
    ##    alpha lambda
    ## 62  0.25      1
    ## 32 x 1 sparse Matrix of class "dgCMatrix"
    ##                       s0
    ## (Intercept)  8.759825328
    ## NRN1.1       .          
    ## NRN2.1       .          
    ## NRN3.1       .          
    ## NRN4.1       .          
    ## NRN5.1       .          
    ## NRN6.1       .          
    ## NRE1.1       .          
    ## NRE2.1      -0.005436968
    ## NRE3.1       0.535890695
    ## NRE4.1      -0.105418879
    ## NRE5.1       .          
    ## NRE6.1      -1.040401235
    ## NRO1.1       .          
    ## NRO2.1       .          
    ## NRO3.1      -0.353905025
    ## NRO4.1      -0.755586865
    ## NRO5.1      -0.037628059
    ## NRO6.1      -0.035011493
    ## NRA1.1      -0.798535341
    ## NRA2.1       .          
    ## NRA3.1       0.910639121
    ## NRA4.1       .          
    ## NRA5.1       0.043903606
    ## NRA6.1       1.049514710
    ## NRC1.1       .          
    ## NRC2.1       0.613662134
    ## NRC3.1       .          
    ## NRC4.1       0.336713746
    ## NRC5.1       .          
    ## NRC6.1       0.085631800
    ## hamd1        1.371430733

``` r
# Test sample
hd.overall.test.enr3 = hd.overall.test.z[,c(1:31)]
pred.hd.ov.enr3.test = predict.glmnet(hd.overall.enr3, newx = data.matrix(hd.overall.test.enr3))
postResample(pred.hd.ov.enr3.test, hd.overall.test$hamd2) 
```

    ##      RMSE  Rsquared       MAE 
    ## 7.3267528 0.1273034 5.9947030

``` r
#### Model 4: Demographics and FFM facets
iv.hd.processed5 = iv.hd.processed[,c(1:30, 32, 33)]
hd.overall.train.z5 = hd.overall.train.z[,c(1:30, 32, 33, 34)]
hd.overall.enr5 = elastic.net(data = hd.overall.train.z5, zscored.ivs = iv.hd.processed5, outcome = hd.overall.train.outcome)
```

    ##   TrainRMSE TrainRsquared TrainMAE method
    ## 1   6.73634     0.1266156 5.632069 glmnet
    ##   alpha lambda    RMSE  Rsquared      MAE    RMSESD RsquaredSD     MAESD
    ## 1  0.25    0.9 6.73634 0.1266156 5.632069 0.4115429 0.07211106 0.3227467
    ##    alpha lambda
    ## 60  0.25    0.9
    ## 33 x 1 sparse Matrix of class "dgCMatrix"
    ##                      s0
    ## (Intercept)  6.42317549
    ## NRN1.1       0.03716171
    ## NRN2.1       .         
    ## NRN3.1       .         
    ## NRN4.1       .         
    ## NRN5.1       .         
    ## NRN6.1       .         
    ## NRE1.1       .         
    ## NRE2.1       .         
    ## NRE3.1       0.87228449
    ## NRE4.1      -0.11194342
    ## NRE5.1       .         
    ## NRE6.1      -1.11580435
    ## NRO1.1       0.02300430
    ## NRO2.1       .         
    ## NRO3.1      -0.44822090
    ## NRO4.1      -1.00151298
    ## NRO5.1      -0.13348780
    ## NRO6.1      -0.12163533
    ## NRA1.1      -0.81178602
    ## NRA2.1       0.06904247
    ## NRA3.1       1.07075153
    ## NRA4.1       .         
    ## NRA5.1       0.27150155
    ## NRA6.1       1.10804626
    ## NRC1.1       .         
    ## NRC2.1       0.69875470
    ## NRC3.1       .         
    ## NRC4.1       0.30272544
    ## NRC5.1       .         
    ## NRC6.1       .         
    ## sex          1.45801856
    ## age          .

``` r
# Test sample
hd.overall.test.enr5 = hd.overall.test.z[,c(1:30, 32, 33)]
pred.hd.ov.enr5.test = predict.glmnet(hd.overall.enr5, newx = data.matrix(hd.overall.test.enr5))
postResample(pred.hd.ov.enr5.test, hd.overall.test$hamd2) 
```

    ##        RMSE    Rsquared         MAE 
    ## 7.993954897 0.008349989 6.518615351

``` r
#### Model 5: Every IV
iv.hd.processed4 = iv.hd.processed[,c(1:33)]
hd.overall.train.z4 = hd.overall.train.z[,c(1:34)]
hd.overall.enr4 = elastic.net(data = hd.overall.train.z4, zscored.ivs = iv.hd.processed4, outcome = hd.overall.train.outcome)
```

    ##   TrainRMSE TrainRsquared TrainMAE method
    ## 1  6.559551     0.1711369 5.462433 glmnet
    ##   alpha lambda     RMSE  Rsquared      MAE    RMSESD RsquaredSD     MAESD
    ## 1   0.5   0.55 6.559551 0.1711369 5.462433 0.4556185     0.0781 0.3920772
    ##    alpha lambda
    ## 94   0.5   0.55
    ## 34 x 1 sparse Matrix of class "dgCMatrix"
    ##                      s0
    ## (Intercept)  6.67135166
    ## NRN1.1       .         
    ## NRN2.1       .         
    ## NRN3.1       .         
    ## NRN4.1       .         
    ## NRN5.1      -0.01839435
    ## NRN6.1       .         
    ## NRE1.1       .         
    ## NRE2.1       .         
    ## NRE3.1       0.61675446
    ## NRE4.1      -0.04600453
    ## NRE5.1       .         
    ## NRE6.1      -1.09118338
    ## NRO1.1       .         
    ## NRO2.1       .         
    ## NRO3.1      -0.48153104
    ## NRO4.1      -0.91594648
    ## NRO5.1       .         
    ## NRO6.1      -0.05051816
    ## NRA1.1      -0.80426341
    ## NRA2.1       .         
    ## NRA3.1       0.95977769
    ## NRA4.1       .         
    ## NRA5.1       .         
    ## NRA6.1       1.06253062
    ## NRC1.1       .         
    ## NRC2.1       0.61400620
    ## NRC3.1       .         
    ## NRC4.1       0.25958072
    ## NRC5.1       .         
    ## NRC6.1       0.02110498
    ## hamd1        1.37055183
    ## sex          1.30316204
    ## age          .

``` r
# Test sample
hd.overall.test.enr4 = hd.overall.test.z[,c(1:33)]
pred.hd.ov.enr4.test = predict.glmnet(hd.overall.enr4, newx = data.matrix(hd.overall.test.enr4))
postResample(pred.hd.ov.enr4.test, hd.overall.test$hamd2) 
```

    ##      RMSE  Rsquared       MAE 
    ## 7.4016368 0.1064288 6.0384361
