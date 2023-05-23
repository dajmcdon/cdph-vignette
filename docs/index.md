--- 
title: "Forecasting using `{epipredict}` + `{epiprocess}`"
author: "Logan C. Brooks, Daniel J. McDonald, and Ryan J. Tibshirani"
date: "2023-05-23"
site: bookdown::bookdown_site
documentclass: book
bibliography: [book.bib, packages.bib]
url: https://delphi.cmu.edu/
# cover-image: path to the social sharing image like images/cover.jpg
description: |
  This is a longform vignette describing the `epiprocess` and `epipredict`
  R packages.
biblio-style: apalike
csl: chicago-fullnote-bibliography.csl
---

# Introduction

This long-form vignette describes some of the functionality of the
`epipredict` R package, with an eye toward creating various types of
forecasters, from simple baselines to more elaborate customizations.

## Installation

You can install the development version of epipredict from [GitHub](https://github.com/) with:

``` r
# install.packages("remotes")
remotes::install_github("cmu-delphi/epipredict")
```

## Documentation

You can view the complete documentation for the package at <https://cmu-delphi.github.io/epipredict>.


## Goals for `epipredict`

**We currently provide:**

1. A set of basic, easy-to-use forecasters that work out of the box. You can do a reasonable amount of customization on them. For the basic forecasters, we currently provide: 
    * Baseline flat-line forecaster 
    * Autoregressive forecaster
    * Autoregressive (multinomial) classifier
2. A framework for creating custom forecasters out of modular components. There are four types of components:
    * Preprocessor: do things to the raw data before model training that should be reproduced at test time.
    * Trainer: train a model on data, resulting in a fitted model object
    * Predictor: make predictions, using a fitted model object
    * Postprocessor: do things to the predictions before returning

**Target audiences:**

* Basic. Has data, calls forecaster with default arguments.
* Intermediate. Wants to examine changes to the arguments, take advantage of some built in flexibility.
* Advanced. Wants to write their own forecasters. Maybe willing to build up from some components that we write. 

Even the Advanced user should find their task to be relatively easy. Examples of these tasks are illustrated in the [vignettes and articles](https://cmu-delphi.github.io/epipredict).

## Intermediate example

The package comes with some built-in historical data for illustration, but
up-to-date versions of this could be downloaded with the [`{covidcast}` package](https://cmu-delphi.github.io/covidcast/covidcastR/index.html) and processed using [`{epiprocess}`](https://cmu-delphi.github.io/epiprocess/).[^index1]

[^index1]: COVIDcast data and other epidemiological signals for non-Covid related illnesses are available with [`{epidatr}`](https://cmu-delphi.github.io/epidatr), which interfaces directly to Delphi's [Epidata API](https://cmu-delphi.github.io/delphi-epidata/).


```r
library(tidyverse)
library(epipredict)
jhu <- case_death_rate_subset
jhu
#> An `epi_df` object, 20,496 x 4 with metadata:
#> * geo_type  = state
#> * time_type = day
#> * as_of     = 2022-05-31 12:08:25
#> 
#> # A tibble: 20,496 × 4
#>    geo_value time_value case_rate death_rate
#>  * <chr>     <date>         <dbl>      <dbl>
#>  1 ak        2020-12-31      35.9      0.158
#>  2 al        2020-12-31      65.1      0.438
#>  3 ar        2020-12-31      66.0      1.27 
#>  4 as        2020-12-31       0        0    
#>  5 az        2020-12-31      76.8      1.10 
#>  6 ca        2020-12-31      96.0      0.751
#>  7 co        2020-12-31      35.8      0.649
#>  8 ct        2020-12-31      52.1      0.819
#>  9 dc        2020-12-31      31.0      0.601
#> 10 de        2020-12-31      65.2      0.807
#> # ℹ 20,486 more rows
```

To create and train a simple auto-regressive forecaster to predict the death rate two weeks into the future using past (lagged) deaths and cases, we could use the following function.


```r
two_week_ahead <- arx_forecaster(
  jhu, 
  outcome = "death_rate", 
  predictors = c("case_rate", "death_rate"),
  args_list = arx_args_list(
    lags = list(c(0, 1, 2, 3, 7, 14), c(0, 7, 14)),
    ahead = 14
  )
) 
```

In this case, we have used a number of different lags for the case rate, while only using 3 weekly lags for the death rate (as predictors). The result is both a fitted model object which could be used any time in the future to create different forecasts, as well as a set of predicted values (and prediction intervals) for each location 14 days after the last available time value in the data.


```r
two_week_ahead$epi_workflow
#> ══ Epi Workflow [trained] ══════════════════════════════════
#> Preprocessor: Recipe
#> Model: linear_reg()
#> Postprocessor: Frosting
#> 
#> ── Preprocessor ────────────────────────────────────────────
#> 5 Recipe Steps
#> 
#> • step_epi_lag()
#> • step_epi_lag()
#> • step_epi_ahead()
#> • step_naomit()
#> • step_naomit()
#> 
#> ── Model ───────────────────────────────────────────────────
#> 
#> Call:
#> stats::lm(formula = ..y ~ ., data = data)
#> 
#> Coefficients:
#>       (Intercept)    lag_0_case_rate    lag_1_case_rate  
#>        -0.0073358          0.0030365          0.0012467  
#>   lag_2_case_rate    lag_3_case_rate    lag_7_case_rate  
#>         0.0009536          0.0011425          0.0012481  
#>  lag_14_case_rate   lag_0_death_rate   lag_7_death_rate  
#>         0.0003041          0.1351769          0.1471127  
#> lag_14_death_rate  
#>         0.1062473  
#> 
#> ── Postprocessor ───────────────────────────────────────────
#> 5 Frosting Layers
#> 
#> • layer_predict()
#> • layer_residual_quantiles()
#> • layer_add_forecast_date()
#> • layer_add_target_date()
#> • layer_threshold()
```

The fitted model here involved preprocessing the data to appropriately generate lagged predictors, estimating a linear model with `stats::lm()` and then postprocessing the results to be meaningful for epidemiological tasks. We can also examine the predictions.


```r
two_week_ahead$predictions
#> An `epi_df` object, 56 x 6 with metadata:
#> * geo_type  = state
#> * time_type = day
#> * as_of     = 2022-05-31 12:08:25
#> 
#> # A tibble: 56 × 6
#>    geo_value time_value .pred         .pred_distn
#>  * <chr>     <date>     <dbl>              <dist>
#>  1 ak        2021-12-31 0.449 [0.05, 0.95]<q-rng>
#>  2 al        2021-12-31 0.574 [0.05, 0.95]<q-rng>
#>  3 ar        2021-12-31 0.673 [0.05, 0.95]<q-rng>
#>  4 as        2021-12-31 0     [0.05, 0.95]<q-rng>
#>  5 az        2021-12-31 0.679 [0.05, 0.95]<q-rng>
#>  6 ca        2021-12-31 0.575 [0.05, 0.95]<q-rng>
#>  7 co        2021-12-31 0.862 [0.05, 0.95]<q-rng>
#>  8 ct        2021-12-31 1.07  [0.05, 0.95]<q-rng>
#>  9 dc        2021-12-31 2.12  [0.05, 0.95]<q-rng>
#> 10 de        2021-12-31 1.09  [0.05, 0.95]<q-rng>
#> # ℹ 46 more rows
#> # ℹ 2 more variables: forecast_date <date>,
#> #   target_date <date>
```

The results above show a distributional forecast produced using data through the end of 2021 for the 14th of January 2022. A prediction for the death rate per 100K inhabitants is available for every state (`geo_value`) along with a 90% predictive interval. The figure below
displays the forecast for a small handful of states. The vertical black line is the forecast date. The forecast doesn't appear to be particularly good, but our choices above were intended to be illustrative of the functionality rather than optimized for accuracy.

<img src="index_files/figure-html/unnamed-chunk-1-1.png" width="90%" style="display: block; margin: auto;" />

## Contents

The remainder of this book examines this software in more detail, illustrating some of the flexibility that is available.



