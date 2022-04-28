## Intro
This report summarizes my process of finding an optimal model for prediction of
power production based on given weather data. Further, I also elaborate on the
pros and cons of using `mlflow` as a tool to track and deploy machine learning
experiments. This will also be put in the context of other available options.

## Choice of model and evaluation metrics
### Problem description and strategy
I want to start by stating the overall goal of this project:

> To find an optimal model for prediction of power production based on the
  provided `static` dataset while using `mlflow` as a main tool to track the whole
  process. Last but not the least, to deploy the best model on Azure VM.

### Exploratory data analysis and preprocessing
My first step was to examine the given dataset. From the available columns,
I decided to use the following as features:

- `Speed`
- `Direction`

I did not consider `Lead_hours` as a feature since for all the records, it was 1. Similarly, the remaining columns did not provide any useful information and therefore I also dropped them to free space in memory. In addition, the raw dataset included `254,967`, yet after removing rows with missing values, I ended up with a dataset which has only `1318` records. Finally, wind direction was encoded as a `string` which from a model perspective might not be telling the full information. Therefore, I decided to transform this wind direction into a wind vector. I already explained this step in detail [here](https://github.com/LudekCizinsky/time-series-prediction/blob/main/report.md#preprocessing). Thus, to summarize, I ended up after preprocessing with the following dataset:
- `1318` records
- `4 columns`: Speed, Wx, Wy, Total where Wx and Wy are components of the wind
  vector

More importantly, I wanted to further understand the features that are at my
disposal, therefore, I created a scatter plot which shows a relationship between the predicted value (`Total`) and predictors:

![](figures/power_vs_var.png)

We can see that the relationship between power and speed seems quite `linear`.
However, this can not be said about the other two independent variables where
we rather see a non-linear relationship. In addition, we can see that the units
of the features are different. Therefore, there are following implications
for the modelling part:

- There is a need for feature `normalization` - for this I will use `Standard
  scaler` from the `sklearn` library

- Given a very small number of data points relative to the number of features,
  the model might over-fit due to the `curse of dimensionality`. Therefore,
  I should start with models which are less complex and use them as a valid
  baseline and then try to increase the complexity - e.g. start with `Linear regression` 
  and then try neural network based model such as `Feed forward neural network`.

### Choice of metrics and experiment methodology

## How the evaluation errors change in terms of the cross-validation parameters

## Choice of best performing model

## Reproducibility
### MLflow

### Option 1

## Summary 


