# Time
from datetime import datetime

# Data structures
import pandas as pd
import numpy as np

# Plotting
import seaborn as sns
sns.set_theme(style="darkgrid")
sns.set(rc={"xtick.bottom" : True, "ytick.left" : True})
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Tracking
import mlflow

# Azure SDK
from azureml.core import Workspace

# Modelling
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor


class Experiment:

  def __init__(self, name, folds, df):
    self.name = name
    self.df = df
    self.folds = folds
    self._preprocess_info = None
    self._figures = None
    self._df_trf = None
    self._scoring = [
        'r2',
        'neg_mean_absolute_error',
        'neg_mean_squared_error'
    ]
    self._setup()
  
  def run(self):

    X, y = self._preprocess()
    self._eda()

    self._run_lr(X, y)
    self._run_dt(X, y)
    self._run_ffnn(X, y)

  def _setup(self):

    # Setup Azure workspace and endpoint where to log the info
    ws = Workspace.from_config()
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

    # Set name of the experiment
    mlflow.set_experiment(self.name)

  def _run_ffnn(self, X, y):

    for f in self.folds:
      run_name = f"FFNN: {datetime.now().strftime('%Y-%m-%d %H:%M')} | {f} folds"
      with mlflow.start_run(run_name=run_name):

        mlflow.log_params(self._preprocess_info)
        for fig, fname in self._figures:
          mlflow.log_figure(fig, fname)

        mlflow.sklearn.autolog()

        pipe = make_pipeline(
            StandardScaler(),
            SelectKBest(f_regression),
            MLPRegressor(random_state=42, shuffle=False)
            )

        params = {
            "selectkbest__k": [1, 2, 3],
            "mlpregressor__hidden_layer_sizes": [[25, 30], [10, 15]],
            "mlpregressor__activation": ['relu', 'tanh'],
            "mlpregressor__learning_rate_init": [.01, .001, .0001, 0.00001],
            "mlpregressor__max_iter": [800, 1000]
        }

        grid = GridSearchCV(
            pipe,
            param_grid=params,
            scoring=self._scoring,
            refit='r2',
            cv=TimeSeriesSplit(f),
            verbose=-1,
            n_jobs=-1
        )

        grid.fit(X, y)

  def _run_dt(self, X, y):

    for f in self.folds:
      run_name = f"DT: {datetime.now().strftime('%Y-%m-%d %H:%M')} | {f} folds"
      with mlflow.start_run(run_name=run_name):

        mlflow.log_params(self._preprocess_info)
        for fig, fname in self._figures:
          mlflow.log_figure(fig, fname)

        mlflow.sklearn.autolog()

        pipe = make_pipeline(
          StandardScaler(),
          SelectKBest(f_regression),
          DecisionTreeRegressor(random_state=42)
        )
      
        params = {
          "selectkbest__k": [1, 2, 3],
          "decisiontreeregressor__splitter": ["best", "random"],
          "decisiontreeregressor__max_depth": [i for i in range(4, 20, 2)],
          "decisiontreeregressor__max_features": ["auto", "sqrt", "log2"]
        }

        grid = GridSearchCV(
            pipe,
            param_grid=params,
            scoring=self._scoring,
            refit='r2',
            cv=TimeSeriesSplit(f),
            verbose=-1,
            n_jobs=-1
        )

        grid.fit(X, y)

  def _run_lr(self, X, y):

    for f in self.folds:
      run_name = f"LR: {datetime.now().strftime('%Y-%m-%d %H:%M')} | {f} folds"
      with mlflow.start_run(run_name=run_name):
        
        mlflow.log_params(self._preprocess_info)
        for fig, fname in self._figures:
          mlflow.log_figure(fig, fname)

        mlflow.sklearn.autolog()

        pipe = make_pipeline(
          StandardScaler(),
          PolynomialFeatures(include_bias=False),
          SelectKBest(f_regression),
          LinearRegression()
          )
      
        params = {
            "selectkbest__k": [1, 2, 3],
            "polynomialfeatures__degree": [i for i in range(2, 12, 2)]
        }

        grid = GridSearchCV(
            pipe,
            param_grid=params,
            refit='r2',
            cv=TimeSeriesSplit(f),
            verbose=-1,
            n_jobs=-1
        )

        grid.fit(X, y) 
  
  def _eda(self):
    
    # Relation between produced power and independent variables
    fig1, ax = plt.subplots(nrows=1, ncols=3, figsize=(25,7))
    sns.scatterplot(x="Speed", y="Total", ax=ax[0], data=self._df_trf, alpha=0.3)
    sns.scatterplot(x="Wx", y="Total", ax=ax[1], data=self._df_trf, alpha=0.3)
    sns.scatterplot(x="Wy", y="Total", ax=ax[2], data=self._df_trf, alpha=0.3)
    
    self._figures = [
      (fig1, 'power_vs_vars.png',),
    ]

  def _preprocess(self):
    
    info = dict()
    
    info['Unique values for Lead hours column'] = list(self.df['Lead_hours'].unique())

    r1 = self.df.shape[0]
    df = self.df.dropna()
    r2 = df.shape[0]
    info['# of dropped rows'] = r1 - r2

    df = df[df["Lead_hours"] == 1]
    info['Lead hours info'] = 'Only records with lead hour 1 are used.'

    cols_to_drop = ["Source_time", "ANM","Non-ANM", "Lead_hours"]
    df = df.drop(labels=cols_to_drop, axis=1)
    info['Dropped columns'] = cols_to_drop
   
    wd = df.pop("Direction")
    wd_deg = wd.apply(self._dir2deg)
    wv = df["Speed"] 
    wd_rad = wd_deg*np.pi / 180
    df['Wx'] = wv*np.cos(wd_rad)
    df['Wy'] = wv*np.sin(wd_rad)
    info['Direction transformation'] = 'Direction was transformed into a wind vector with two components'
    
    info['# of rows of the preprocessed data'] = df.shape[0]

    self._preprocess_info = info
    
    self._df_trf = df

    y = np.array(df['Total'])
    X = np.array(df[['Speed', 'Wx', 'Wy']])

    return X, y

  def _dir2deg(self, s):
    """Code from:
    Https://codegolf.stackexchange.com/questions/54755/convert-a-point-of-the-compass-to-degree
    """

    if 'W' in s:
        s = s.replace('N','n')
    a=(len(s)-2)/8
    if 'b' in s:
        a = 1/8 if len(s)==3 else 1/4
        return (1-a)*f(s[:-2])+a* self._dir2deg(s[-1])
    else:
        if len(s)==1:
            return 'NESWn'.find(s)*90
        else:
            return (self._dir2deg(s[0]) + self._dir2deg(s[1:]))/2

