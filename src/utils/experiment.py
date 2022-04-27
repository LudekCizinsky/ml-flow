# Data structures
import pandas as pd
import numpy as np

# MLflow
import mlflow

# Azure SDK
from azureml.core import Workspace

# CLI
import argparse

# Modelling
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


class Experiment:

  def __init__(self, df):
    self.df = df
    self._parser = argparse.ArgumentParser()
    self._params = None
    self._setup()
   
  def run(self):

    X, y = self._preprocess()

    mlflow.sklearn.autolog()

    pipe = make_pipeline(
      StandardScaler(),
      PolynomialFeatures(include_bias=False),
      LinearRegression()
      )
  
    params = {
        "polynomialfeatures__degree": [i for i in range(5)]
    }

    grid = GridSearchCV(
        pipe,
        param_grid=params,
        scoring='r2',
        cv=TimeSeriesSplit(5),
        verbose=-1
    )

    grid.fit(X, y)
    
  def _setup(self):

    # Setup Azure workspace and endpoint where to log the info
    ws = Workspace.from_config()
    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

    # Parse arguments from CL
    self._parser.add_argument("-exn", "--expname",type=str, default='Unclassified experiment')
    self._params = self._parser.parse_args()

    # Set name of the experiment
    mlflow.set_experiment(self._params.expname) 

  def _preprocess(self):

    df = self.df.dropna()
   
    df = df.drop(labels=["Source_time", "ANM","Non-ANM"], axis=1) 
   
    wd = df.pop("Direction")
    wd_deg = wd.apply(self._dir2deg)
    wv = df["Speed"] 
    wd_rad = wd_deg*np.pi / 180
    df['Wx'] = wv*np.cos(wd_rad)
    df['Wy'] = wv*np.sin(wd_rad)
    
    df = df[df["Lead_hours"] == 1]

    y = np.array(df.pop('Total'))
    X = np.array(df)

    return X, y

  def _dir2deg(self, s):
    """Copied from:
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

