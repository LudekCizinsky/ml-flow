import pandas as pd
import mlflow
import os
path = os.path.dirname(os.path.abspath(__file__))

from azureml.core import Workspace
ws = Workspace.from_config()
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

## NOTE: Optionally, you can use the public tracking server.  Do not use it for data you cannot afford to lose. See note in assignment text. If you leave this line as a comment, mlflow will save the runs to your local filesystem.
# mlflow.set_tracking_uri("http://training.itu.dk:5000/")

mlflow.set_experiment("luci - First try: Linear regression with polynomial features")

# Import some of the sklearn modules you are likely to use.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


os.environ['logcl'] = 'blue'

# Add path to the external scripts
import sys
path = os.path.dirname(os.path.abspath(__file__))[:-3] + 'ext/time-series-prediction/src/'
sys.path.insert(0, path)

# Preprocessing
from scripts import preprocess, eda

def main():

  with mlflow.start_run(run_name="Test"):
      
      path = os.path.dirname(os.path.abspath(__file__))
      df = pd.read_json(path + "/data/raw/dataset.json", orient="split")
      df = df.dropna()
      print(df)

      return

      pipeline = Pipeline([('scaler', StandardScaler()), ('LR', LinearRegression())])

      metrics = [
          ("MAE", mean_absolute_error, []),
      ]

      X = df[["Speed"]]
      y = df["Total"]

      number_of_splits = 5
   
      for train, test in TimeSeriesSplit(number_of_splits).split(X,y):
          pipeline.fit(X.iloc[train],y.iloc[train])
          predictions = pipeline.predict(X.iloc[test])
          truth = y.iloc[test]

          from matplotlib import pyplot as plt 
          plt.plot(truth.index, truth.values, label="Truth")
          plt.plot(truth.index, predictions, label="Predictions")
          #plt.show()
          
          # Calculate and save the metrics for this fold
          for name, func, scores in metrics:
              score = func(truth, predictions)
              scores.append(score)
      
      # Log a summary of the metrics
      for name, _, scores in metrics:
          mean_score = sum(scores)/number_of_splits
          mlflow.log_metric(f"mean_{name}", mean_score)


if __name__ == '__main__':
  main()

