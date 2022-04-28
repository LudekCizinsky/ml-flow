# General
import pandas as pd
import os

# Data structure
import pandas as pd

# Utilities
from utils.experiment import Experiment

def main():

  # Load data
  path = os.path.dirname(os.path.abspath(__file__))
  df = pd.read_json(path + "/data/raw/dataset.json", orient="split")

  # Setup the experiment
  experiment = Experiment(
      name='Orkney - dev2',
      folds=[2, 4, 6, 8 , 10],
      df=df
  )

  # Run the experiment
  experiment.run()

if __name__ == '__main__':
  main()

