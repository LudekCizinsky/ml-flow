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
  experiment = Experiment(df)

  # Run the experiment
  experiment.run() 

  # Final info


if __name__ == '__main__':
  main()
