import pandas as pd
import numpy as np

def load_chart_data(fpath):
    chart_data = pd.read_csv(fpath, thousands=',', header=None)
    chart_data.columns = ['data', 'open', 'high', 'low', 'close', 'volume']
    return chart_data
