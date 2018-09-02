import numpy as np
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc


class Visualizer:


    def __init__(self):
        self.fig = None  # Matplotlib Figure Class Object use as Canvas
        self.axes = None  # Matplotlib Axes Class Object for drawing charts

    def prepare(self, chart_data):
        self.fig, self.axes = plt.subplots(nrows=4, ncols=1, facecolor='w', sharex=True)
        for ax in self.axes:
            ax.get_xaxis().get_major_formatter().set_scientific(False)
            ax.get_yaxis().get_major_formatter().set_scientific(False)

        # Chart 1. Day Chart
        self.axes[0].set_ylabel('Env.')

        # Visualize Volume
        x = np.array(len(chart_data))
        volume = np.array(chart_data)[:, -1].tolist()
        self.axes[0].bar(x, volume, color='b', alpha=0.3)

        # Print candlestick on self.axes[0]
        # OHLC = open, high, low, closead
        ax = self.axes[0].twinx()
        ohlc = np.hstack((x.reshape(-1, 1), np.array(chart_data)[:, 1:-1]))
        candlestick_ohlc(ax, ohlc, colorup='g', colordown='r')
