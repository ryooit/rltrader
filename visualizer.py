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
        self.axes[0].bar(x, volume, color='b', alpha=0.3)  # alpha = transparency

        # Print candlestick on self.axes[0]
        # OHLC = open, high, low, close
        ax = self.axes[0].twinx()
        ohlc = np.hstack((x.reshape(-1, 1), np.array(chart_data)[:, 1:-1]))
        candlestick_ohlc(ax, ohlc, colorup='g', colordown='r')

    def plot(self, epoch_str=None, num_epoches=None, epsilon=None, action_list=None,
             actions=None, num_stocks=None, outvals=None, exps=None, learning=None,
             initial_balance=None, pvs=None):
        x = np.arrange(len(actions))  # shared x-coordinate by all charts
        actions = np.array(actions)   # An array of agent actions
        outvals = np.array(outvals)   # An array of outputs from policy network
        pvs_base = np.zeros(len(actions)) + initial_balance

        # Chart 2. Agent Status(action, # of stocks)
        colors = ['g', 'r']
        for actiontype, color in zip(action_list, colors):
            for i in x[actions == actiontype]:
                self.axes[1].axvline(i, color=color, alpha=0.1)  # Change background color according to actions
        self.axes[1].plot(x, num_stocks, '-k')  # Draw the number of holding stocks by the black line(-k)

        # Chart 3. Print & Exploration the policy network
        for exp_idx in exps:
            self.axes[2].axvline(exp_idx, color='y')  # Color background with yellow for exploration
        for idx, outval in zip(x, outvals):
            color = 'white'
            if outval.argmax() == 0:  # BUY = Red
                color = 'r'
            elif outval.argmax() == 1:  # SELL = Blue
                color = 'b'
            self.axes[2].axvline(idx, color=color, alpha=0.1)  # Color background with red or blue according to action
        styles = ['.r', '.b']  # Red and Blue dots
        for action, style in zip(action_list, styles):
            self.axes[2].plot(x, outvals[:, action], style)

        # Chart 4. Portfolio Value
        self.axes[3].axhline(initial_balance, linestyle='-', color='gray')
        self.axes[3].fill_between(x, pvs, pvs_base, where=pvs > pvs_base, facecolor='r', alpha=0.1)
        self.axes[3].fill_between(x, pvs, pvs_base, where=pvs < pvs_base, facecolor='b', alpha=0.1)
        self.axes[3].plot(x, pvs, '-k')
        for learning_idx, delayed_reward in learning:  # Mark learning point
            if delayed_reward > 0:
                self.axes[3].axvline(learning_idx, color='r', alpha=0.1)
            else:
                self.axes[3].axvline(learning_idx, color='b', alpha=0.1)

        # Ratio & Exploration rate
        self.fig.suptitle('Epoch %s/%s (e=%.2f)' % (epoch_str, num_epoches, epsilon))
        # Manage canvas layout
        plt.tight_layout()
        plt.subplots_adjust(top=.9)

    def clear(self, xlim):
        for ax in self.axes[1:]:
            ax.cla()  # Erase green chart
            ax.relim()  # Initialize limit
            ax.autoscale()  # Reset scale

        self.axes[1].set_ylabel('Agent')
        self.axes[2].set_ylabel('PG')
        self.axes[3].set_ylabel('PV')
        for ax in self.axes:
            ax.set_xlim(xlim)  # Reset x-coordinate limit
            ax.get_xaxis().get_major_formatter().set_scientific(False)
            ax.get_yaxis().get_major_formatter().set_scientific(False)
            ax.ticklabel_format(useOffset=False)  # Divide x-coordinate evenly

    def save(self, path):
        plt.savefig(path)