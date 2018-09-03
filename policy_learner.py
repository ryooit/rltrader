import os  # Create folder, file path
import locale  # Currency string format
import time  # Get time & time string format
import datetime
import numpy as np
import settings  # Investment and logging settings
from environment import Environment
from agent import Agent
from policy_network import PolicyNetwork
from visualizer import Visualizer


logger = logging.getLogger(__name__)


class PolicyLearner:

    def __init__(self, stock_code, chart_data, training_data=None,
                 min_trading_unit=1, max_trading_unit=2,
                 delayed_reward_threshold=.05, lr=0.01):
        self.stock_code = stock_code
        self.chart_data = chart_data
        self.environment = Environment(chart_data)
        self.agent = Agent(self.environment, min_trading_unit=min_trading_unit, max_trading_unit=max_trading_unit,
                           delayed_reward_threshold=delayed_reward_threshold)
        self.training_data = training_data
        self.sample = None
        self.training_data_idx = -1

        self.num_features = self.training_data.shape[1] + self.agent.STATE_DIM
        self.policy_network = PolicyNetwork(input_dim=self.num_features, output_dim=self.agent.NUM_ACTIONS, lr=lr)
        self.visualizer = Visualizer()

    def reset(self):
        self.sample = None
        self.training_data_idx = -1
