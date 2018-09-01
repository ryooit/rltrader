'''
Agent Class
Agent is an investor that chooses an action {buy, sell, hold}
Portfolio Value(PV) = balance + the number of stocks * price of stocks
'''

import numpy as np


class Agent:
    STATE_DIM = 2

    # Ignore trading expenses
    TRADING_CHARGE = 0
    TRADING_TAX = 0

    # ACTIONS
    ACTION_BUY = 0
    ACTION_SELL = 1
    ACTION_HOLD = 2
    ACTIONS = [ACTION_BUY, ACTION_SELL]
    NUM_ACTIONS = len(ACTIONS)

    def __init__(
            self, environment, min_trading_unit=1, max_trading_unit=2,
            delayed_reward_threshold=0.05):
        self.environment = environment

        self.min_trading_unit = min_trading_unit
        self.max_trading_unit = max_trading_unit
        self.delayed_reward_threshold = delayed_reward_threshold

        # Agent class attributes
        self.initial_balance = 0  # initial cash
        self.balance = 0  # current cash
        self.num_stocks = 0  # the number of stocks
        self.portfolio_value = 0  # balance + num_stocks * price of stocks
        self.base_portfolio_value = 0  # PV before learning
        self.num_buy = 0  # the number of buy
        self.num_sell = 0  # the number of sell
        self.num_hold = 0  # the number of hold
        self.immediate_reward = 0  # immediate reward

        # Agent class status
        self.ratio_hold = 0
        self.ratio_portfolio_value = 0

    def reset(self):  # initialize class attributes
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.base_portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.immediate_reward = 0
        self.ratio_hold = 0
        self.ratio_portfolio_value = 0

    def set_balance(self, balance):  # set initial balance
        self.initial_balance = balance

    def get_states(self):
        self.ratio_hold = self.num_hold / int(self.portfolio_value / self.environment.get_price())
        self.ratio_portfolio_value = self.portfolio_value / self.initial_balance
        return (
            self.ratio_hold,
            self.ratio_portfolio_value
        )

    def decide_action(self, policy_network, sample, epsilon):
        confidence = 0

        # Exploration Decision
        if np.random.rand() < epsilon:
            exploration = True
            action = np.random.randint(self.NUM_ACTIONS)  # random action
        else:
            exploration = False
            probs = policy_network.predict(sample)  # the probabilities of actions
            action = np.argmax(probs)
            confidence = 1 + probs[action]

        return action, confidence, exploration

    def validate_action(self, action):
        validity = True
        if action == Agent.ACTION_BUY:  # check if such position is affordable
            if self.balance < self.environment.get_price() * \
                    (1 + self.TRADING_CHARGE) * self.min_trading_unit:
                validity = False
        elif action == Agent.ACTION_SELL:  # check if there is any stock to sell
            if self.num_stocks <= 0:
                validity = False
        return validity

    def decide_trading_unit(self, confidence):  # The more confident, the more investing
        if np.isnan(confidence):
            return self.min_trading_unit
        added_trading = max(min(
            int(confidence * (self.max_trading_unit - self.min_trading_unit)),
            self.max_trading_unit - self.min_trading_unit
        ), 0)
        return self.min_trading_unit + added_trading

    def act(self, action, confidence):
        if not self.validate_action(action):  # If the action is not validate
            action = Agent.ACTION_HOLD

        curr_price = self.environment.get_price()
        self.immediate_reward = 0

        if action == Agent.ACTION_BUY:  # BUY Case
            trading_unit = self.decide_trading_unit(confidence)
            balance = self.balance - curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            if balance < 0:  # if there is not enough cash, then buy as much as we can
                trading_unit = max(min(
                    int(self.balance / (curr_price * (1 + self.TRADING_CHARGE))), self.max_trading_unit),
                    self.min_trading_unit
                )
            # Calculate final invest_amount with trading charge
            invest_amount = curr_price * (1 + self.TRADING_CHARGE) * trading_unit
            self.balance -= invest_amount
            self.num_stocks += trading_unit
            self.num_buy += 1

        elif action == Agent.ACTION_SELL:  # SELL Case
            trading_unit = self.decide_trading_unit(confidence)
            # Sell as much as we can
            trading_unit = min(trading_unit, self.num_stocks)

            invest_amount = curr_price * (
                    1 - (self.TRADING_TAX + self.TRADING_CHARGE)) * trading_unit
            self.num_stocks -= trading_unit
            self.balance += invest_amount
            self.num_sell += 1

        elif action == Agent.ACTION_HOLD:  # HOLD Case
            self.num_hold += 1

        # Renew Portfolio Value
        self.portfolio_value = self.balance + curr_price * self.num_stocks
        profitloss = (
                (self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value
        )

        # Immediate reward: 1 if there is profit, -1 otherwise
        self.immediate_reward = 1 if profitloss >= 0 else -1

        # Delayed reward
        if profitloss > self.delayed_reward_threshold:  # Success to achieve target value
            delayed_reward = 1
            self.base_portfolio_value = self.portfolio_value
        elif profitloss < -self.delayed_reward_threshold:  # Failure to achieve target value
            delayed_reward = -1
            self.base_portfolio_value = self.portfolio_value
        else:
            delayed_reward = 0
        return self.immediate_reward, delayed_reward
