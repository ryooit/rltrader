'''
Environment Class
: manages the chart data that an agent is going to invest
'''


class Environment:
    PRICE_IDX = 4  # index of the end price

    def __init__(self, chart_data=None):
        self.chart_data = chart_data
        self.observation = None
        self.idx = -1

    def reset(self):  # Reset the index
        self.observation = None
        self.idx = -1

    def observe(self):  # if there is any data return the observation
        if len(self.chart_data) > self.idx + 1:
            self.idx += 1
            self.observation = self.chart_data.iloc[self.idx]
            return self.observation
        return None

    def get_price(self):  # return the end price
        if self.observation is not None:
            return self.observation[self.PRICE_IDX]
        return None
