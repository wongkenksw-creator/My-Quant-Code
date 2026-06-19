import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class SMABacktester():
    def __init__(self, symbol, SMA_S, SMA_L, start, end):
        self.symbol = symbol
        self.SMA_S = SMA_S
        self.SMA_L = SMA_L
        self.start = start
        self.end = end
        self.results = None
        self.get_data()
        
    def get_data(self):
        df = yf.download(self.symbol, start=self.start, end=self.end)[['Close']]
        data = df
        data["returns"] = np.log(data.Close.div(data.Close.shift(1)))
        data["SMA_S"] = data.Close.rolling(self.SMA_S).mean()
        data["SMA_L"] = data.Close.rolling(self.SMA_L).mean()
        data.dropna(inplace=True)
        self.data2 = data
        return data
    
    def test_results(self):
        data = self.data2.copy().dropna()
        data["position"] = np.where(data["SMA_S"] > data["SMA_L"], 1, -1)
        data["strategy"] = data["returns"] * data.position.shift(1)
        data.dropna(inplace=True)
        data["returnsbh"] = data["returns"].cumsum().apply(np.exp)
        data["returnsstrategy"] = data["strategy"].cumsum().apply(np.exp)
        perf = data["returnsstrategy"].iloc[-1]
        outperf = perf - data["returnsbh"].iloc[-1]
        self.results = data
        ret = np.exp(data["strategy"].sum())
        std = data["strategy"].std() * np.sqrt(252)
        return round(perf, 6), round(outperf, 6)
    
    def plot_results(self):
        if self.results is None:
            print("Run the test please")
        else:
            title = "{} | SMA_S={} | SMA_L={}".format(self.symbol, self.SMA_S, self.SMA_L)
            self.results[["returnsbh", "returnsstrategy"]].plot(title=title, figsize=(12, 8))
            plt.show()
    
    def plot_moving_averages(self):
        data = self.data2.copy()
        plt.style.use("seaborn-v0_8")
        data[["Close", "SMA_S", "SMA_L"]].plot(figsize=(12, 8), fontsize=15)
        plt.legend(loc="upper left", fontsize=15)
        plt.title(f"{self.symbol} Price with SMA{self.SMA_S} and SMA{self.SMA_L}", fontsize=15)
        plt.show()
    
    def plot_return_histogram(self):
        data = self.data2.copy()
        ret = data["Close"].pct_change().mul(100).dropna()
        plt.style.use("seaborn-v0_8")
        ret.plot(kind="hist", figsize=(12, 8), bins=100)
        plt.title(f"{self.symbol} Daily Return Distribution", fontsize=15)
        plt.xlabel("Daily Return (%)", fontsize=15)
        plt.ylabel("Frequency", fontsize=15)
        plt.show()