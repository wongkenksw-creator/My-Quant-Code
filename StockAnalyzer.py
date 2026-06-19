import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class StockAnalyzer:
    def __init__(self, tickers, start_date="2010-01-01", end_date="2021-01-01"):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.stocks = yf.download(tickers, start=start_date, end=end_date)
        self.close = self.stocks.loc[:, "Close"].copy()
    
    def plot_normalized_prices(self):
        normclose = self.close.div(self.close.iloc[0]).mul(100)
        plt.style.use("seaborn-v0_8")
        normclose.plot(figsize=(15, 8), fontsize=12)
        plt.legend(fontsize=12)
        plt.show()
    
    def plot_risk_return(self):
        ret = self.close.pct_change().dropna()
        summary = ret.describe().T.loc[:, ["mean", "std"]]
        summary["mean"] = summary["mean"] * 252
        summary["std"] = summary["std"] * np.sqrt(252)
        
        summary.plot.scatter(x="std", y="mean", figsize=(12, 8), s=50, fontsize=15)
        for i in summary.index:
            plt.annotate(i, xy=(summary.loc[i, "std"] + 0.002, summary.loc[i, "mean"] + 0.002), size=15)
        plt.xlabel("Annual risk (std)", fontsize=15)
        plt.ylabel("Annual return", fontsize=15)
        plt.title("Risk/return", fontsize=25)
        plt.show()
    
    def plot_correlation_heatmap(self):
        ret = self.close.pct_change().dropna()
        plt.figure(figsize=(12, 8))
        sns.set(font_scale=1.4)
        sns.heatmap(ret.corr(), cmap="Reds", annot=True, annot_kws={"size": 15}, vmax=0.6)
        plt.show()