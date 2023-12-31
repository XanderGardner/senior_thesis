import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import timedelta
import random
import math

BIN_UNIT_TIME_DIFF = 60
BIN_UNIT_TIME_DIFF_STR = '1min'

### the announcement_model class represent an active announcement event and provides analysis and modeling functions
class announcement_model:

  ##### constructor functions #####

  def __init__(self, ticker, announcement_time, taq_df):
    self.ticker = ticker
    self.time_n = announcement_time
    self.time_h = self.time_n + timedelta(minutes=1)
    self.time_c = self.time_n + timedelta(minutes=15)
    self.df = self.construct_df(taq_df)

    self.df_bin = self.df.resample(BIN_UNIT_TIME_DIFF_STR, on='DATETIME').agg({
    'TICKER': 'first',
    'PRICE': 'mean',
    'BID': 'mean',
    'ASK': 'mean'
    }).reset_index()
    self.returns_df = self.construct_returns_df()

    self.mu_hat, self.sigma_squared_hat, self.j_hat = self.estimate_single_slow_jump_model()


  def construct_df(self, taq_df):

    def lee_ready_algorithm(row):
      mid_price = (row['BID'] + row['ASK']) / 2

      if row['PRICE'] > mid_price:
          return 'B'  # Buyer-initiated trade
      elif row['PRICE'] < mid_price:
          return 'S'  # Seller-initiated trade
      else:
          return 'N'  # Indeterminate
      
    taq_df['MID_PRICE'] = (taq_df['BID'] + taq_df['ASK']) / 2
    taq_df['TRADE_DIRECTION'] = taq_df.apply(lee_ready_algorithm, axis=1)
    taq_df.sort_values(by='DATETIME', inplace=True)
    return taq_df
  
  def construct_returns_df(self):
    returns_df = self.df_bin[(self.df_bin['DATETIME'] >= self.time_c) | (self.df_bin['DATETIME'] <= self.time_n)]
    returns_df['TIME_DIFF'] = returns_df['DATETIME'].diff().dt.total_seconds()
    returns_df['LOG_RETURN'] = np.log(returns_df['PRICE'] / returns_df['PRICE'].shift(1))
    
    # drop nans and time differences that are too large
    returns_df.dropna(inplace=True)
    returns_df = returns_df[(returns_df['TIME_DIFF'] == BIN_UNIT_TIME_DIFF)]
    return returns_df
  
  # using the returns dataframe, calculates the MLE for mu, sigma, and J
  def estimate_single_slow_jump_model(self):
    rs = list(self.returns_df['LOG_RETURN'])
    n = len(rs)
    
    mu_hat_r = sum(rs) / n
    sigma_squared_hat_r = sum([(r - mu_hat_r) ** 2 for r in rs]) / n

    mu_hat = mu_hat_r + sigma_squared_hat_r/2
    sigma_squared_hat = sigma_squared_hat_r

    # get closest times and stock prices around the announcement and convergence
    df_bin_before = self.df_bin[(self.df_bin['DATETIME'] <= self.time_n)]['DATETIME'].idxmax()
    df_bin_after = self.df_bin[(self.df_bin['DATETIME'] >= self.time_c)]['DATETIME'].idxmin()
    t_n = self.df_bin.loc[df_bin_before]['DATETIME']
    s_t_n = self.df_bin.loc[df_bin_before]['PRICE']
    t_c = self.df_bin.loc[df_bin_after]['DATETIME']
    s_t_c = self.df_bin.loc[df_bin_after]['PRICE']
    delta_t = (t_c - t_n).total_seconds() / BIN_UNIT_TIME_DIFF
    j_hat = np.log(s_t_c / s_t_n) - (mu_hat - (sigma_squared_hat * delta_t / 2))

    return float(mu_hat), float(sigma_squared_hat), float(j_hat) 




  ##### simulation #####

  # given a list of datetimes and parameters S(0)
  # returns the simulated stock prices for each time according to estimated parameters
  def simulate_single_slow_jump_model(self, times, s_0):
    out = [s_0]
    for i in range(1, len(times)):
      # simulate movement from i-1 to i
      delta_t = (times[i] - times[i-1]).total_seconds() / BIN_UNIT_TIME_DIFF
      wiener_t = random.gauss(0, delta_t ** 0.5)
      exp = (self.mu_hat - self.sigma_squared_hat / 2)*delta_t + ((self.sigma_squared_hat ** 0.5)*wiener_t)
      if (times[i] <= self.time_n and times[i-1] <= self.time_n) or (times[i] >= self.time_c and times[i-1] >= self.time_c):
        val = out[-1] * math.exp(exp)
      else:
        # include jump component
        val = out[-1] * math.exp(exp + self.j_hat)
      out.append(val)
    return out


  ##### plotting #####

  def plot_trades_and_quotes(self):
    # Line plot for prices over time
    plt.plot(self.df['DATETIME'], self.df['PRICE'], label='Price', color='black')

    # Scatter plot for bids and asks
    plt.plot(self.df['DATETIME'], self.df['BID'], label='Bid', color='green')
    plt.plot(self.df['DATETIME'], self.df['ASK'], label='Ask', color='red')

    plt.title('Trade Data Over Time')
    plt.xlabel('Datetime')
    plt.ylabel('Price / Bid / Ask')
    plt.legend()
    plt.grid(True)
    plt.show()

  def plot_trade_prices(self):
    # Line plot for prices over time
    plt.plot(self.df['DATETIME'], self.df['PRICE'], label='Trade Price', color='black')

    plt.title('Trade Prices Over Time')
    plt.xlabel('Datetime')
    plt.ylabel('Trade Price')
    plt.legend()
    plt.grid(True)
    plt.show()

  def plot_midpoint_prices(self):
    # Line plot for prices over time
    plt.plot(self.df['DATETIME'], self.df['MID_PRICE'], label='Mid Price', color='black')

    plt.title('Quote Mid Prices Over Time')
    plt.xlabel('Datetime')
    plt.ylabel('Quote Mid Price')
    plt.legend()
    plt.grid(True)
    plt.show()

  
  # Line plot for prices, bids, and asks over time aggregated in bins
  def plot_bin(self):
    plt.plot(self.df_bin['DATETIME'], self.df_bin['PRICE'], label='Trade', color='black')
    plt.plot(self.df_bin['DATETIME'], self.df_bin['BID'], label='Bid', color='green')
    plt.plot(self.df_bin['DATETIME'], self.df_bin['ASK'], label='Ask', color='red')
    plt.axvline(x=self.time_n, color='orange', linestyle='--', label='Announcement Time')
    plt.axvline(x=self.time_h, color='blue', linestyle='--', label='Human Reaction Time')
    plt.axvline(x=self.time_c, color='violet', linestyle='--', label='Convergence Time')

    plt.title(f'{self.ticker} TAQ Data in 1 Minute Bins')
    plt.xlabel('Datetime')
    plt.ylabel('USD')
    plt.legend()
    plt.grid(True)
    plt.show()

  def plot_single_slow_jump_model_simulation(self):
    valid_df = self.df_bin[(self.df_bin['DATETIME'] >= self.time_c) | (self.df_bin['DATETIME'] <= self.time_n)]
    times = valid_df['DATETIME'].tolist()
    s_0 = float(valid_df.iloc[0]['PRICE'])
    simulated_prices = self.simulate_single_slow_jump_model(times, s_0)
    
    plt.plot(times, simulated_prices, label='Example Simulated Single Slow Jump Model with Same S(0)', color='blue')
    plt.plot(self.df_bin['DATETIME'], self.df_bin['PRICE'], label='Trade', color='black')
    plt.axvline(x=self.time_n, color='orange', linestyle='--', label='Announcement Time')
    plt.axvline(x=self.time_h, color='blue', linestyle='--', label='Human Reaction Time')
    plt.axvline(x=self.time_c, color='violet', linestyle='--', label='Convergence Time')

    plt.title(f'{self.ticker} TAQ Data in 1 Minute Bins')
    plt.xlabel('Datetime')
    plt.ylabel('USD')
    plt.legend()
    plt.grid(True)
    plt.show()



### the model_store class represents the total collection of all annoucement_models and provides cummulative analysis and access
class model_store:

  ##### constructor functions #####

  def __init__(self):
    self.models = []
    self.construct_models()

  # helper function for constructor to create the initial models
  def construct_models(self):
    # load from announcement_taq and select only valid floats
    taq_all = pd.read_csv("./announcement_taq_csv/taq_all.csv")
    taq_all['DATETIME'] = pd.to_datetime(taq_all['DATETIME'])
    taq_all['TICKER'] = taq_all['TICKER'].astype('string')
    
    # load active announcements from active_market_anns_map.pkl
    with open("./market_anns_storage/active_market_anns_map.pkl", "rb") as file:
      market_anns_map = pickle.load(file)

    # create announcement_models for each announcement
    for year, dict1 in market_anns_map.items():
      for ticker, dict2 in dict1.items():
        for date, time in dict2.items():
          
          curr_datetime =  pd.to_datetime(date + ' ' + time, format='%Y-%m-%d %H:%M:%S.%f')

          curr_df = taq_all[
                      (taq_all['TICKER'] == ticker) & 
                      (taq_all['DATETIME'] >= curr_datetime - pd.Timedelta(minutes=30)) &
                      (taq_all['DATETIME'] <= curr_datetime + pd.Timedelta(minutes=30))
                      ]
          if len(curr_df) > 0:
            self.models.append(announcement_model(ticker, curr_datetime, curr_df))

  ##### getter function #####

  def get_models(self):
    return self.models

  ##### analysis functions #####
  
