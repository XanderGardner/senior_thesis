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
    }).reset_index().dropna()
    self.returns_df = self.construct_returns_df()

    self.basic_mu1_hat, self.basic_mu2_hat, self.basic_sigma_squared_hat = self.estimate_slow_price_step_model()
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

    if math.isnan(mu_hat):
      raise ValueError(f"model failed: mu_hat is {mu_hat}")
    if math.isnan(sigma_squared_hat):
      raise ValueError(f"model failed: sigma_squared_hat is {sigma_squared_hat}")
    if math.isnan(j_hat):
      raise ValueError(f"model failed: j_hat is {j_hat}")
    
    return float(mu_hat), float(sigma_squared_hat), float(j_hat) 

  # get net dollar position of high frequency traders
  def get_hft_dollar_position(self):
    # get df of hft trades
    hft_df = self.df[(self.df['DATETIME'] > self.time_n) & (self.df['DATETIME'] < self.time_h)]
    hft_df['DOLLAR_VOLUME'] = hft_df["SIZE"] * hft_df["PRICE"]

    # find dollar volume
    hft_dollar_volume = hft_df[hft_df["TRADE_DIRECTION"] == "B"]["DOLLAR_VOLUME"].sum()
    hft_dollar_volume -= hft_df[hft_df["TRADE_DIRECTION"] == "S"]["DOLLAR_VOLUME"].sum()
    return hft_dollar_volume

  # get dollar volume of trades by high frequency traders
  def get_hft_dollar_volume(self):
    # get df of hft trades
    hft_df = self.df[(self.df['DATETIME'] > self.time_n) & (self.df['DATETIME'] < self.time_h)]
    hft_df['DOLLAR_VOLUME'] = hft_df["SIZE"] * hft_df["PRICE"]

    # find dollar volume
    hft_dollar_volume = hft_df[hft_df["TRADE_DIRECTION"] == "B"]["DOLLAR_VOLUME"].sum()
    hft_dollar_volume += hft_df[hft_df["TRADE_DIRECTION"] == "S"]["DOLLAR_VOLUME"].sum()
    return hft_dollar_volume
  
  # get trade volume of trades by high frequency traders
  def get_hft_trade_volume(self):
    # get df of hft trades
    hft_df = self.df[(self.df['DATETIME'] > self.time_n) & (self.df['DATETIME'] < self.time_h)]

    # find trade volume
    hft_trade_volume = (hft_df['TRADE_DIRECTION'] == 'B').sum()
    hft_trade_volume += (hft_df['TRADE_DIRECTION'] == 'S').sum()
    return hft_trade_volume
  
  # get profit for hft under detailed model
  def get_hft_profit(self):
    hft_dollar_volume = self.get_hft_dollar_position()
    profit = hft_dollar_volume * math.exp(self.j_hat) - hft_dollar_volume
    return profit

  # define contribution to be price change caused by HFT
  def get_hft_price_change(self):
    hft_df = self.df[(self.df['DATETIME'] > self.time_n) & (self.df['DATETIME'] < self.time_h)]
    if len(hft_df.index) == 0:
      return 0
    return float(hft_df.iloc[-1]["PRICE"]) - float(hft_df.iloc[0]["PRICE"])

  # modeled price change according to jump
  def get_jump_price_change(self):
    hft_df = self.df[(self.df['DATETIME'] > self.time_n)]
    return (float(hft_df.iloc[0]["PRICE"]) * math.exp(self.j_hat)) - float(hft_df.iloc[0]["PRICE"])

  # contribution provided / percent jump
  def get_hft_contribution(self):
    return self.get_hft_price_change() / self.get_jump_price_change()

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

    plt.title(f'{self.ticker} Trade Prices in 1 Minute Bins')
    plt.xlabel('Datetime')
    plt.ylabel('USD')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))
    plt.grid(True)
    plt.show()


  ##### basic model function #####
    
  # contructor function for estimating model parameters
  def estimate_slow_price_step_model(self):
  # return basic_mu1_hat, basic_mu2_hat, and basic_sigma_squared_hat
    prices_before = list(self.df[self.df['DATETIME'] < self.time_n]['PRICE'])
    prices_after = list(self.df[self.df['DATETIME'] >= self.time_c]['PRICE'])
    
    basic_mu1_hat = sum(prices_before) / len(prices_before)
    basic_mu2_hat = sum(prices_after) / len(prices_after)
    m = len(prices_before) + len(prices_after)
    basic_sigma_squared_hat = sum([(price - basic_mu1_hat) ** 2 for price in prices_before]) / m
    basic_sigma_squared_hat += sum([(price - basic_mu2_hat) ** 2 for price in prices_after]) / m
    
    return basic_mu1_hat, basic_mu2_hat, basic_sigma_squared_hat

  def plot_slow_price_step_model(self):
    times_before = [self.df_bin['DATETIME'].iloc[0], self.time_n]
    prices_before = [self.basic_mu1_hat, self.basic_mu1_hat]
    times_after = [self.df_bin['DATETIME'].iloc[-1], self.time_c]
    prices_after = [self.basic_mu2_hat, self.basic_mu2_hat]
    
    plt.plot(times_before, prices_before, label='Slow Price Step Model Expectation', color='blue')
    plt.plot(times_after, prices_after, color='blue')
    plt.plot(self.df_bin['DATETIME'], self.df_bin['PRICE'], label='Trade', color='black')
    plt.axvline(x=self.time_n, color='orange', linestyle='--', label='Announcement Time')
    plt.axvline(x=self.time_h, color='blue', linestyle='--', label='Human Reaction Time')
    plt.axvline(x=self.time_c, color='violet', linestyle='--', label='Convergence Time')

    plt.title(f'{self.ticker} Trade Prices in 1 Minute Bins')
    plt.xlabel('Datetime')
    plt.ylabel('USD')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))
    plt.grid(True)
    plt.show()

  ##### access functions #####

  def year(self):
    return float(self.time_n.year)



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

  def get_models(self, year=None):
    if year == None:
      return self.models
    
    models_out = []
    for model in self.models:
      if model.year() == year:
        models_out.append(model)
    return models_out
