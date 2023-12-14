import pandas as pd
import pickle
import matplotlib.pyplot as plt


### the announcement_model class represent an active announcement event and provides analysis and modeling functions
class announcement_model:

  ##### constructor functions #####

  def __init__(self, ticker, announcement_time, taq_df):
    self.ticker = ticker
    self.announcement_time = announcement_time
    self.df = self.construct_df(taq_df)

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
    return taq_df

  ##### df counting #####

  # def count_zero_bids(self):

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
  
