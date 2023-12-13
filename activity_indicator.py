### activity_indicator.py
# produced quote_tickers.txt and active_market_anns_map.pkl

import pandas as pd
from datetime import datetime
import pickle
import math

TRADE_COUNT_CUTOFF = 60*60*60/5 # one trade every 5 second
PERCENT_CHANGE_CUTOFF = 0.001 # 0.1% change in stock price

### load the market_anns_map from using pickle
def load_market_anns_map():
  with open("./market_anns_storage/market_anns_map.pkl", "rb") as file:
    wrds_market_anns_map = pickle.load(file)
  
  for year, dict1 in wrds_market_anns_map.items():
    for ticker, dict2 in dict1.items():
      for date, time in dict2.items():
        dict2[date] = datetime.strptime(time, "%H:%M:%S")

  return wrds_market_anns_map

def get_market_anns_list(market_anns_map):
  market_anns_list = []

  for year, dict1 in market_anns_map.items():
    for ticker, dict2 in dict1.items():
      for date, time in dict2.items():
        market_anns_list.append([year, ticker, date, time])

  return market_anns_list

def get_market_anns_year_list(market_anns_map, selected_year):
  market_anns_year_list = []

  for year, dict1 in market_anns_map.items():
    if year == selected_year:
      for ticker, dict2 in dict1.items():
        for date, time in dict2.items():
          market_anns_year_list.append([year, ticker, date, time])

  return market_anns_year_list

def get_df_year(year):
  df = pd.read_csv(f"./announcement_trades_csv/trades_{2023}.csv")
  df['TIME_M'] = pd.to_datetime(df['TIME_M'], format='%H:%M:%S.%f')
  df = df[['DATE', 'TIME_M', 'SYM_ROOT', 'SIZE', 'PRICE']]
  df.dropna(inplace=True)
  return df

def get_avg_price_before_ann(df, market_anns_map, year, ticker, date):
  annoucement_time = market_anns_map[year][ticker][date]
  return float(df[(df['SYM_ROOT'] == ticker) & (df['TIME_M'] < annoucement_time)]['PRICE'].mean())
  
def get_avg_price_after_ann(df, market_anns_map, year, ticker, date):
  annoucement_time = market_anns_map[year][ticker][date]
  return float(df[(df['SYM_ROOT'] == ticker) & (df['TIME_M'] > annoucement_time)]['PRICE'].mean())

def get_avg_price_change(df, market_anns_map, year, ticker, date):
  p2 = get_avg_price_after_ann(df, market_anns_map, year, ticker, date)
  p1 = get_avg_price_before_ann(df, market_anns_map, year, ticker, date)
  return abs((p2-p1)/p1)

def get_count_trades(df, ticker, date):
  return float((df[(df['SYM_ROOT'] == ticker) & (df['DATE'] == date)]['PRICE']).sum())


if __name__ == "__main__":
  market_anns_map = load_market_anns_map()
  market_anns_list = get_market_anns_list(market_anns_map)
  active_market_anns_map = {}

  # for year in range(2005, 2024):
  for iteration_year in range(2023,2024):
    market_anns_list_year = get_market_anns_year_list(market_anns_map, iteration_year)
    df = get_df_year(iteration_year)

    # add trade count and percent change to annoucement list
    market_anns_list_stats = []
    for year, ticker, date, anntim in market_anns_list_year:
      percent_change = get_avg_price_change(df, market_anns_map, year, ticker, date)
      trade_count = get_count_trades(df, ticker, date)
      if not math.isnan(trade_count) and not math.isnan(percent_change):
        market_anns_list_stats.append([year, ticker, date, anntim, trade_count, percent_change])


    active_market_anns = []
    for year, ticker, date, anntim, trade_count, percent_change in market_anns_list_stats:
      if trade_count > TRADE_COUNT_CUTOFF and percent_change > PERCENT_CHANGE_CUTOFF:
        active_market_anns.append([year, ticker, date, anntim, trade_count, percent_change])
    
    # save tickers
    tickers_out = [a[1] for a in active_market_anns]
    with open(f"quote_ticker_queries/quote_tickers_{year}.txt", 'w') as file:
      for ticker in tickers_out:
        file.write(f"{ticker}\n")


    ### use active_market_anns to create a dictionary for each year, ticker, and annoucement date
    ### save market_anns with pickle to ./market_anns_storage
    ### add to active_market_anns_map
    for year, ticker, date, anntim, trade_count, percent_change in active_market_anns:
      
      if year not in active_market_anns_map:
        active_market_anns_map[year] = {}
      
      if ticker not in active_market_anns_map[year]:
        active_market_anns_map[year][ticker] = {}
        
      active_market_anns_map[year][ticker][date] = anntim.strftime("%H:%M:%S")

  ### after calculating for each year, save active_market_anns_map with pickle
  with open("./market_anns_storage/active_market_anns_map.pkl", "wb") as file:
    pickle.dump(active_market_anns_map, file)



    
