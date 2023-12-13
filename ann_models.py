import pandas as pd
import pickle


class announcement_model:
  def __init__(self, ticker, announcement_time, taq_df):
    self.ticker = ticker
    self.announcement_time = announcement_time
    self.taq_df = taq_df


class model_store:

  ### constructor functions

  def __init__(self):

    self.models = []
    self.construct_models()

  # helper function for constructor to create the initial models
  def construct_models(self):
    # load from announcement_taq
    # [['DATETIME', 'TICKER', 'SIZE', 'PRICE', 'BID', 'ASK']]
    # [[datetime, string, int, float, float, float]]
    taq_all = pd.read_csv("./announcement_taq_csv/taq_all.csv")
    
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
          
          self.models.append(announcement_model(ticker, curr_datetime, curr_df))


    ### analysis functions
  
a = model_store()