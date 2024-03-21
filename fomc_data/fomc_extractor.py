import csv
from datetime import datetime, timedelta
import zipfile
import pandas as pd
from io import StringIO

### load fomc minutes announcement times
with open('fomc_ann_times.txt', 'r') as file:
    ann_dates = set()
    for line in file:
        ann_dates.add(line.strip())
        
### produce a csv file with trades/quotes near announcements using zip_file_path and wrds_market_anns_map and saving to csv_file_path
def create_announcement_csv(zip_file_path, csv_file_path):
  # define a custom function to check if a trade is within 30 minutes of an announcement
  # also exclude exchange D, the "FINRA Alternative Display Facility"
  def near_annoucement(row):
    date = row['DATE']
    return date in ann_dates

  # pipeline data from zip to csv, reducing the amount of data
  with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    file_list = zip_ref.namelist()
    assert(len(file_list)==1)

    with zip_ref.open(file_list[0]) as input_file:

      # create column names from the first row
      first_chunk = pd.read_csv(input_file, nrows=1)
      column_names = first_chunk.columns

      with open(csv_file_path, 'w') as output_file:

        while True:
          # loop while there is a chunk
          chunk_size = 10000
          chunk = input_file.read(chunk_size)
          if not chunk:
              break
          
          df_chunk = pd.read_csv(StringIO(chunk.decode('utf-8')), names=column_names)
          df_chunk = df_chunk[df_chunk.apply(near_annoucement, axis=1)]

          df_chunk.to_csv(output_file, mode='a', header=not output_file.tell(), index=False)


### for a given zip file of trades/quotes, year, and if trades (else quotes), produce a csv file with trades or quotes near announcements
def zip_to_annoucement_csv(zip_csv_file_name, year, is_trade):
  if is_trade:
    announcement_csv_file_name = f"./fomc_trades_csv/trades_{year}.csv"
    print(f"creating csv for trades near annoucements at {announcement_csv_file_name}")
    create_announcement_csv(zip_csv_file_name, announcement_csv_file_name)
  else:
    announcement_csv_file_name = f"./fomc_quotes_csv/quotes_{year}.csv"
    print(f"creating csv for quotes near annoucements at {announcement_csv_file_name}")
    create_announcement_csv(zip_csv_file_name, announcement_csv_file_name)
    

def main():
  # get annoucement trades/quotes for each year
  compute_list = [
    ["./web_query_output/l8scdswpii8xnwbb_csv.zip", 2005, True],
    ["./web_query_output/aygi4bhauel4hjev_csv.zip", 2006, True],
    ["./web_query_output/s2rksnwxlemzeec1_csv.zip", 2007, True]
  ]

  for zip_csv_file_name, year, is_trade in compute_list:
    zip_to_annoucement_csv(zip_csv_file_name, year, is_trade)
    print(f"{year} done!")

if __name__ == "__main__":
  main()
  