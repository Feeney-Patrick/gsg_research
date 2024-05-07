import math
import datetime
import pandas as pd
import numpy as np
import scipy as sp
import os
import requests
import requests
import json

def minute_diff(datetime1: datetime.datetime, datetime2: datetime.datetime) -> int:
  """
  Calculates the time difference between two datetime objects in minutes.

  Args:
      datetime1 (datetime.datetime): The first datetime object (assumed to be earlier).
      datetime2 (datetime.datetime): The second datetime object.

  Returns:
      int: The difference in minutes between datetime1 and datetime2. 
          A positive value indicates datetime2 is later, 
          and a negative value indicates datetime2 is earlier.

  Raises:
      TypeError: If either argument is not a datetime object.
  """

  # Validate input types
  if not isinstance(datetime1, datetime.datetime):
    raise TypeError("datetime1 must be a datetime object")
  if not isinstance(datetime2, datetime.datetime):
    raise TypeError("datetime2 must be a datetime object")

  # Calculate the difference in seconds with datetime2 as the ending point
  time_diff = datetime2 - datetime1
  total_seconds = time_diff.total_seconds()

  # Convert the total difference in seconds to minutes and round to nearest minute
  minute_diff = round(total_seconds / 60)

  return minute_diff

def select_option_strike_put_call(option: dict) -> tuple[float, float]:
  """
  This function selects the strike price and calculates the corresponding price difference 
  between the call and put options within a given option dictionary.

  Args:
      option (dict): A dictionary containing option data, including 'bid', 'ask', 
                     'bid.1', 'ask.1', and 'strike' keys. 
                     'bid' and 'ask' are assumed to be for call options, 
                     and 'bid.1' and 'ask.1' are assumed to be for put options.

  Returns:
      tuple[float, float]: A tuple containing the selected strike price (float) 
                          and the corresponding price difference between call and put (float), 
                          rounded to two decimal places.
  """

  # Calculate the middle price for call and put options
  mid_call = (option['bid_x'] + option['ask_x']) / 2
  mid_put = (option['bid_y'] + option['ask_y']) / 2

  # Calculate the absolute difference in price between call and put options for each strike
  strike_diff = abs(mid_call - mid_put)

  # Find the index (strike price) with the minimum price difference
  min_strike = strike_diff.idxmin()

  # Select the strike price and corresponding price difference based on the minimum index
  option_strike = option['strike'][min_strike]
  option_price_diff = (mid_call - mid_put)[min_strike]
  option_price_diff = round(option_price_diff, 2)

  return option_strike, option_price_diff

def select_near_next_date(options: pd.DataFrame, date: datetime.date, constant_maturity: int = 30) -> tuple[datetime.date, datetime.date]:
  """
  Selects the nearest expiration date (before the provided date with constant maturity added) 
  and the next expiration date (after the provided date with constant maturity added) 
  from the 'expiration_date' column of an options DataFrame.

  Args:
      options (pd.DataFrame): A DataFrame containing an 'expiration_date' column with datetime objects.
      date (datetime.date): The reference date for comparison.
      constant_maturity (int, optional): The additional maturity (in days) to consider for 
                                           selecting near and next dates. Defaults to 30.

  Returns:
      tuple[datetime.date, datetime.date]: A tuple containing the nearest expiration date (before) 
                                           and the next expiration date (after), both as datetime.date objects.
  """

  # Extract unique expiration dates from the DataFrame
  unique_dates = pd.to_datetime(options['expiration_date'], format="%Y-%m-%d").dt.date.unique()

  # Include a candidate date with constant_maturity added as reference
  candidate_date = date + datetime.timedelta(days=constant_maturity)
  unique_dates = np.concatenate([unique_dates, [candidate_date]])

  # Sort all unique dates (including the candidate) in ascending order
  unique_dates.sort()

  # Find indices of the target dates (nearest expiration before and next expiration after)
  target_date_idx = np.where(unique_dates == candidate_date)[0][0]

  # Select the nearest and next expiration dates based on the target date index
  near_date = unique_dates[target_date_idx - 1]
  next_date = unique_dates[target_date_idx + 1]

  return near_date, next_date

def calc_rf_spline_k0(date: datetime.date, near_date: datetime.date, next_date: datetime.date,
                     constant_maturity: int = 30) -> tuple[float, float]:
  """
  This function calculates the risk-free interest rates (r) for the nearest and next expiration dates 
  based on a cubic spline interpolation of historical daily treasury rates.

  Args:
      date (datetime.date): The reference date for calculations.
      near_date (datetime.date): The nearest expiration date to the reference date.
      next_date (datetime.date): The next expiration date to the reference date.
      constant_maturity (int, optional): The additional maturity (in days) to consider 
                                           for calculating risk-free rates. Defaults to 30.

  Returns:
      tuple[float, float]: A tuple containing the risk-free interest rate (r) for the 
                           nearest expiration date and the next expiration date.
  """

  # Read daily treasury rates data (assuming 'data/daily_treasury_rates.csv' format)
  fed_rates = pd.read_csv('data/daily_treasury_rates.csv')

  # Clean column names (lowercase and replace spaces with underscores)
  fed_rates.columns = [col.lower().replace(' ', '_') for col in fed_rates.columns]

  # Ensure consistent date format for 'date' column
  date_format = "%m/%d/%Y"
  fed_rates['date'] = pd.to_datetime(fed_rates['date'], format=date_format).dt.date

  # Fixed maturities (in days) for which rates are available
  fixed_mat_days = [30, 60, 91, 121, 182, 365, 730, 1095, 1825, 2555, 3650, 7300, 10950]

  # Find the index of the most recent date (excluding today) in the data
  recent_rate_idx = np.argmax(fed_rates['date'] <= (date - datetime.timedelta(days=1)))

  # Extract the most recent rates for available maturities
  recent_rates = fed_rates.iloc[recent_rate_idx-1].drop('date').tolist()
  # recent_rates = [.03, .02, .04, .045, .05, .08, .11, .22, .59, 1.00, 1.37, 2.03, 2.21] # Testing

  if np.isnan(recent_rates).any():
    recent_rates = numpy_fill(recent_rates)

  # Cubic spline interpolation with natural boundary conditions (second derivative = 0 at ends)
  # This enforces a smoother curve with zero curvature at the beginning and end.
  rf_spline = sp.interpolate.CubicSpline(fixed_mat_days, recent_rates, bc_type=((2, -0.000021), (2, 0.0)))

  # Calculate the date with constant maturity added as reference
  constant_maturity_date = date + datetime.timedelta(days=constant_maturity)

  # Calculate risk-free rates for near and next expiration dates based on their distance from constant_maturity_date
  near_days = constant_maturity - abs(near_date - constant_maturity_date).days
  next_days = (next_date - constant_maturity_date).days + constant_maturity

  # Use the spline to interpolate rates for the calculated days
  near_bey = rf_spline(near_days)
  next_bey = rf_spline(next_days)

  # Convert discount yield (basis points) to annual percentage yield (APY)
  near_apy = calc_apy(near_bey)
  next_apy = calc_apy(next_bey)

  # Convert APY to log return (risk-free interest rate)
  near_r = np.log(1 + near_apy)
  next_r = np.log(1 + next_apy)

  return near_r, next_r

def numpy_fill(arr):
    nan_indices = np.where(np.isnan(arr))[0]
    fill = arr[nan_indices[0] + 1]
    arr[nan_indices[0]] = fill
    return arr

def calc_apy(rate: float, n: int = 2) -> float:
  """
  This function calculates the Annual Percentage Yield (APY) for a given discount yield 
  and compounding frequency.

  Args:
      rate (float): The discount yield as a decimal (e.g., 0.01 for 1%).
      n (int, optional): The number of compounding periods per year. Defaults to 2 (semi-annual).

  Returns:
      float: The calculated APY as a decimal.
  """

  apy = (1 + rate / n) ** n - 1
  return apy

def calc_sigma_sq(options: pd.DataFrame, t: float, r: float, f: float) -> float:
  """
  This function calculates the implied volatility squared (sigma^2) 
  for a set of European call and put options using the Garman-Forrand formula.

  Args:
      options (pd.DataFrame): A DataFrame containing columns for 'strike', 'bid', 
                              'ask', 'bid.1', and 'ask.1'. 'bid' and 'ask' are assumed 
                              to be for call options, and 'bid.1' and 'ask.1' are assumed 
                              to be for put options.
      t (float): Time to expiration (in years).
      r (float): Risk-free interest rate (as a decimal).
      f (float): Forward price of the underlying asset.

  Returns:
      float: The calculated implied volatility squared (sigma^2).
  """
  
  # combined_df = pd.DataFrame({'Strike': strike_list, 'Call Midpoint': call_mid_point, 'Put Midpoint': put_mid_point})
  try:
    k0 = options[options['strike'] < f].iloc[-1].strike

    # Evaluate Puts
    filtered_put_strikes = options[options['strike'] <= k0]
    filtered_put_strikes = filtered_put_strikes.drop(columns=['bid_x', 'ask_x', 'option_x'])

    p_mask = (filtered_put_strikes['bid_y'] == 0) & (filtered_put_strikes['bid_y'].shift(-1) == 0)
    if p_mask.any():
      p_max_row = filtered_put_strikes.index[p_mask].tolist()[-1]
      filtered_put_strikes = filtered_put_strikes[filtered_put_strikes.index >= p_max_row + 2]
      filtered_put_strikes = filtered_put_strikes[filtered_put_strikes['bid_y'] != 0.0]
  
    # Evaluate Calls
    filtered_call_strikes = options[options['strike'] >= k0]
    filtered_call_strikes = filtered_call_strikes.drop(columns=['bid_y', 'ask_y', 'option_y'])

    c_mask = (filtered_call_strikes['bid_x'] == 0) & (filtered_call_strikes['bid_x'].shift(-1) == 0)

    if p_mask.any():
      c_max_row = filtered_call_strikes.index[c_mask].tolist()[-1]
      # Get the row numbers where the condition is met
      filtered_call_strikes = filtered_call_strikes[filtered_call_strikes.index <= c_max_row - 2]
      filtered_call_strikes = filtered_call_strikes[filtered_call_strikes['bid_x'] != 0.0]

    put_mid_point = (filtered_put_strikes['bid_y'] + filtered_put_strikes['ask_y']) / 2
    call_mid_point = (filtered_call_strikes['bid_x'] + filtered_call_strikes['ask_x']) / 2

    put_call_mid_point = (put_mid_point.iloc[-1] + call_mid_point.iloc[0]) / 2
    
    put_mid_point_filtered = put_mid_point.iloc[:-1]  # Exclude first and last row
    k0_index = options[options['strike'] == k0].index[0]
    put_mid_point_filtered.loc[k0_index] = put_call_mid_point

    call_mid_point_filtered = call_mid_point.iloc[1:] 

    merged_options = pd.concat([put_mid_point_filtered, call_mid_point_filtered], axis=0)


      # Extract strike prices and calculate midpoints for call and put options
    strike_list = pd.Series(options['strike'].values, index=options['strike'].index)
    strike_list_filtered = strike_list[merged_options.index]
    

    strike_merged_df = pd.concat([merged_options, strike_list_filtered], axis=1)
    strike_merged_df.columns = (['price', 'strike'])

    # Calculate delta_k (change in strike) and k_sq (strike squared) for each option
    delta_k = []
    k_sq = []

    # Handle first element (minimum strike) separately
    delta_k.append(strike_merged_df['strike'].iloc[1] - strike_merged_df['strike'].iloc[0])  # Change in strike for minimum strike option
    k_sq.append(strike_merged_df['strike'].iloc[0] ** 2)  # Square of minimum strike

    # Calculate delta_k and k_sq for remaining strikes
    for strike in range(1, len(strike_merged_df) - 1):
      delta_k.append((strike_merged_df['strike'].iloc[strike + 1] - strike_merged_df['strike'].iloc[strike - 1]) / 2)  # Change in strike for middle strikes
      k_sq.append(strike_merged_df['strike'].iloc[strike] ** 2)  # Square of middle strikes

    # Handle last element (maximum strike) separately
    delta_k.append(strike_merged_df['strike'].iloc[-1] - strike_merged_df['strike'].iloc[-2])  # Change in strike for maximum strike option
    k_sq.append(strike_merged_df['strike'].iloc[-1] ** 2)  # Square of maximum strike

    # Calculate summation term for each option (weighted by delta_k/k_sq and option midpoints)
    summ_term = []
    for val in range(0, len(strike_merged_df)):
      summ_term.append(delta_k[val] / k_sq[val] * math.exp(r/100 * t) * strike_merged_df['price'].iloc[val])

    # Calculate the sum of all terms and adjust by time to expiration
    summ = sum(summ_term)
    sigma_sq = summ * (2 / t)

    # Calculate the implied volatility squared using the Garman-Forrand formula
    sigma_sq = sigma_sq - ((1 / t) * ((f / k0) - 1) ** 2)

    return sigma_sq
  except:
    print("The VIX cannot be calculated for this date.")

def calc_vix(date: datetime.date, options: pd.DataFrame, days: int = 30) -> float:
  """
  This function calculates the VIX (Volatility Index) for a given date based 
  on the implied volatility of SPX options using the closing prices on the 
  settlement date of the nearest and next to nearest SPX option expirations.

  Args:
      date (datetime.date): The date for which to calculate the VIX.
      days (int, optional): The maturity (in days) used for weighting between 
                             the nearest and next to nearest expirations. Defaults to 30.

  Returns:
      float: The calculated VIX value.
  """

  now = datetime.datetime.now()
  hour=now.hour
  minute=now.minute
  second=now.second

  # Select the nearest and next to nearest expiration dates
  near_date, next_date = select_near_next_date(options=options, date=date)

  # Calculate risk-free interest rates for near and next expirations
  near_r, next_r = calc_rf_spline_k0(date, near_date, next_date)

  # ---------- Single Term Variance Calculations ----------

  # Standard time for calculations (10:45 AM)
  day_start_time = datetime.time(hour=hour, minute=minute, second=second)
  print('day start time: ', day_start_time)

  # Calculate time to expiration in minutes for near and next options
  near_minute_diff = minute_diff(datetime1=datetime.datetime.combine(date, day_start_time),
                                  datetime2=datetime.datetime.combine(near_date, datetime.time(9, 30)))
  next_minute_diff = minute_diff(datetime1=datetime.datetime.combine(date, day_start_time),
                                 datetime2=datetime.datetime.combine(next_date, datetime.time(16, 00)))

  near_t = near_minute_diff / 525600  # Convert minutes to years
  next_t = next_minute_diff / 525600

  # Filter options for near and next expiration dates
  near_options = options[options['expiration_date'] == near_date]
  next_options = options[options['expiration_date'] == next_date]

  # Select option with minimum price difference (assumes ATM option)
  near_strike, near_price_diff = select_option_strike_put_call(option=near_options)
  next_strike, next_price_diff = select_option_strike_put_call(option=next_options)

  # Calculate forward price for near and next options
  near_f = near_strike + math.exp(near_r / 100 * near_t) * (near_price_diff)
  next_f = next_strike + math.exp(next_r / 100 * next_t) * (next_price_diff)

  # Calculate implied volatility squared for near and next options
  near_sigma_sq = calc_sigma_sq(options=near_options, t=near_t, r=near_r, f=near_f)
  next_sigma_sq = calc_sigma_sq(options=next_options, t=next_t, r=next_r, f=next_f)

  # ---------- VIX Calculation ----------
  try:
  # Weighting factor based on the time difference to the target maturity
    near_w = ((next_minute_diff - (60 * 24 * days)) / (next_minute_diff - near_minute_diff))
    next_w = (((60 * 24 * days) - near_minute_diff) / (next_minute_diff - near_minute_diff))

    # VIX formula with time scaling and annualization factors
    vix = 100 * math.sqrt(((near_t * near_sigma_sq * near_w) + (next_t * next_sigma_sq * next_w)) *
                          ((525600 / (60 * 24 * days))))
    return vix
  
  except:
    raise TypeError("The VIX calculation could not be completed")

def download_treasury_csv(date):
  """
  Downloads the daily treasury rates CSV file for the given year and saves it locally.

  Args:
      date (datetime.date): The date object for which to download the data.
  """

  # Folder to store downloaded files (create if it doesn't exist)
  data_folder = "data/"
  os.makedirs(data_folder, exist_ok=True)  # Create folder if it doesn't exist

  # Filename for downloaded CSV (replace if needed)
  expected_filename = "daily_treasury_rates.csv"

  # Remove existing file if present (optional)
  download_path = os.path.join(data_folder, expected_filename)
  if os.path.exists(download_path):
    os.remove(download_path)

  # Extract year from the date
  year = date.year

  # Construct the download URL (replace with the actual URL if needed)
  url = f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv"

  try:
    print(f"Downloading rates for year: {year}")

    # Send a GET request with streaming download
    response = requests.get(url, stream=True)

    # Check for successful response (status code 200)
    if response.status_code == 200:
      # Open the file in write binary mode
      with open(download_path, 'wb') as f:
        # Download content in chunks and write to file
        for chunk in response.iter_content(1024):
          if chunk:  # filter out keep-alive new chunks
            f.write(chunk)
      print(f"CSV downloaded successfully! Saved as: {download_path}")
    else:
      print(f"Error downloading CSV. Status code: {response.status_code}")

  except requests.exceptions.RequestException as e:
    print(f"An error occurred while downloading the CSV: {e}")

def fetch_spx_options_data(save_to_csv=False):
  """Fetches S&P 500 options data from the CBOE API and returns the data as a dictionary.

  Returns:
      dict: The dictionary containing the S&P 500 options data retrieved from the API.

  Raises:
      requests.exceptions.RequestException: If an error occurs during the request.
  """

  url = "https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json"

  try:
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for unsuccessful requests
  except requests.exceptions.RequestException as e:
    print("An error occurred:", e)
    return None  # Indicate error by returning None

  data = response.json()

  # Optional processing or logging of data (commented out)
  print("Timestamp:", data["timestamp"])
  print("Number of option contracts:", len(data["data"]["options"]))

  if save_to_csv:
  # Optionally, save the data to a file named "spx_options.json" (commented out)
    with open("spx_options.json", "w") as outfile:
      json.dump(data, outfile)

  return data

def process_spx_options_data(data):
  """Processes S&P 500 options data from a dictionary and returns DataFrames for calls/puts (daily/weekly).

  Args:
      data (dict): The dictionary containing S&P 500 options data retrieved from the API.

  Returns:
      tuple: A tuple containing four DataFrames: daily calls, daily puts, weekly calls, and weekly puts.
  """

  # Get the list of options data from the JSON (assuming structure)
  options_data = data["data"]["options"]

  df = pd.DataFrame(options_data)

  # Define columns to remove (as a list)
  columns_to_remove = ['bid_size', 'ask_size', 'volume','iv', 'open_interest',
                        'volume', 'delta', 'gamma', 'theta', 'rho', 'vega', 'theo',
                        'change', 'open', 'high', 'low', 'tick', 'last_trade_price',
                        'last_trade_time', 'percent_change', 'prev_day_close']

  # Remove columns using drop
  df_modified = df.drop(columns_to_remove, axis=1) 

  # Extract strike prices (assuming the last 8 characters represent the strike)
  df_modified["strike"] = df_modified["option"].str[-8:]

  # Separate calls and puts based on the option type (C or P) at the 11th position
  daily_calls = df_modified[df_modified["option"].str[9] == "C"]
  daily_puts = df_modified[df_modified["option"].str[9] == "P"]
  weekly_calls = df_modified[df_modified["option"].str[10] == "C"]
  weekly_puts = df_modified[df_modified["option"].str[10] == "P"]

  daily_calls = daily_calls.assign(expiration_date=daily_calls["option"].apply(extract_expiration_date))
  daily_puts = daily_puts.assign(expiration_date=daily_puts["option"].apply(extract_expiration_date))
  weekly_calls = weekly_calls.assign(expiration_date=weekly_calls["option"].apply(extract_expiration_date))
  weekly_puts = weekly_puts.assign(expiration_date=weekly_puts["option"].apply(extract_expiration_date))

  daily_calls = daily_calls.assign(strike=daily_calls["strike"].apply(convert_strike_to_float))
  daily_puts = daily_puts.assign(strike=daily_puts["strike"].apply(convert_strike_to_float))
  weekly_calls = weekly_calls.assign(strike=weekly_calls["strike"].apply(convert_strike_to_float))
  weekly_puts = weekly_puts.assign(strike=weekly_puts["strike"].apply(convert_strike_to_float))

  return daily_calls, daily_puts, weekly_calls, weekly_puts

def extract_expiration_date(option_name):
  """Extracts the expiration date from the option name in 'YYYY-MM-DD' format.

  Args:
      option_name (str): The option name string.

  Returns:
      str: The expiration date in 'YYYY-MM-DD' format, or None if not found.
  """

  if "W" in option_name[3]:  # Check for 'W' at the 4th character position
    # Weekly option format
    year = "20" + option_name[4:6]
    month = option_name[6:8]
    day = option_name[8:10]
  else:
    # Daily option format (assume current year)
    year = "20" + option_name[3:5]
    month = option_name[5:7]
    day = option_name[7:9]

  # Create the date string and return
  return f"{year}-{month}-{day}"

def convert_strike_to_float(strike_string):
  """Converts a strike price string (e.g., "01400000") to a float.

  Args:
      strike_string (str): The strike price string.

  Returns:
      float: The strike price as a float, or None if the conversion fails.
  """

  try:
    # Remove leading zeros and convert to float
    strike_string = strike_string.lstrip("0")
    strike_float = float(strike_string[:-3])
    return strike_float
  except ValueError:
    print("Error: Invalid strike price format")
    return None
  
def convert_to_date(date_str):
  try:
    # Attempt conversion using '%Y-%m-%d' format (assuming YYYY-MM-DD)
    return pd.to_datetime(date_str, format='%Y-%m-%d').date()
  except ValueError:
    # Handle potential errors (e.g., invalid format)
    return None  # Or return a specific value to indicate error

def initialize_spx_options_data():

  data = fetch_spx_options_data()
  daily_calls, daily_puts, weekly_calls, weekly_puts = process_spx_options_data(data)

  # Get unique expiration dates from daily options
  daily_expirations = pd.concat([daily_calls["expiration_date"], daily_puts["expiration_date"]]).unique()

  # Filter weekly options by excluding matching expiration dates
  weekly_calls_filtered = weekly_calls[~weekly_calls["expiration_date"].isin(daily_expirations)]
  weekly_puts_filtered = weekly_puts[~weekly_puts["expiration_date"].isin(daily_expirations)]

  # Full outer merge on "strike" and "expiration_date"
  daily_options = daily_calls.merge(daily_puts, how='outer', on=['strike', 'expiration_date'])

  # Full outer merge on "strike" and "expiration_date" for weekly options
  weekly_options = weekly_calls_filtered.merge(weekly_puts_filtered, how='outer', on=['strike', 'expiration_date'])

  # Concatenate daily and weekly options
  all_options = pd.concat([daily_options, weekly_options])

  # Apply the conversion function to the 'date_string' column
  all_options['expiration_date'] = all_options['expiration_date'].apply(convert_to_date)

  return all_options

if __name__ == "__main__":

  date = datetime.date.today()

  # Pull interest rates
  download_treasury_csv(date)

  all_options = initialize_spx_options_data()

  vix = calc_vix(date = date, options = all_options)

  print(f"The vix for date, {date} is {vix}.")