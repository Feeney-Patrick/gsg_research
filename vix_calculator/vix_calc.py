import math
import datetime
import pandas as pd
import numpy as np
import scipy as sp
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import requests
from yahoo_fin import stock_info as si
from bs4 import BeautifulSoup


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
  mid_call = (option['bid'] + option['ask']) / 2
  mid_put = (option['bid.1'] + option['ask.1']) / 2

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

  # Ensure consistent format for expiration dates (assuming '%a %b %d %Y' format)
  options['expiration_date'] = pd.to_datetime(options['expiration_date'], format="%a %b %d %Y").dt.date

  # Extract unique expiration dates from the DataFrame
  unique_dates = options['expiration_date'].unique()

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


import math


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

  # Extract strike prices and calculate midpoints for call and put options
  strike_list = pd.Series(options['strike'].values, index=options['strike'].index)
  
  # combined_df = pd.DataFrame({'Strike': strike_list, 'Call Midpoint': call_mid_point, 'Put Midpoint': put_mid_point})
  try:
    k0 = options[options['strike'] < f].iloc[-1].strike

    # Evaluate Puts
    filtered_put_strikes = options[options['strike'] <= k0]
    filtered_put_strikes = filtered_put_strikes.drop(columns=['bid', 'ask'])

    p_mask = (filtered_put_strikes['bid.1'] == 0) & (filtered_put_strikes['bid.1'].shift(-1) == 0)
    if p_mask.any():
      p_max_row = filtered_put_strikes.index[p_mask].tolist()[-1]
      filtered_put_strikes = filtered_put_strikes[filtered_put_strikes.index >= p_max_row + 2]
      filtered_put_strikes = filtered_put_strikes[filtered_put_strikes['bid.1'] != 0.0]
  
    # Evaluate Calls
    filtered_call_strikes = options[options['strike'] >= k0]
    filtered_call_strikes = filtered_call_strikes.drop(columns=['bid.1', 'ask.1'])

    c_mask = (filtered_call_strikes['bid'] == 0) & (filtered_call_strikes['bid'].shift(-1) == 0)

    if p_mask.any():
      c_max_row = filtered_call_strikes.index[c_mask].tolist()[-1]
      # Get the row numbers where the condition is met
      filtered_call_strikes = filtered_call_strikes[filtered_call_strikes.index <= c_max_row - 2]
      filtered_call_strikes = filtered_call_strikes[filtered_call_strikes['bid'] != 0.0]

    put_mid_point = (filtered_put_strikes['bid.1'] + filtered_put_strikes['ask.1']) / 2
    call_mid_point = (filtered_call_strikes['bid'] + filtered_call_strikes['ask']) / 2

    put_call_mid_point = (put_mid_point.iloc[-1] + call_mid_point.iloc[0]) / 2
    
    put_mid_point_filtered = put_mid_point.iloc[:-1]  # Exclude first and last row
    new_index = put_mid_point_filtered.index.max() + 1
    put_mid_point_filtered.loc[new_index] = put_call_mid_point

    call_mid_point_filtered = call_mid_point.iloc[1:] 

    merged_options = pd.concat([put_mid_point_filtered, call_mid_point_filtered], axis=0)

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


def calc_vix(date: datetime.date, options_data_source: int = 3, days: int = 30) -> float:
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

  # Read SPX options data (assuming 'data/may_spx_quotedata.csv' format)
  if options_data_source == 1:
    options = pd.read_csv('data/spx_quotedata.csv', skiprows=3) # for cboe download
    options.columns = [col.lower().replace(' ', '_') for col in options.columns]
    now = datetime.datetime.now()
    hour=now.hour
    minute=now.minute
    second=now.second
  elif options_data_source == 2:
    options = pd.read_csv('data/spx_quotedata.csv')
    options.columns = [col.lower().replace(' ', '_') for col in options.columns]
    now = datetime.datetime.now()
    hour=now.hour
    minute=now.minute
    second=now.second
  elif options_data_source == 3:
    options = pd.read_csv('data/vix_options.csv') # Testing
    hour=10
    minute=45
    second=0

  # Select the nearest and next to nearest expiration dates
  near_date, next_date = select_near_next_date(options=options, date=date)

  # Calculate risk-free interest rates for near and next expirations
  near_r, next_r = calc_rf_spline_k0(date, near_date, next_date)

  # ---------- Single Term Variance Calculations ----------

  # Standard time for calculations (10:45 AM)
  day_start_time = datetime.time(hour=hour, minute=minute, second=second)

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


def get_month_abbreviation(date_obj):
  """
  This function takes a datetime.date object and returns the corresponding three-letter month abbreviation.

  Args:
      date_obj (datetime.date): The date object.

  Returns:
      str: The three-letter abbreviation of the month, or None if the date object is invalid.
  """

  # Check if the input is a valid datetime.date object
  if not isinstance(date_obj, datetime.date):
    print("Invalid input. Please provide a datetime.date object.")
    return None

  month_number = date_obj.month

  month_numbers = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec"
  }

  if month_number in month_numbers:
    return month_numbers[month_number + 1]
  else:
    return None

def pull_spx_quote_data(date):
  folder = "data/"
  expected_file_name = "spx_quotedata.csv"  # Replace with actual filename
  if os.path.exists(folder + expected_file_name):
      os.remove(folder + expected_file_name)

  month = get_month_abbreviation(date)

  # Get the current working directory
  cwd = os.getcwd()
  # Create a relative path for the download directory
  relative_path = "data"  # Example: "downloads" folder in the same directory as your script
  # Combine the CWD and relative path
  download_path = os.path.join(cwd, relative_path)

  # Replace with the path to your ChromeDriver
  options = Options()
  options.add_experimental_option("prefs", {"download.default_directory": download_path })

  driver = webdriver.Chrome(options=options)

  # Get the VIX Delayed Quotes page
  url = "https://www.cboe.com/delayed_quotes/spx/"

  try:
    driver.get(url)
    # Wait for the page to load (adjust wait time if needed)
    driver.implicitly_wait(10)
    quote_table_link = driver.find_element(By.ID, "quote_table")
    quote_table_link.click()

    driver.implicitly_wait(10)
    second_button = driver.find_element(By.XPATH, "(//button[contains(text(), 'I agree')])")
    second_button.click()

    driver.implicitly_wait(10)
    second_button = driver.find_element(By.XPATH, f"(//button/*[contains(text(), '{month}')])[1]")
    second_button.click()

    driver.implicitly_wait(10)
    target_element = driver.find_element(By.XPATH, "(//a/*[contains(text(), 'Download')])[1]")
    target_element.click()
  except:
    print("There was an issue!")
    driver.quit()
    
  # Define a maximum wait time (adjust as needed)
  max_wait_time = 30  # Seconds

  # Start timer
  start_time = time.time()
  # Check if the file exists in the download directory
  download_path = os.path.join(download_path, expected_file_name)
  print(download_path)

  while True:
    
    if os.path.isfile(download_path):
      # Download finished!
      print(f"Download finished! File: {download_path}")
      break

    # Check if timeout has been reached
    elapsed_time = time.time() - start_time
    if elapsed_time > max_wait_time:
      print("Download timed out after", max_wait_time, "seconds.")
      break

    # Sleep for a short duration before checking again
    time.sleep(1)

  # Close the browser window
  driver.quit()

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

def generate_unix_timestamps(start_date, end_date):
  """
  Generates a list of Unix timestamps (seconds since epoch in UTC) for each day
  between the given start and end dates (inclusive).

  Args:
      start_date (datetime.date): The starting date (inclusive).
      end_date (datetime.date): The ending date (inclusive).

  Returns:
      list[int]: A list of Unix timestamps for each day in the date range.
  """

  # Format input dates as strings in YYYY-MM-DD format
  start_date_str = start_date.strftime('%Y-%m-%d')
  end_date_str = end_date.strftime('%Y-%m-%d')

  # Convert strings to datetime objects with time set to 00:00:00
  start_dt = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
  end_dt = datetime.datetime.strptime(end_date_str, '%Y-%m-%d')

  # Initialize an empty list to store timestamps
  timestamps = []

  # Iterate through each day between start and end dates (inclusive)
  current_dt = start_dt
  while current_dt <= end_dt:
    # Get the Unix timestamp for 12:00:00 AM (UTC) of the current date
    timestamp = int(current_dt.replace(tzinfo=datetime.timezone.utc).timestamp())
    timestamps.append(timestamp)

    # Move to the next day
    current_dt += datetime.timedelta(days=1)

  return timestamps


def get_date_range(start_date):
  """
  Calculates a date range centered around a given start date.

  This function takes a `start_date` as input, adds 30 days to it, and then returns a tuple
  containing two `datetime.date` objects representing the start and end dates of a 31-day range.
  The range extends 15 days before and 15 days after the new date (which is 30 days from the
  original start date), ensuring the original start date is included.

  Args:
      start_date (datetime.date): The starting date for the date range calculation.

  Returns:
      tuple[datetime.date, datetime.date]: A tuple containing two elements:
          - The start date of the 31-day range.
          - The end date of the 31-day range.
  """

  # Calculate a new date 30 days from the start date
  new_date = start_date + datetime.timedelta(days=30)

  # Calculate the start date for the 15-day range before the new date
  start_of_range = new_date - datetime.timedelta(days=15)

  # Calculate the end date for the 15-day range after the new date
  end_of_range = new_date + datetime.timedelta(days=15)

  # Return the start and end dates of the 31-day range
  return start_of_range, end_of_range


def generate_option_urls(unix_timestamps):
  """
  Generates a list of Yahoo Finance options page URLs for the given Unix timestamps.

  This function takes a list of Unix timestamps (seconds since epoch) and creates a
  corresponding list of URLs for the Yahoo Finance options page. Each URL points to
  the options page for the specific date represented by the timestamp.

  Args:
      unix_timestamps (list[int]): A list of integers representing Unix timestamps.

  Returns:
      list[str]: A list of strings containing the constructed URLs for the options pages.
  """

  # Initialize an empty list to store URLs
  urls = []

  # Loop through each Unix timestamp
  for timestamp in unix_timestamps:
    # Optionally convert timestamp to human-readable date (uncomment if needed)
    # human_readable_date = datetime.datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d')

    # Construct the base URL for Yahoo Finance options page with the timestamp
    url = f"https://finance.yahoo.com/quote/%5ESPX/options?date={timestamp}"
    urls.append(url)

  # Return the list of generated URLs
  return urls


def retrieve_yahoo_option_data(date):

  start_date, end_date = get_date_range(date)
  unix_dates = generate_unix_timestamps(start_date, end_date)
  option_urls = generate_option_urls(unix_dates)

  folder = "data/"
  expected_file_name = "spx_quotedata.csv"  # Replace with actual filename
  if os.path.exists(folder + expected_file_name):
      os.remove(folder + expected_file_name)

  # Replace with the path to your ChromeDriver
  driver = webdriver.Chrome()

  spx_options_chain = pd.DataFrame()

  for val in range(0, len(option_urls)):
  # Get the SPX Delayed Quotes page
    try:
      driver.get(option_urls[val])
      # Wait for the page to load (adjust wait time if needed)
      driver.implicitly_wait(10)
      call_table = driver.find_element(By.XPATH, "(//table)[1]")
      html_content = call_table.get_attribute("outerHTML")

      # Parse the HTML content using BeautifulSoup
      soup = BeautifulSoup(html_content, "html.parser")

      # Extract table data using BeautifulSoup methods (adjust as needed)
      table_data = []
      for row in soup.find_all("tr"):
        # Extract data from each cell within the row
        row_data = []
        for cell in row.find_all("td"):
          cell_text = cell.text.strip()  # Get cell text and remove leading/trailing whitespace
          row_data.append(cell_text)
        table_data.append(row_data)

      # Create a Pandas DataFrame from the extracted data
      df_c = pd.DataFrame(table_data)
      df_c = df_c.iloc[1:]
      date_time = datetime.datetime.fromtimestamp(unix_dates[val])
      formatted_date = date_time.strftime('%a %b %d %Y')
      df_c.columns = ['contract', 'expiration_date', 'strike', 'last', 'bid', 'ask', '6','7','8', '9', '10']
      df_c['expiration_date'] = formatted_date
      df_c = df_c.drop(['contract', 'last','6','7','8', '9', '10'], axis=1)

      put_table = driver.find_element(By.XPATH, "(//table)[2]")
      html_content = put_table.get_attribute("outerHTML")

      # Parse the HTML content using BeautifulSoup
      soup = BeautifulSoup(html_content, "html.parser")

      # Extract table data using BeautifulSoup methods (adjust as needed)
      table_data = []
      for row in soup.find_all("tr"):
        # Extract data from each cell within the row
        row_data = []
        for cell in row.find_all("td"):
          cell_text = cell.text.strip()  # Get cell text and remove leading/trailing whitespace
          row_data.append(cell_text)
        table_data.append(row_data)

      # Create a Pandas DataFrame from the extracted data
      df_p = pd.DataFrame(table_data)
      df_p = df_p.iloc[1:]
      date_time = datetime.datetime.fromtimestamp(unix_dates[val])
      formatted_date = date_time.strftime('%a %b %d %Y')
      df_p.columns = ['contract', 'expiration_date', 'strike', 'last', 'bid.1', 'ask.1', '6','7','8', '9', '10']
      df_p['expiration_date'] = formatted_date
      df_p = df_p.drop(['expiration_date','contract', 'last','6','7','8', '9', '10'], axis=1)

      merged_df = df_c.merge(df_p, on='strike', how='outer').dropna()
      merged_df['expiration_date'] = formatted_date


      spx_options_chain = pd.concat([spx_options_chain, merged_df], ignore_index=True)
      
    except:
      print(f"No data was found for url: {option_urls[val]}.")

  driver.quit()

  spx_options_chain['bid'] = spx_options_chain['bid'].str.replace(',', '').astype(float).fillna(0)
  spx_options_chain['ask'] = spx_options_chain['ask'].str.replace(',', '').astype(float).fillna(0)
  spx_options_chain['bid.1'] = spx_options_chain['bid.1'].str.replace(',', '').astype(float).fillna(0)
  spx_options_chain['ask.1'] = spx_options_chain['ask.1'].str.replace(',', '').astype(float).fillna(0)
  spx_options_chain['strike'] = spx_options_chain['strike'].str.replace(',', '').astype(int).fillna(0)

  spx_options_chain.to_csv((folder + expected_file_name), index=False) 

def convert_to_int(value):
  try:
    # Remove commas and convert to integer
    return int(value.replace(",", ""))
  except:
    # Handle errors (e.g., invalid characters, missing values)
    # You can return original value, log errors, or raise exceptions
    return value



# ---------------------------

# # Pulls from Cboe as csv from what is given here: https://www.cboe.com/delayed_quotes/spx/
# # There is not sufficient data for an analysis, refer to function pull_spx_quote_data for data source
# date = datetime.date.today()
# pull_spx_quote_data(date)
# options_data_source = 1

# # Pull options from yahoo finance website data available on website is incomplete as compared to vix example.
# # Requires user to have Chrome installed
# # Refer to function retrieve_yahoo_option_data for data source. 
date = datetime.date.today()
retrieve_yahoo_option_data(date)
options_data_source = 2

# Explicitly set the date for calculation (replace with your desired date)
# date = pd.to_datetime("2022-09-27").date()
# options_data_source = 3

# Pull interest rates
download_treasury_csv(date)

vix = calc_vix(date, options_data_source)
print(f"The vix for date, {date} is {vix}.")
