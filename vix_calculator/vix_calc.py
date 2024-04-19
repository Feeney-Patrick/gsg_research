import math
import datetime
import pandas as pd
import numpy as np
import scipy as sp
import os


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
  recent_rates = fed_rates.iloc[recent_rate_idx].drop('date').tolist()

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
  strike_list = pd.Series(options['strike'].values, index=range(1, len(options['strike']) + 1))
  call_mid_point = (options['bid'] + options['ask']) / 2
  put_mid_point = (options['bid.1'] + options['ask.1']) / 2

  # Calculate delta_k (change in strike) and k_sq (strike squared) for each option
  delta_k = []
  k_sq = []

  # Handle first element (minimum strike) separately
  delta_k.append(strike_list.iloc[1] - strike_list.iloc[0])  # Change in strike for minimum strike option
  k_sq.append(strike_list.iloc[1] ** 2)  # Square of minimum strike

  # Calculate delta_k and k_sq for remaining strikes
  for strike in range(2, len(options['strike']) - 1):
    delta_k.append((strike_list.iloc[strike + 1] - strike_list.iloc[strike - 1]) / 2)  # Change in strike for middle strikes
    k_sq.append(strike_list.iloc[strike] ** 2)  # Square of middle strikes

  # Handle last element (maximum strike) separately
  delta_k.append(strike_list.iloc[-1] - strike_list.iloc[-2])  # Change in strike for maximum strike option
  k_sq.append(strike_list.iloc[-1] ** 2)  # Square of maximum strike

  # Calculate summation term for each option (weighted by delta_k/k_sq and option midpoints)
  summ_term = []
  for val in range(1, len(options['strike']) - 1):
    summ_term.append(delta_k[val] / k_sq[val] * math.exp(r * t) * call_mid_point.iloc[val])
    summ_term.append(delta_k[val] / k_sq[val] * math.exp(r * t) * put_mid_point.iloc[val])

  # Calculate the sum of all terms and adjust by time to expiration
  summ = sum(summ_term)
  sigma_sq = summ * (2 / t)

  # Find the maximum strike price less than the forward price (k0)
  k0 = strike_list[strike_list < f].max()

  # Calculate the implied volatility squared using the Garman-Forrand formula
  sigma_sq = sigma_sq - ((1 / t) * ((f / k0) - 1) ** 2)

  return sigma_sq



import datetime
import math


def calc_vix(date: datetime.date, days: int = 30) -> float:
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
  options = pd.read_csv('data/may_spx_quotedata.csv')
  options.columns = [col.lower().replace(' ', '_') for col in options.columns]

  # Select the nearest and next to nearest expiration dates
  near_date, next_date = select_near_next_date(options=options, date=date)

  # Calculate risk-free interest rates for near and next expirations
  near_r, next_r = calc_rf_spline_k0(date, near_date, next_date)

  # ---------- Single Term Variance Calculations ----------

  # Standard time for calculations (10:45 AM)
  day_start_time = datetime.time(hour=10, minute=45, second=0)

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

  # Weighting factor based on the time difference to the target maturity
  near_w = ((next_minute_diff - (60 * 24 * days)) / (next_minute_diff - near_minute_diff))
  next_w = (((60 * 24 * days) - near_minute_diff) / (next_minute_diff - near_minute_diff))

  # VIX formula with time scaling and annualization factors
  vix = 100 * math.sqrt(((near_t * near_sigma_sq * near_w) + (next_t * next_sigma_sq * next_w)) *
                         ((525600 / (60 * 24 * days))))

  return vix


# This line would calculate the VIX for today's date
# date = datetime.date.today()

# Explicitly set the date for calculation (replace with your desired date)
date = pd.to_datetime("2024-04-15").date()

vix = calc_vix(date)
print(vix)
