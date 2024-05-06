import bs4 as bs
import requests
import yfinance as yf
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

def download_save_sp500_stocks():
  """
  This function downloads and saves daily adjusted closing prices for S&P 500 companies from Wikipedia and Yahoo Finance.

  Downloads data for the year 2016. Saves the data to a CSV file named 'data/sp500.csv'.
  """

  # Download list of S&P 500 tickers from Wikipedia
  resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
  soup = bs.BeautifulSoup(resp.text, 'lxml')
  table = soup.find('table', {'class': 'wikitable sortable'})

  tickers = []
  for row in table.findAll('tr')[1:]:  # Skip the header row
    ticker = row.findAll('td')[0].text
    tickers.append(ticker.strip())  # Remove potential leading/trailing whitespace

  # Download stock price data from Yahoo Finance for the specified period
  start = datetime.datetime(2016, 1, 1)
  end = datetime.datetime(2016, 12, 31)
  sp500 = yf.download(tickers, start=start, end=end)

  # Data cleaning and preprocessing
  sp500 = sp500.iloc[0:, 0:503]  # Select data (adjust indexing if necessary)
  sp500 = sp500.loc[:, sp500.columns.get_level_values(0) == 'Adj Close']  # Select 'Adj Close' prices
  sp500.columns = sp500.columns.droplevel()  # Remove multi-level indexing from columns
  sp500 = sp500.ffill().infer_objects(copy=False)  # Forward fill missing values and infer data types
  sp500 = sp500.dropna(axis=1)  # Drop columns with all NaN values

  # Save the data as a CSV file
  sp500.to_csv('data/sp500.csv', index=True)


def download_save_sp500():
  """
  This function downloads and saves the daily adjusted closing price of the SPDR S&P 500 ETF (SPY) 
  from Yahoo Finance for the year 2016. 

  The data is saved to a CSV file named 'data/spy.csv'.
  """

  # Define start and end dates for data download (2016)
  start = datetime.datetime(2016, 1, 1)
  end = datetime.datetime(2016, 12, 31)

  # Download price data for 'SPY' (SPDR S&P 500 ETF) from Yahoo Finance
  spy = yf.download('SPY', start=start, end=end)

  # Select only the 'Adj Close' price column
  spy = spy.loc[:, spy.columns.get_level_values(0) == 'Adj Close']

  # Forward fill missing values and infer data types (avoid unnecessary copying)
  spy = spy.ffill().infer_objects(copy=False)

  # Rename the 'Adj Close' column to 'spy' for clarity
  spy.rename(columns={'Adj Close': 'spy'}, inplace=True)

  # Drop columns with all NaN values
  spy = spy.dropna(axis=1)

  # Save the data as a CSV file
  spy.to_csv('data/spy.csv', index=True)

# Function to calculate logarithmic returns (consider reusing calc_log_ret)
def calculate_log_returns(df):
  """
  This function calculates the logarithmic returns (log returns) of a DataFrame containing financial data.

  Args:
      df (pandas.DataFrame): A pandas DataFrame containing a numeric column representing prices or values.

  Returns:
      pandas.Series: A pandas Series containing the calculated log returns for each period in the DataFrame.
  """
  return np.log1p(df.pct_change())

# Standardize log returns (consider reusing standardize)
def standardize_series(series):
  """
  This function standardizes a pandas Series (column) by subtracting the mean and dividing by the standard deviation.

  Args:
      series (pandas.Series): A pandas Series representing the data to be standardized.

  Returns:
      pandas.Series: A pandas Series containing the standardized data.
  """
  means = series.mean()
  stds = series.std()
  return (series - means) / stds

def plot_portfolio_components(weighted_ret_port1, weighted_ret_port2):
  """
  This function creates a time series plot comparing the returns of two portfolios.

  Args:
      weighted_ret_port1 (pandas.Series): A pandas Series representing the returns of portfolio 1.
      weighted_ret_port2 (pandas.Series): A pandas Series representing the returns of portfolio 2.
  """

  # Create a figure with two subplots (2 rows, 1 column) sharing the x-axis
  fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

  # Plot portfolio 1 returns in the first subplot
  ax1.plot(weighted_ret_port1, label='Portfolio 1', linewidth=0.5)
  ax1.set_title('Portfolio 1 (Weight)')
  ax1.set_ylabel('Returns')

  # Plot portfolio 2 returns in the second subplot
  ax2.plot(weighted_ret_port2, label='Portfolio 2', linewidth=0.5)
  ax2.set_title('Portfolio 2 (Weight/Volatility)')
  ax2.set_ylabel('Returns')
  ax2.set_xlabel('Date')  # Set x-axis label only on the bottom subplot

  # Add a title for the entire figure
  fig.suptitle('Portfolio Component Returns')

  # Optional layout adjustments
  plt.subplots_adjust(left=0.1)  # Adjust left margin for better readability
  plt.tight_layout()  # Automatically adjust spacing between subplots

  # Format y-axis labels to display 5 decimal places
  formatter = mticker.FormatStrFormatter('%.5f')
  ax1.yaxis.set_major_formatter(formatter)
  ax2.yaxis.set_major_formatter(formatter)

  # Set major x-axis ticks to occur monthly
  plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

  # Optional: Rotate x-axis labels for better readability if many data points
  plt.xticks(rotation=45)

  # Display the plot
  plt.show()

def plot_weighted_ret_port(weighted_ret_port):
  """
  This function creates a time series plot comparing the returns of SPY (S&P 500 ETF) 
  and two custom portfolios.

  Args:
      weighted_ret_port (pandas.DataFrame): A pandas DataFrame containing columns for 
          'spy' (SPY returns), 'spx_port1' (Portfolio 1 returns), and 'spx_port2' (Portfolio 2 returns).
  """
  # Get all column names excluding 'spy' (assuming 'spy' is always present)
  portfolio_columns = [col for col in weighted_ret_port.columns if col != 'spy']

  # Plot the returns of SPY, Portfolio 1, and Portfolio 2 on the same plot
  plt.plot(weighted_ret_port['spy'], label='SPY', linewidth=0.5)
  for col in portfolio_columns:
    plt.plot(weighted_ret_port[col], label=col, linewidth=0.5)

  # Optional layout adjustments
  plt.subplots_adjust(left=0.1)  # Adjust left margin for better readability
  plt.tight_layout()  # Automatically adjust spacing between elements

  # Format y-axis labels to display 5 decimal places
  formatter = mticker.FormatStrFormatter('%.5f')
  plt.gca().yaxis.set_major_formatter(formatter)

  # Add a title for the plot
  plt.title('Portfolio Log Returns')

  # Set major x-axis ticks to occur monthly
  plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

  # Add a legend to distinguish the plotted lines
  plt.legend(loc='lower right')  # Place the legend in the lower right corner

  # Customize x and y-axis labels (replace with your desired labels)
  plt.xlabel("Date")
  plt.ylabel("Returns")

  # Optional: Rotate x-axis labels for better readability with many data points
  plt.xticks(rotation=45)

  # Display the plot
  plt.show()

def plot_weighted_ret_port_cumm(weighted_ret_port_cumm):
  """
  This function creates a time series plot comparing the cumulative log returns of SPY (S&P 500 ETF) 
  and two custom portfolios.

  Args:
      weighted_ret_port_cumm (pandas.DataFrame): A pandas DataFrame containing columns for 
          'spy' (cumulative SPY returns), 'spx_port1' (cumulative Portfolio 1 returns), 
          and 'spx_port2' (cumulative Portfolio 2 returns).
  """
  # Get all column names excluding 'spy' (assuming 'spy' is always present)
  portfolio_columns = [col for col in weighted_ret_port_cumm.columns if col != 'spy']

  # Plot the returns of SPY, Portfolio 1, and Portfolio 2 on the same plot
  plt.plot(weighted_ret_port_cumm['spy'], label='SPY', linewidth=0.5)
  for col in portfolio_columns:
    plt.plot(weighted_ret_port_cumm[col], label=col, linewidth=0.5)

  # Optional layout adjustments
  plt.subplots_adjust(left=0.1)  # Adjust left margin for better readability
  plt.tight_layout()  # Automatically adjust spacing between elements

  # Format y-axis labels to display 5 decimal places
  formatter = mticker.FormatStrFormatter('%.5f')
  plt.gca().yaxis.set_major_formatter(formatter)

  # Add a title for the plot
  plt.title('Cumulative Portfolio Log Returns')

  # Set major x-axis ticks to occur monthly
  plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

  # Add a legend to distinguish the plotted lines
  plt.legend(loc='lower right')  # Place the legend in the lower right corner

  # Customize x and y-axis labels (replace with your desired label)
  plt.xlabel("Date")
  plt.ylabel("Cumulative Returns")  # Update label to reflect cumulative returns

  # Optional: Rotate x-axis labels for better readability with many data points
  plt.xticks(rotation=45)

  # Display the plot
  plt.show()


def calculate_sharpe_ratio(returns, risk_free_rate=.0024):
    """
    This function calculates the Sharpe ratio of a given return series.

    Args:
        returns (pandas.Series): A pandas Series containing the return values.
        risk_free_rate (float, optional): The risk-free rate of return. Defaults to 0.0.

    Returns:
        float: The Sharpe ratio of the return series.
    """
    # Calculate average return
    average_return = returns.mean()
    # Calculate standard deviation (volatility)
    volatility = returns.std()
    # Calculate excess return (difference between average return and risk-free rate)
    excess_return = average_return - risk_free_rate
    # Check for zero or negative volatility to avoid division by zero
    sharpe_ratio = excess_return / volatility
    return sharpe_ratio

def normalize_eigenvector(eigenvector):
  """
  This function normalizes an eigenvector to sum to 1.

  Args:
      eigenvector (np.ndarray): A numpy array representing the eigenvector to normalize.

  Returns:
      np.ndarray: The normalized eigenvector.
  """
  return np.real(eigenvector) / np.sum(np.real(eigenvector))

def filter_by_std_dev(data, filter_array, threshold_multiplier=2.5):
  """
  This function filters a dictionary or NumPy array and optionally filters a corresponding array based on standard deviation thresholds.

  Args:
      data (dict or np.ndarray): The data to be filtered (dictionary with numeric values or NumPy array).
      threshold_multiplier (float, optional): The multiplier for standard deviation to define the threshold above the mean. Defaults to 2.5.
      filter_array (np.ndarray, optional): An optional NumPy array to filter based on the same threshold derived from the data.

  Returns:
      dict or np.ndarray, (dict or np.ndarray, optional): The filtered data and optionally the filtered corresponding array.
  """

  # If data is a dictionary, convert values to a NumPy array
  if isinstance(data, dict):
    data_array = np.array([val for val in data.values()])
  else:
    data_array = data

  # Calculate mean and standard deviation
  mean_data = np.mean(data_array)
  std_dev_data = np.std(data_array)

  # Define threshold based on multiplier
  threshold = mean_data + threshold_multiplier * std_dev_data

  # Filter data based on threshold
  if isinstance(data, dict):
    filtered_data = {key: val for key, val in data.items() if val <= threshold}
  else:
    filtered_data = data_array[data_array <= threshold]

  # Filter optional array based on the same threshold
  if filter_array is not None:
    # Ensure filter_array has the same number of elements along the same axis as data
    if len(data_array.shape) != len(filter_array.shape):
      raise ValueError("filter_array must have the same number of dimensions as data")
    if data_array.shape != filter_array.shape:
      raise ValueError("filter_array dimensions must match data along filtering axis")
    filtered_array = filter_array[data_array <= threshold]

  return filtered_data, filtered_array

sp500 = pd.read_csv('data/sp500.csv', index_col=0)
spy = pd.read_csv('data/spy.csv', index_col=0)

# Calculate log returns for SPY and S&P 500 data
log_ret_spy = calculate_log_returns(spy)
log_ret_sp500 = calculate_log_returns(sp500)

log_ret_standardized = log_ret_sp500.apply(standardize_series, axis=0)

# Calculate correlation and covariance matrices
corr_matrix = log_ret_standardized.corr()
cov_matrix = log_ret_standardized.cov()

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Calculate weights for Portfolio 1 (weighting by eigenvector component 1)
eigenvectors_norm_1 = normalize_eigenvector(eigenvectors[:, 0])
weighted_ret_port1_components = log_ret_sp500 * eigenvectors_norm_1
weighted_ret_port1_components.dropna(axis=0, inplace=True)

# Calculate volatility of each stock
volatility = log_ret_sp500.std()

# Calculate weights for Portfolio 2 (weighting by eigenvector component 1 normalized by volatility)
eigenvectors_norm_2 = normalize_eigenvector(eigenvectors[:, 0] / volatility)
weighted_ret_port2_components = log_ret_sp500 * eigenvectors_norm_2
weighted_ret_port2_components.dropna(axis=0, inplace=True)

# # Plotting functions - not shown here (consider using as separate functions)
# plot_portfolio_components(weighted_ret_port1_components, weighted_ret_port2_components)

# Combine log returns into DataFrame for portfolio returns
weighted_ret_port = pd.DataFrame()
weighted_ret_port['spy'] = log_ret_spy.sum(axis=1)
weighted_ret_port['eigenvector'] = weighted_ret_port1_components.sum(axis=1)
weighted_ret_port['eigenvector/volatility'] = weighted_ret_port2_components.sum(axis=1)

# # Plotting functions - not shown here (consider using as separate functions)
# plot_weighted_ret_port(weighted_ret_port)

# Calculate cumulative returns for each portfolio
weighted_ret_port_cumm = pd.DataFrame()
weighted_ret_port_cumm['spy'] = weighted_ret_port['spy'].cumsum()
weighted_ret_port_cumm['eigenvector'] = weighted_ret_port['eigenvector'].cumsum()
weighted_ret_port_cumm['eigenvector/volatility'] = weighted_ret_port['eigenvector/volatility'].cumsum()

# # Plotting functions - not shown here (consider using as separate functions)
# plot_weighted_ret_port_cumm(weighted_ret_port_cumm)

weighted_ret_port_sharpe = calculate_sharpe_ratio(returns = weighted_ret_port_cumm)

print("The Sharpe ratio of the original weighted portfolio is : ", weighted_ret_port_sharpe, ".")

"""

As a general overview, what we've accomplished is that of creating a portfolio which in 
this case seeks to replicate/track the spx index. The individual companies of which the 
S&P 500 index is comprised have their data downloaded, cleaned, and is transformed to
provide us with the log returns of the individual components. It was asked that the returns
were then normalized and represented in a correlation matrix. I chose the covariance 
instead because this is what is usually used, and is in what I'm familiar. The eigenvalues
are then derived from the covariance matrix. What these eigenvalues and eigenvectors allow
us to do is understand how the individual component stocks drive portfolio returns. 

Now, this does not truly reflect the allocation weights used in the SPY ETF or the S&P 500
index itself, but it does help us understand the relative drivers. In this example, the 
eigenvectors were used as weights to demonstrate that by using the components derived we
could closely replicate the actual index. We can also see by the Sharpe ratio this 
replicated portfolio (Portfolio 1) maintains similar risk to that of the SPY ETF itself.

To provide a less risky option we also created a portfolio (Portfolio 2) which not only
takes the eigenvectors for weights but also divides the used eigenvector by the associated
stocks' volatility. Those stocks which have greater volatility will see a lower weight 
applied to their allocation as in our Portfolio 1 which uses the eigenvectors as weights
(weight(eigenvector)/volatility). There could be many reasons for this, however, one for
which I'm aware of is that of price shocks. Typically, when volatility spikes, in is not 
in the positive direction, but rather in the negative. By reducing the weights for stocks
which have higher volatility we can also reduce the associated negative effects to the 
portfolio.

This is what we see when we compare the Sharpe ratios of the two Portfolios. Portfolio 1
has a Sharpe ratio of .826693 while Portfolio 2 has a Sharpe of 1.005666. From what we
know about the portfolio allocation strategy of Portfolio 2 and the Sharpe methodology,
it is apparent that we are both seeking to reduce the volatility by decreasing the
allocation to volatile stocks while are increasing the allocation to the less volatile.

This presents us with intuitive insights and information. With this information we can
even seek out and remove the stocks altogether which have carry to much volatility. An
analysis could be performed as well to further increase the Sharpe of the portfolios.
Upon increasing the Sharpe, and while decreasing volatility we very well may also 
reduce the returns of the portfolios. However, this can be remedied through leverage.

This has been demonstrated below.

"""

# Filter volatility data
filtered_volatility, filtered_eigenvector = filter_by_std_dev(volatility, eigenvectors[:, 0], threshold_multiplier=0.0)
eigenvectors_norm_3 = normalize_eigenvector(filtered_eigenvector / filtered_volatility)

# Assuming the order of tickers is the same
filtered_log_ret_sp500 = log_ret_sp500[filtered_volatility.keys()]
weighted_ret_port3_components = filtered_log_ret_sp500 * eigenvectors_norm_3
weighted_ret_port3_components.dropna(axis=0, inplace=True)

weighted_ret_port['modified eigenvector/volatility'] = weighted_ret_port3_components.sum(axis=1)

# # Plotting functions - not shown here (consider using as separate functions)
# plot_weighted_ret_port(weighted_ret_port)

weighted_ret_port_cumm['modified eigenvector/volatility'] = weighted_ret_port['modified eigenvector/volatility'].cumsum()

# # Plotting functions - not shown here (consider using as separate functions)
# plot_weighted_ret_port_cumm(weighted_ret_port_cumm)

weighted_ret_port_sharpe = calculate_sharpe_ratio(returns = weighted_ret_port_cumm)
print("The Sharpe ratio of the modified weighted portfolio is : ")
print(weighted_ret_port_sharpe)

"""

As we can see in this evaluation we have nearly removed half of all stocks from the
portfolio for being to volatile, but what we do see is that we have increased
the Sharpe ratio by roughly 33 percent. There most certainly is a Sharpe maximizing
portfolio which would provide the greatest market opportunity. But for this analysis
we can certainly see that by removing the stocks with the greatest portfolio we have
increased the Sharpe. One can see that the returns are minutely weaker, but there
are remedies to this ass mentioned earlier.

"""

# # Plotting functions - not shown here (consider using as separate functions)
plot_weighted_ret_port_cumm(weighted_ret_port_cumm)