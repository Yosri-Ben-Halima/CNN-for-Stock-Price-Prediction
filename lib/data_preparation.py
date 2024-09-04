import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

def load_data(ticker, start_date, end_date):
    """
    Downloads and prepares stock data using yfinance.

    Parameters
    ----------
    ticker : str
        The stock ticker symbol of the company (e.g., 'AAPL' for Apple).
    start_date : str
        The start date for the data in 'YYYY-MM-DD' format.
    end_date : str
        The end date for the data in 'YYYY-MM-DD' format.

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame containing the historical 'Close' prices of the stock,
        sorted by date in ascending order.

    Description
    -----------
    This function fetches historical stock data for the given `ticker` from
    `start_date` to `end_date` using the `yfinance` library. It extracts the
    'Close' price column, sorts the data by date, and returns the resulting
    DataFrame.
    """

    df = yf.download(ticker, start=start_date, end=end_date)
    df = df[['Close']]
    df.sort_index(inplace=True)
    return df

def prepare_data(df, window_size=10):
    """
    Prepares the data for model input.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame containing the stock data, with a 'Close' price column.
    window_size : int, optional
        The size of the window (number of days) to use for creating input
        sequences for the model. Default is 10.

    Returns
    -------
    data : numpy.ndarray
        A 4D array of shape (number of samples, window_size, 1, 1) containing
        the scaled 'Close' prices.
    labels : numpy.ndarray
        A 2D array of shape (number of samples, 1) containing the 'Close' prices
        for the day following each input sequence.

    Description
    -----------
    This function prepares the input data for a deep learning model by
    creating sequences of 'Close' prices from the stock data. It creates a
    sliding window over the data, normalizes the input sequences using 
    `StandardScaler`, and reshapes the data to fit the model's expected input
    format. The labels are the 'Close' prices for the day following each input
    sequence.
    """

    data = []
    labels = []
    for i in range(len(df) - window_size):
        window = df['Close'].iloc[i:i + window_size].values
        data.append(window)
        labels.append(df['Close'].iloc[i + window_size])
    
    data = np.array(data)
    labels = np.array(labels)
    
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    data = data.reshape((data.shape[0], window_size, 1, 1))
    labels = labels.reshape(-1, 1)
    
    return data, labels
