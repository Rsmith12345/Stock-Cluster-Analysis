import math
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.decomposition import PCA
import plotly.express as px

"""
Author: Richard Smith
Date: 8/4/2024

The following code implements a K-means clustering algorithm 
(made from scratch) for a list of stocks that the user can input
from a CSV file.  It clusters based on the following data:
mean daily returns (TTM), volatility (annualized TTM), volume (TTM),
forward PE ratio, and beta.  It then plots the graph of the clusters 
and prints out the info about the data point centroids for each cluster.

**Please note that the centroid data comes from a dataset where all
data categories have been scaled to a 1 to 10 system, so all values
should be viewed in relation to the other clusters.  For example,
if the Beta values for 4 clusters are ... 3.2  6.1  4.4  5.6 ...
then this does not mean that cluster 1 has an average beta
of 3.2, but it does mean that the stocks belonging to cluster 1
may be characterized as having lesser beta values
compared to those of the stocks belonging to other clusters.


This clustering has multiple potential uses.  It could be used as a tool
to aid in portfolio management, sub-categorize/analyze existing indices, or
find a starting place for the identification of pairs for a pairs-trading
investment strategy.
"""


def get_tickers(filename: str) -> list[str]:
    """
    Returns an array of stock tickers

     Args:
         filename (str): This is the name of the CSV file with the tickers.
                         It should be formatted to fit all on ONE line only!
                         See the example stocks.csv file if necessary.

    Returns:
        [str]: This returns a list of strings of the tickers in the file.
    """

    # make sure csv is one line long...
    ticker_frame = pd.read_csv(filename, header=None, dtype=str)
    ticker_list = ticker_frame.iloc[0].tolist()

    return ticker_list


def get_returns(ticker_list: list[str], valid_threshold: float) -> list[float]:
    """
    Finds the mean daily return (percent) based on TTM for each
    stock in the ticker list.

     Args:
         ticker_list [str]: This is the list of tickers you want to retrieve
                            data from.
         valid_threshold (float): This is the percent - as a decimal - of
                                  trading days from the past year that yfinance
                                  must have stock data for in order for the
                                  final TTM calculations for that stock to be
                                  considered valid.  For example, if yfinance
                                  is missing data from 50% of the trading days
                                  over the last year for a given stock and
                                  valid_threshold = .8, then it will input
                                  None as the final array value for that stock.

    Returns:
        [float]: This returns a list of floats for the mean daily return for
                 each stock for the TTM.
    """

    TRADING_DAYS = 252

    final_returns = []

    for ticker in ticker_list:

        annual_data = yf.Ticker(ticker).history(period='1y', interval='1d')

        # daily return percents
        daily_return_series = annual_data['Close'].pct_change()

        if len(daily_return_series) >= (TRADING_DAYS * valid_threshold):

            average_returns = daily_return_series.mean()
            final_returns.append(average_returns)

        else:

            final_returns.append(None)

    return final_returns


def get_volatility(
        ticker_list: list[str],
        valid_threshold: float
) -> list[float]:
    """
    Finds the price volatility based on TTM for each
    stock in the ticker list.

     Args:
         ticker_list [str]: This is the list of tickers you want to retrieve
                            data from.
         valid_threshold (float): This is the percent - as a decimal - of
                                  trading days from the past year that yfinance
                                  must have stock data for in order for the
                                  final TTM calculations for that stock to be
                                  considered valid.  See get_returns docstring
                                  for an example.

    Returns:
        [float]: This returns a list of floats for the volatility of each
                 stock for the TTM.
    """

    TRADING_DAYS = 252

    final_volatility_list = []

    for ticker in ticker_list:

        annual_data = yf.Ticker(ticker).history(period='1y', interval='1d')

        daily_return_series = annual_data['Close'].pct_change()

        if len(daily_return_series) >= (TRADING_DAYS * valid_threshold):

            # using annualized volatility formula, stdev of daily returns...
            # times the sqr root of 252 = volatility
            daily_volatility = daily_return_series.std()
            final_volatility_list.append(daily_volatility
                                          * math.sqrt(TRADING_DAYS))

        else:

            final_volatility_list.append(None)

    return final_volatility_list


def get_volume(ticker_list: list[str], valid_threshold: float) -> list[float]:
    """
    Finds the mean daily trading volume based on TTM for each
    stock in the ticker list.

     Args:
         ticker_list [str]: This is the list of tickers you want to retrieve
                            data from.
         valid_threshold (float): This is the percent - as a decimal - of
                                  trading days from the past year that yfinance
                                  must have stock data for in order for the
                                  final TTM calculations for that stock to be
                                  considered valid.  See get_returns docstring
                                  for an example.

    Returns:
        [float]: This returns a list of floats for the mean daily volume of
                each stock for the TTM.
    """

    TRADING_DAYS = 252

    final_volumes = []

    for ticker in ticker_list:

        annual_data = yf.Ticker(ticker).history(period='1y', interval='1d')

        daily_volume_series = annual_data['Volume']

        if len(daily_volume_series) >= (TRADING_DAYS * valid_threshold):

            daily_volume = daily_volume_series.mean()
            final_volumes.append(daily_volume)

        else:

            final_volumes.append(None)

    return final_volumes


def get_pe_ratio(ticker_list: list[str]) -> list[float]:
    """
    Finds the forward PE ratio for each stock in the ticker list.

    Args:
         ticker_list [str]: This is the list of tickers you want to retrieve
                            data from.

    Returns:
        [float]: This returns a list of floats for the forward PE of
                 each stock for the TTM.
    """

    pe_ratios = []

    for ticker in ticker_list:

        stock_info = yf.Ticker(ticker).info
        forward_pe = stock_info.get('forwardPE')

        pe_ratios.append(forward_pe)

    return pe_ratios


def get_beta(ticker_list: list[str]) -> list[float]:
    """
    Finds the historical beta for each stock in the ticker list.

     Args:
         ticker_list [str]: This is the list of tickers you want to retrieve
                            data from.

    Returns:
        [float]: This returns a list of floats for the beta of
                 each stock for the TTM.
    """

    betas = []

    for ticker in ticker_list:

        stock_info = yf.Ticker(ticker).info
        beta = stock_info.get('beta')

        betas.append(beta)

    return betas


def make_table(ticker_list: list[str], threshold: float) -> pd.DataFrame:
    """
    Creates a pd.DadaFrame of all stock factors described in the functions
    above.

     Args:
         ticker_list [str]: This is the list of tickers you want to retrieve
                            data from.
         threshold (float): This is to be used as the valid_threshold variable
                            for all the "get" functions above.  See the
                            get_return function docstring for more info.

    Returns:
        pd.DataFrame: This returns a DataFrame of different pieces of data
                      for each stock in the ticker list.
    """

    returns = get_returns(ticker_list, threshold)
    volatility = get_volatility(ticker_list, threshold)
    volume = get_volume(ticker_list, threshold)
    pe = get_pe_ratio(ticker_list)
    beta = get_beta(ticker_list)

    total_data = {

        'Ticker': ticker_list,
        'Mean Return': returns,
        'Volatility': volatility,
        'Volume': volume,
        'PE Ratio': pe,
        'Beta': beta

    }

    data_frame = pd.DataFrame(total_data)

    return data_frame


def make_centroids(data: pd.DataFrame, cluster_num: int) -> pd.DataFrame:
    """
    Creates initial centroids randomly for a k-clustering algorithm

     Args:
         data (pd.DataFrame): The dataframe made by the make_table function

         cluster_num (int): The number of clusters to initialize

    Returns:
        pd.DataFrame: This returns a DataFrame of the datapoints / coordinates
        of the clusters.
    """

    centroids_array = []

    for centroid in range(cluster_num):

        centroid = data.apply(lambda column: float(column.sample().iloc[0]))
        centroids_array.append(centroid)

    centroids_frame = pd.concat(centroids_array, axis=1)
    return centroids_frame


def get_clusters(data: pd.DataFrame, centroids: pd.DataFrame) -> pd.Series:
    """
    Labels each row of data with the cluster it belongs to (using cluster
    centroids).

     Args:
         data (pd.DataFrame): The dataframe made by the make_table function

         centroids (pd.DataFrame): The location of each cluster's centroid

    Returns:
        pd.Series: a series of cluster labels (ints) matching each row of the
                   input dataframe.
    """

    # creates a DataFrame of the distances between each stock and each...
    # centroid using a geometric distance formula for n-dimensions...
    # with the retrieved data being used as "coordinates"

    distances = (centroids.apply(lambda col:
                                 np.sqrt(((data - col) ** 2).sum(axis=1))))

    # pd.Series made from each stock and the cluster centroid it is closest to
    return distances.idxmin(axis=1)


def update_centroids(
        data: pd.DataFrame,
        cluster_labels: pd.Series
) -> pd.DataFrame:
    """
    Labels each row of data with the cluster it belongs to (using cluster
    centroids).

     Args:
         data (pd.DataFrame): The dataframe made by the make_table function

         cluster_labels (pd.Series): a series of the cluster labels for the
                                     data DataFrame

    Returns:
        pd.DataFrame: a DataFrame of the datapoints / coordinates for each
                      centroid of each cluster
    """

    # calculates the geometric mean of each cluster and returns the new mean...
    # datapoints / coordinates as a DataFrame describing the new centroids.
    # It also transposes the table to make it easier to work with later.

    return data.groupby(cluster_labels).apply(lambda point:
                                              np.exp(np.log(point).mean())).T


def plot_clustering(
        data: pd.DataFrame,
        cluster_labels: pd.Series,
        ticker_labels: pd.Series
) -> None:
    """
    This plots the graph of the clusters with ticker labels for each point.
    It uses plotly so the graph is more dynamic.

     Args:
         data (pd.DataFrame): The dataframe made by the make_table function

         cluster_labels (pd.Series): a series of the cluster labels for the
                                     data DataFrame

         ticker_labels (pd.Series): a series of the ticker labels for the
                                    data DataFrame

    Returns:
        None ... but it shows the graph in a webpage window.
    """

    # PCA to make the data 2d
    pca = PCA(n_components=2)
    graphable_data = pca.fit_transform(data)

    # Create a scatter plot using Plotly
    fig = px.scatter(
        x=graphable_data[:, 0],
        y=graphable_data[:, 1],
        color=cluster_labels,  # Color points by cluster
        text=ticker_labels,  # Add ticker labels
        title='K-Means Stock Clustering',
    )

    # label position
    fig.update_traces(textposition='top center')

    # more titles / labels
    fig.update_layout(
        title='K-Means Stock Clustering',
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2',
        autosize=True,
        margin=dict(l=40, r=0, b=40, t=40)
    )

    fig.show()


def main(filename: str) -> None:
    """
    The main method that inputs the filename, completes the k-clustering,
    and displays the results.

    Args:
        filename (str): the name of the csv file to be used.

    Returns:
        None ... but shows the graph and some info about the clusters
        in the terminal.
    """

    # converting the csv of tickers and making the stock datatable
    tickers = get_tickers(filename)
    stock_data = make_table(tickers, .9)

    # features to be considered for clustering
    features = ['Mean Return', 'Volatility', 'Volume', 'PE Ratio', 'Beta']

    # cleaning dataset to get rid of non-clusterable values (or lack thereof)
    stock_data = stock_data.dropna(subset=features)

    # making dataFrame without tickers and scaling data to 1-10 system
    no_tickers_data = stock_data[features].copy()
    no_tickers_data = (((no_tickers_data - no_tickers_data.min())
                        / (no_tickers_data.max() - no_tickers_data.min()))
                       * 9 + 1)

    # setting max iterations to find final cluster assignments and then...
    # setting number of clusters
    max_iterations = 120
    n_clusters = 4

    # the following is the core of the k-means clustering algorithm

    centroids = make_centroids(no_tickers_data, n_clusters)
    old_centroids = pd.DataFrame()
    iteration = 1

    while iteration < max_iterations and not centroids.equals(old_centroids):
        old_centroids = centroids

        cluster_labels = get_clusters(no_tickers_data, centroids)
        centroids = update_centroids(no_tickers_data, cluster_labels)

        iteration += 1

    # plotting the clusters
    plot_clustering(no_tickers_data, cluster_labels, stock_data['Ticker'])

    # printing info about clusters (remember, the data shown here was scaled)
    print(centroids)


# main method with input file
main('stocks.csv')
