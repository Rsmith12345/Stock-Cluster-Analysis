# K-Means Stock Clustering

## Overview
The following code implements a K-means clustering algorithm 
(made from scratch) for a list of stocks that the user can input
from a CSV file.  It clusters based on the following data:
mean daily returns (TTM), volatility (annualized from TTM data), volume (TTM),
forward PE ratio, and beta.  It then plots the graph of the clusters 
and prints out info describing the data points of the centroids for each cluster.

This clustering program has multiple potential uses.  It could be used as a tool
to aid in portfolio management, sub-categorize/analyze existing indices, or
find a starting place for the identification of pairs for a pairs-trading
investment strategy.

## Main Features

- **K-means clustering algorithm**
- **Dynamic Visualization**
- **Easily adjustable features to use with clustering alg**

## Requirements (Necessary libraries)**

- **numpy**
- **pandas**
- **math**
- **yfinance**
- **sklearn** >> only Principal Component Analysis is used
- **plotly**

## Instructions

Update the name of the csv file input in the final line of code for the StockClustering.py
file to the one you want to use. Then run the StockCLustering.py file.

Also, if you wish to change the number of clusters, update line 441 in StockClustering.py.

## Important Notes

- Please ensure that the csv file is formatted as a SINGLE line of comma separated tickers
- Make sure all tickers are valid and match the format used by yahoo finance
- Note that if yahoo finance is missing one of the main pieces of data used in the clustering process for any given stock, then that stock will not be included on the graph / printed analysis
- The header numbers of the terminal-printed data table match the numeric labels for the clusters / color groups on the graph
- Please note the following for the printed data table of cluster centroid information: the centroid data comes from a dataset where all
data categories have been scaled to a 1 to 10 system, so all values
should be viewed in relation to the other clusters' centroid values.  For example,
if the Beta values for 4 clusters are shown to be ...| 3.2 |  6.1 | 4.4 | 5.6 |...
this does NOT mean that cluster 0 has an average beta
of 3.2, but it does mean that the stocks belonging to cluster 0
may be characterized as having lesser beta values
compared to those of the stocks belonging to other clusters.
- In my experience, plotlty may not be 100% reliable in terms of displaying the graph. If there is any error in displaying the graph, run the program again.
