## Project: Predicting Boston Housing Prices

## Project Overview
In this project, basic machine learning concepts are applied on data collected for housing prices in the Boston, Massachusetts area to predict the selling price of a new home. 

## Description
The Boston housing market is highly competitive, and you want to be the best real estate agent in the area. To compete with your peers, you decide to leverage a few basic machine learning concepts to assist you and a client with finding the best selling price for their home. Luckily, you\'ve come across the Boston Housing dataset which contains aggregated data on various features for houses in Greater Boston communities, including the median value of homes for each of those areas. Your task is to build an optimal model based on a statistical analysis with the tools available. This model will then be used to estimate the best selling price for your clients\' homes.

##Statistical Analysis
- Total number of houses: 489
- Total number of features: 3
- Minimum house price: 105,000.00
- Maximum house price: 1,024,000.00
- Mean house price: 454,342.90
- Median house price: 438,900.00
- Standard deviation of house price: 165,340.30

## Evaluating model performance
The problem of predicting the housing prices is not a classification problem since the numbers changing with the time. So it is a Regression problem and uses regression problem's evaluation metrics for model evaluation.


## Software and Libraries
This project uses the following software and Python libraries:

- [Python 2.7](https://www.python.org/download/releases/2.7/)
- [NumPy](http://www.numpy.org/)
- [pandas](http://pandas.pydata.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [matplotlib](http://matplotlib.org/)
- [Jupyter Notebook](http://ipython.org/notebook.html).

Files

- `boston_housing.ipynb`: This is the main file of the project
- `housing.csv`: The project dataset.
- `visuals.py`: This Python script provides supplementary visualizations for the project.
#Run

In a terminal or command window, navigate to the top-level project directory boston_housing/ (that contains this file) and run one of the following commands:

jupyter notebook boston_housing.ipynb

This will open the jupyter Notebook software and project file in browser.