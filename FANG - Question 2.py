# -*- coding: utf-8 -*-
"""
FANG Portfolio Construction and Optimization - Q2.ipynb
"""
#Importing all the necessary modules
import datetime
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import seaborn as sns
import scipy.optimize as sco
%matplotlib inline
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# Constants
__MATPLOTLIB_STYLESHEET__ = 'fivethirtyeight'
__STOCKS__ = ['FB', 'AMZN', 'NFLX', 'GOOG']
# End Constants


# MatPlotLib stylesheet to use
plt.style.use(__MATPLOTLIB_STYLESHEET__)


# Methods
def download_stock_data(stock_list: [str], start_date: object, end_date: object):
    return web.DataReader(stock_list, 'yahoo', start_date, end_date)['Adj Close']

# Build the LSTM model to have four LSTM layers with 75 neurons and three Dense layers,
# one with 50 neurons, second with 25 and the last one with 1 neuron.
def LSTM_model(x_train, y_train):
    # Build the LSTM network model
    model = Sequential()
    model.add(LSTM(units=75, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=75, return_sequences=False))
    model.add(Dense(units=50))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    return model


def plot_time_series_graph(stock, train, valid):
    # Visualize the data
    plt.figure(figsize=(16, 8))
    plt.title('Model for ' + str(stock))
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Adj Close Price USD ($)', fontsize=18)
    plt.plot(train)
    plt.plot(valid)
    plt.legend(['Train', 'Validation', 'Predictions'], loc='lower right')
    plt.show()


def get_train_test_split(training_data_frame, predict_df, stock_name, scaler):
    res = []
    data = training_data_frame.filter([stock_name])
    dataset = data[stock_name].values
    scaled_data = scaler.fit_transform(data)
    training_data_len = math.ceil(len(dataset))
    res.append(scaled_data[0:training_data_len])

    data_test = predict_df.filter([stock_name])
    dataset_test = data_test[stock_name].values
    scaled_data_test = scaler.fit_transform(data_test)
    testing_data_len = math.ceil(len(dataset_test))
    res.append(scaled_data_test[0:testing_data_len])
    return res

start = datetime.datetime(2020, 1, 1)
end = datetime.datetime(2020, 12, 31)
df = download_stock_data(__STOCKS__, start, end)
df.columns = __STOCKS__

# Plot Adj Close Price History with the given data frame
plt.figure(figsize=(16, 8))
plt.title('Close Price History')
plt.plot(df)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Adj Close Price USD ($)', fontsize=18)
#plt.savefig('Stock_Price_History_Chart.png')
plt.show()
print("\n")

start = datetime.datetime(2021, 1, 1)
end = datetime.datetime(2021, 1, 31)
prediction_data_frame = web.DataReader(__STOCKS__, 'yahoo', start, end)['Adj Close']
prediction_data_frame.columns = __STOCKS__

scaler = MinMaxScaler(feature_range=(0, 1)) # For scaling the data before giving to the neural network

def create_train_dataset(stock):
  data = df.filter([stock])
  dataset = data[stock].values
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaled_data = scaler.fit_transform(data)
  training_data_len = math.ceil(len(dataset))
  train_data = scaled_data[0:training_data_len]
  #Split the data into x_train and y_train data sets
  x_train=[]
  y_train = []
  for i in range(0,len(train_data)-60):
      x_train.append(train_data[i:i+60,0])
      y_train.append(train_data[i+60,0])
  # Converting to numpy arrays so they can be used for training the LSTM model.
  x_train, y_train = np.array(x_train), np.array(y_train)
  # Reshape the data to be 3-dimensional because the LSTM model is expecting a 3-dimensional data set
  x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
  return data,dataset,training_data_len,train_data,x_train,y_train

def create_test_dataset(stock):
  start=datetime.datetime(2021, 1, 1)
  end=datetime.datetime(2021,1,31)
  df_test = web.DataReader(__STOCKS__, 'yahoo', start, end)['Adj Close']
  df_test.columns = ['FB','AMZN','NFLX','GOOG']
  # df_test = download_stock_data(__STOCKS__, datetime(2020, 1, 1), datetime(2020, 1, 31))
  data_test = df_test.filter([stock])
  dataset_test = data_test[stock].values
  scaled_data_test = scaler.fit_transform(data_test)
  testing_data_len = math.ceil(len(dataset_test))
  test_data = scaled_data_test[0:testing_data_len]
  #Create the x_test and y_test data sets
  x_test = []
  y_test = []
  for i in range(0,len(test_data)-10):
      x_test.append(test_data[i:i+10,0])
      y_test.append(test_data[i+10,0])
  x_test = np.array(x_test)
  # Reshape the data into the shape accepted by the LSTM
  x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
  return data_test,dataset_test,testing_data_len,test_data,x_test,y_test

#Methods used in Monte Carlo Simulation
def print_weights(weights):
    w = [round(i * 100, 2) for i in weights]
    w = pd.DataFrame(w, columns=['Weight (%)'], index=returns.columns)
    print("\n", w)

def calculate_return(weights):   # Computes the annualized return for the selected portfolio
    return np.dot(returns.mean(), weights) * 252

def calculate_std_dev(weights): # Computes the annualized volatility for the selected portfolio
    return math.sqrt(np.dot(weights, np.dot(returns.cov(), weights)) * 252)

def calculate_sharpe(weights): # Computes the Sharpe ratio for the selected portfolio. We assume a risk-free rate of 0.01
    return (calculate_return(weights) - risk) / calculate_std_dev(weights)

def min_max_sharpe(w):
    return -calculate_sharpe(w)

for stock in df:
  data,dataset,training_data_len,train_data,x_train,y_train = create_train_dataset(stock)
  data_test,dataset_test,testing_data_len,test_data,x_test,y_test = create_test_dataset(stock)
  model = LSTM_model(x_train,y_train)
  #Getting the models predicted price values
  predictions = model.predict(x_test)
  predictions = scaler.inverse_transform(predictions)#Undo scaling
  #print(predictions)
  rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
  #print(rmse)
  train = data[:training_data_len]
  valid = data_test[0:testing_data_len-10]
  valid['Predictions'] = predictions
  plot_time_series_graph(stock,train,valid)
print("\n")

plt.scatter(rets.mean(), rets.std())
plt.xlabel('Expected returns')
plt.ylabel('Standard deviations')
for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(
        label,
        xy=(x, y), xytext=(20, -20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
plt.show()
print("\n")
print("\n")
rets = df.pct_change()
# Histogram of Returns & Correlations
sns.pairplot(data=rets)
solar_corr = rets.corr()
print(solar_corr)
print("\n")
print(rets.mean())
print("\n")
print(rets.std())
print("\n")
returns = np.log(df / df.shift(1)).dropna()
returns.head()

# Initial Weight
eq_w = len(__STOCKS__) * [1 / len(__STOCKS__)]

# When no risk is involved:
risk = 0.01


print("Return: %s" % (calculate_return(eq_w)))
print("Standard Deviation: %s" % (calculate_std_dev(eq_w)))
print("Sharpe: %s" % (calculate_sharpe(eq_w)))
print("\n")

# Monte Carlo Simulation of 1,000 portfolios
np.random.seed(21)
num_ports = 1000
random_weights = np.random.random((num_ports, len(__STOCKS__))) #Random Portfolios Generator
random_weights = (random_weights.T / random_weights.sum(axis=1)).T

# Metrics Calculator
simulations = [(calculate_return(w), calculate_std_dev(w), calculate_sharpe(w)) for w in random_weights]
simulations = np.array(simulations)
simulations[:5]

print("Max Sharpe Ratio using Monte Carlo Simulation: {:.3f}".format(simulations[:, 2].max()))
print_weights(random_weights[simulations[:, 2].argmax(), :])

min_vol_ret = simulations[simulations[:, 1].argmin(), 0]
min_vol_vol = simulations[simulations[:, 1].argmin(), 1]

max_sharpe_ret = simulations[simulations[:, 2].argmax(), 0]
max_sharpe_vol = simulations[simulations[:, 2].argmax(), 1]

max_ret_ret = simulations[simulations[:, 0].argmax(), 0]

plt.figure(figsize=(10, 6))
plt.scatter(simulations[:, 1], simulations[:, 0], c=simulations[:, 2], cmap='Blues')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.title('Risk/Return for 1,000 Simulated Portfolios', fontsize=18, fontweight="bold", pad=20)
plt.scatter(max_sharpe_vol, max_sharpe_ret, c='orange', s=100, label='maximum sharpe')
plt.scatter(min_vol_vol, min_vol_ret, c='purple', s=100, label='minimum volatility')
plt.legend(frameon=False);
plt.show()
#plt.savefig('FANG_montecarlo.png', facecolor='None')
print("\n")


#Creating Efficient Frontier
frontier_y = np.linspace(min_vol_ret, max_ret_ret)

frontier_x = []

for possible_return in frontier_y:
    cons = ({'type': 'eq', 'fun': lambda eq_w: eq_w.sum() - 1},
            {'type': 'eq', 'fun': lambda eq_w: calculate_return(eq_w) - possible_return})
    bounds = len(__STOCKS__) * [(0, 1)]

    result = sco.minimize(calculate_std_dev, eq_w, bounds=bounds, constraints=cons)
    frontier_x.append(result['fun'])

plt.figure(figsize=(10, 6))
plt.scatter(simulations[:, 1], simulations[:, 0], c=simulations[:, 2], cmap='Blues')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.title('Efficient Frontier based on MC Simulation', fontsize=18, fontweight="bold", pad=20)
plt.plot(frontier_x, frontier_y, 'r--', linewidth=3)
plt.scatter(max_sharpe_vol, max_sharpe_ret, c='orange', s=100, label='maximum sharpe')
plt.scatter(min_vol_vol, min_vol_ret, c='purple', s=100, label='minimum volatility')
plt.legend(frameon=False);
#plt.savefig('FANG_frontier.png', facecolor='None')
plt.show()
print("\n")

#Computational Optimizer to find best Sharpe Ratio Portfolio
# Set constrains and initial guess
cons = {'type': 'eq', 'fun': lambda eq_w: eq_w.sum() - 1}
bounds = len(__STOCKS__) * [(0, 1)]

opt_results = sco.minimize(min_max_sharpe, eq_w, bounds=bounds, constraints=cons)
opt_w = opt_results['x']
print("\n")
print("Computational Optimized weights:")
print_weights(opt_w)
print("\n")
print("Indicators of Optimal Portfolio:")
print("Return {:.3f}, Volatility {:.3f}, Sharpe {:.3f}".format(calculate_return(opt_w), calculate_std_dev(opt_w),
                                                               calculate_sharpe(opt_w)))