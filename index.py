# Stock Predictions
# Data Source: https://www.nasdaq.com/market-activity/stocks/
# Author: William Rivera Ramos

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy import stats

sns.set_theme(color_codes=True)

# Load dataframe
df = pd.read_csv('Data/Microsoft_HistoricalData.csv', index_col='Date', parse_dates=True)

# Define plot size
plt.figure(figsize=(12,6))

# Set seed for reproductivity
np.random.seed(0)

print('Original Data Frame:\n', df.head())

# Rename column 'Close/Last' to 'Close'
df.rename(columns={'Close/Last': 'Close'}, inplace=True)

# Remove dollar sign from rows and replace them with nothing
price_close = df['Close'].str.replace('$', '', regex=False)
price_open = df['Open'].str.replace('$', '', regex=False)
price_high = df['High'].str.replace('$', '', regex=False)
price_low = df['Low'].str.replace('$', '', regex=False)

# Convert from object to float data types
df['Close'] = price_close.astype(float)
df['Open'] = price_open.astype(float)
df['High'] = price_high.astype(float)
df['Low'] = price_low.astype(float)

# Print Dataframe data types
print(df.dtypes)

# Get gain or lost by substracting Close less Open
gain_lost = df['Close'] - df['Open']

# Create new column 'Gain_Lost'
df['Gain_Lost'] = gain_lost

print('Modified Data Frame:\n', df.head())
print('\n', df.describe())

# Plot correlation heat map
sns.heatmap(df.corr(method='pearson'), cmap='Blues', annot=True)

# Data normalization
normalized_data = stats.boxcox(df.Close)

# Plot both together to compare (original and normalized data)
fig, ax = plt.subplots(1, 2, figsize=(12, 3))

sns.histplot(df.Close, ax=ax[0], kde=True, legend=False)
ax[0].set_title("Original Data")

sns.histplot(normalized_data[0], ax=ax[1], kde=True, legend=False)
ax[1].set_title("Normalized data")

# Target
y = df.Close

# Data Features
data_features = ['Open', 'Low', 'High', 'Gain_Lost']

X = df[data_features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Model
data_model = RandomForestRegressor(random_state=1)

# Fit
data_model.fit(train_X, train_y)

# Predict
predictions = data_model.predict(val_X)

# Close mean
close_mean = y.mean()

# Print MAE in dollars
val_MAE = mean_absolute_error(val_y, predictions)
print('\nMean Absolute Error: ${:,.2f}'.format(val_MAE))

# Print MAE in percent
val_MAE_percent = (val_MAE / close_mean) * 100
print('Mean Absolute Error: {:,.2f}'.format(val_MAE_percent),'%\n')

# Print the predictions
print('The Predictions are:\n', predictions)
print('\nPredictions Mean:', predictions.mean())
print('Predictions Max: ', predictions.max())
print('Predictions Min: ', predictions.min())

# Up/Down side
up_down_side = (predictions.max() - df.Close[0] ) / predictions.max() * 100
print('\nUp-Down side: {:,.2f}'.format(up_down_side), '%\n')

if up_down_side >= 20:
    print('BUY OR STRONG BUY!\n')
elif up_down_side > 15 and up_down_side < 20:
    print('MODERATE BUY!\n')
elif up_down_side > 10 and up_down_side < 15:
    print('HOLD!\n')
elif up_down_side > 5 and up_down_side < 10:
    print('MODERATE SELL!\n')
elif up_down_side < 5:
    print('SELL OR DONT BUY!\n')

# Create a CSV file with the predictions
output = pd.DataFrame({'Close': predictions})
output.to_csv('Data/Microsoft_stock_predictions.csv', index=False)

print('\nFile created successfully!\n')

# Plot predictions
sns.relplot(data=output, kind='line')
plt.show()