import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta

# Set API credentials
api_key = 'AKNNSBS4RWQZDYMB5Y9Y'
api_secret = 'eqL6DrGzsCZfHbvrWP6mN2gwZAQhnOpFzx2fVB8a'
base_url = 'https://paper-api.alpaca.markets'

# Connect to API
api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')

# Get historical data for AAPL
symbol = 'AAPL'
timeframe = '1D'
start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')

historical_bars = api.get_bars(symbol, timeframe, start=start_date, end=end_date)

# Extract OHLCV data
data = {
    'time': [],
    'open': [],
    'high': [],
    'low': [],
    'close': [],
    'volume': []
}

for bar in historical_bars:
    data['time'].append(bar.t)
    data['open'].append(bar.o)
    data['high'].append(bar.h)
    data['low'].append(bar.l)
    data['close'].append(bar.c)
    data['volume'].append(bar.v)

# Convert data to pandas DataFrame
df = pd.DataFrame(data)

# Load data
X = df[['open', 'high', 'low', 'volume']].values
y = df['close'].values

# Scale data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.reshape(-1, 1))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model architecture
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=4))
model.add(Dense(1, activation='linear'))

# Compile model
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))

# Train model
model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1)

# Evaluate model
mse = model.evaluate(X_test, y_test, verbose=0)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

# Make predictions on test set
y_pred = model.predict(X_test)

# Inverse scale predictions and actual values
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

# Print predicted and actual values
for i in range(len(y_pred)):
    print('Predicted:', y_pred[i][0], 'Actual:', y_test[i][0])
