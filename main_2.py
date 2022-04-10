import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

a=1255 #预测天数
#b=    #预测第n天

apple_training_complete = pd.read_csv('gold.csv')
apple_training_processed = apple_training_complete.iloc[:, 1:2].values
apple_testing_processed_d = apple_training_complete.iloc[0:a, 1:2].values

scaler = MinMaxScaler(feature_range=(0, 1))
apple_training_scaled = scaler.fit_transform(apple_training_processed)

features_set = []
labels = []
for i in range(61, a):
    features_set.append(apple_training_scaled[i - 60:i, 0])
    labels.append(apple_training_scaled[i, 0])

features_set, labels = np.array(features_set), np.array(labels)
features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(features_set, labels, epochs=30, batch_size=16)

apple_testing_complete = pd.read_csv('gold.csv')
apple_testing_processed = apple_testing_complete.iloc[:, 1:2].values
apple_total = pd.concat((apple_training_complete['gold'], apple_testing_complete['gold']), axis=0)
test_inputs = apple_total[len(apple_total) - len(apple_testing_complete) - 60:].values
test_inputs = test_inputs.reshape(-1, 1)
test_inputs = scaler.transform(test_inputs)
test_features = []
for i in range(60, a+60):
    test_features.append(test_inputs[i - 60:i, 0])
test_features = np.array(test_features)
test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))
predictions = model.predict(test_features)
predictions = scaler.inverse_transform(predictions)
np.savetxt("ep=20_all_g.csv", predictions, delimiter=',')
plt.figure(figsize=(10, 6))
plt.plot(apple_testing_processed_d, color='blue', label='Actual Bitcoin Price')
plt.plot(predictions, color='red', label='Predicted Bitcoin Price')
plt.title('Bitcoin Price Prediction')
plt.xlabel('Date')
plt.ylabel('Bitcoin Price')
plt.legend()
plt.show()