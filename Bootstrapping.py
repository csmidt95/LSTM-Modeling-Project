from sklearn.utils import resample
import pandas
import matplotlib.pyplot as plt
import numpy
import numpy as np
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility
numpy.random.seed(7)

# load dataset
dataframe = pandas.read_csv('Well2Cleaned.csv', engine = 'python')
dataset = dataframe.values
dataset = dataset.astype('float32')


# normalize the dataset
scaler = MinMaxScaler(feature_range = (0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * .67)
test_size = len(dataset) - train_size
train, test = dataset[0 : train_size,:], dataset[train_size:len(dataset),:]

# reshape into X=t and Y=t+1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

#Bootstrap data
#prepare bootstrap sample
bootX = resample(testX, replace=True, n_samples=60, random_state=1)
bootY = resample(testY, replace=True, n_samples=60, random_state=1)

#Concatenate old test sample with bootstrapped stample 
new_test_X = np.array(list(bootX) + list(testX))
new_test_Y = np.array(list(bootY) + list(testY))

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
new_test_X = numpy.reshape(new_test_X, (new_test_X.shape[0], 1, new_test_X.shape[1]))

#set the look_back
look_back = 17
# create and fit the LSTM network
model = Sequential()
#hidden layer with 4 LSTM blocks, or neurons
model.add(LSTM(4, return_sequences=True, input_shape = (1, look_back)))
model.add(LSTM(4, return_sequences=False, input_shape = (1, look_back)))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])
model.fit(trainX, trainY, epochs = 100, batch_size = 1, verbose = 2)
history = model.fit(trainX, trainY, epochs=100, validation_data = (new_test_X, new_test_Y), shuffle = False)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(new_test_X)
#invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
new_test_Y = scaler.inverse_transform([new_test_Y])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' %(trainScore))
testScore = math.sqrt(mean_squared_error(new_test_Y[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' %(testScore))














