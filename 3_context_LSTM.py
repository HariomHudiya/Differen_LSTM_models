# Importing Relevant packages
# Each element in the sequence is then provided as a new input feature to the network.

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils

print("Giving more Context by providing bigger Sequence at each time-Step")
print("Hence Seq_length is #Number of Features(per_time_step) instead of each character being a different timestep")
np.random.seed(7)

dataset = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def mapping(dataset,debugging_info = True):
	print("Creating Mapping")
	vocab = sorted(set(dataset))
	char_to_int = {c : i for i,c in enumerate(vocab)}
	int_to_char = {i : c for i,c in enumerate(vocab)}

	if debugging_info:
		print("Vocab :" ,  len(vocab))
		print(vocab)
		print("-------------------------------")
		print("Char to Int")
		print(char_to_int)
		print("-------------------------------")
		print("Int to Char")
		print(int_to_char)

	return char_to_int,int_to_char

char_to_int , int_to_char = mapping(dataset,debugging_info=True)
print("---------------------------------------------------")

# Preparing Dataset

def prepare_dataset(dataset,seq_length , char_to_int,debugging_info =True):
	# prepare the dataset of input to output pairs encoded as integers
	print("Preparing Dataset")
	seq_length = seq_length
	dataX = []
	dataY = []
	for i in range(0, len(dataset) - seq_length, 1):
		seq_in = dataset[i:i + seq_length]
		seq_out = dataset[i + seq_length]
		dataX.append([char_to_int[char] for char in seq_in])
		dataY.append(char_to_int[seq_out])

		if debugging_info:
			print(seq_in, '->', seq_out)
	return np.array(dataX),np.array(dataY)


dataX,dataY = prepare_dataset(dataset,3,char_to_int,debugging_info=True)
print("DataX_shape : ",dataX.shape)
print("Data_Y_shape : ",dataY.shape)

def preprocess_X_Y (dataX,seq_length , dataY):
	print("Preprocessing dataX and dataY ....")
	# reshape X to be [samples, time steps, features]  --> Input Expected by LSTM/RNN/GRU Layers
	X = np.reshape(dataX, (len(dataX), 1, seq_length))
	# noramlise
	X = X / len(dataX)
	# making y a categorical prediction 1 of 26 characters as next prediction
	y = np_utils.to_categorical(dataY)
	print("---------------------------------------")
	print("X-shape : ", X.shape)      #[batch_size,timesteps_per_sample,features_per_sample]
	print("y-shape : ", y.shape)

	return X,y

print("---------------------------------")

X,y = preprocess_X_Y(dataX,3,dataY)

def model_1(X,y):
	model = Sequential()
	model.add(LSTM(32 , input_shape = (X.shape[1], X.shape[2])))
	model.add(Dense(y.shape[1],activation='softmax'))
	print(model.summary())
	return model
model_11 = model_1(X,y)

model_11.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_11.fit(X, y, epochs=500, batch_size=1)

def scores(model_11):
	# summarize performance of the model
	scores = model_11.evaluate(X, y, verbose=0)
	print("Model Accuracy: %.2f%%" % (scores[1] * 100))
	return scores

scores  = scores(model_11)
print("--------------------------------------")
def predict(dataX,dataset,model_11):
	# demonstrate some model predictions
	for pattern in dataX:
		x = np.reshape(pattern, (1, 1 , len(pattern)))
		x = x / float(len(dataset))
		prediction = model_11.predict(x, verbose=0)
		index = np.argmax(prediction)
		result = int_to_char[index]
		seq_in = [int_to_char[value] for value in pattern]
		print(seq_in, "->", result)

predict(dataX,dataset,model_11)