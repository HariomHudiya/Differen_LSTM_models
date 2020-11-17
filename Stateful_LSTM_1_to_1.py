
# Importing Relevant packages

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils

print("Ideally, we want to expose the network to the entire sequence and let it learn the inter-dependencies, "
      "rather than us define those dependencies explicitly in the framing of the problem")

print("We can do this in Keras by making the LSTM layers stateful and manually resetting the state of the network "
      "at the end of the epoch, "
      "which is also the end of the training sequence.")

print("Highlights :")
print("Here in this Example batch_size = 1")
print(" we must explicitly specify the batch size as a dimension on the input shape. "
      "This also means that when we evaluate the network or make predictions, we must also specify and adhere to this same batch size. "
      "This is not a problem now as we are using a batch size of 1.")

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

char_to_int , int_to_char = mapping(dataset,debugging_info=False)
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


dataX,dataY = prepare_dataset(dataset,1,char_to_int,debugging_info=True)
print("DataX_shape : ",dataX.shape)
print("Data_Y_shape : ",dataY.shape)

def preprocess_X_Y (dataX,seq_length , dataY):
    print("Preprocessing dataX and dataY ....")
    # reshape X to be [samples, time steps, features]  --> Input Expected by LSTM/RNN/GRU Layers
    X = np.reshape(dataX, (len(dataX), seq_length, 1))
    # noramlise
    X = X / len(dataX)
    # making y a categorical prediction 1 of 26 characters as next prediction
    y = np_utils.to_categorical(dataY)
    print("---------------------------------------")
    print("X-shape : ", X.shape)      #[batch_size,timesteps_per_sample,features_per_sample]
    print("y-shape : ", y.shape)

    return X,y

print("---------------------------------")

X,y = preprocess_X_Y(dataX,1,dataY)

def model_1(X,y):

    model = Sequential()
    batch_size = 1
    model.add(LSTM(50, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(y.shape[1],activation='softmax'))
    print(model.summary())
    return model
model_11 = model_1(X,y)

model_11.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model_11.fit(X, y, epochs=500, batch_size=1)
batch_size = 1
for i in range(300):
    model_11.fit(X, y, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
    model_11.reset_states()

def scores(model_11,batch_size = 1):
    # summarize performance of the model
    scores = model_11.evaluate(X, y,batch_size=batch_size, verbose=0)
    model_11.reset_states()
    print("Model Accuracy: %.2f%%" % (scores[1] * 100))
    return scores

scores  = scores(model_11)
print("--------------------------------------")
def predict(dataX,dataset,model_11):
    # demonstrate some model predictions
    print("Making Predictions......")
    for pattern in dataX:
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(len(dataset))
        prediction = model_11.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        print(seq_in, "->", result)
    model_11.reset_states()

predict(dataX,dataset,model_11)

def random_prediction(dataset):
    # demonstrate predicting random patterns
    print("Test a Random Pattern:")

    for i in range(0, 20):
        pattern_index = np.random.randint(len(dataX))
        pattern = dataX[pattern_index]
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(len(dataset))
        prediction = model_11.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        print(seq_in, "->", result)

    model_11.reset_states()

random_prediction(dataset)