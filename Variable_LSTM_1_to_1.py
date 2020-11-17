
import numpy as np
import tensorflow as tf
from keras.layers import Dense,LSTM
from keras.models import Sequential
from keras.utils import np_utils
from keras_preprocessing.sequence import pad_sequences

print("This is a type of stateless LSTM model.")
print("We can use arbritary long input sequence. ")
print("Here max_input_len is 5 , just for training purposes..")
print("The input sequences vary in length between 1 and max_len and therefore require zero padding."
      " Here, we use left-hand-side (prefix) padding with the Keras built in pad_sequences() function.")

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

def preprocessing_dataset(dataset):
    """
    This function splits the dataset into dataX and dataY...

    :param dataset: dataset
    :return: dataX, dataY
    """
    # prepare the dataset of input to output pairs encoded as integers
    num_inputs = 1000
    max_len = 5
    dataX = []
    dataY = []
    for i in range(num_inputs):
        start = np.random.randint(len(dataset) - 2)
        end = np.random.randint(start, min(start + max_len, len(dataset) - 1))
        sequence_in = dataset[start:end + 1]
        sequence_out = dataset[end + 1]
        dataX.append([char_to_int[char] for char in sequence_in])
        dataY.append(char_to_int[sequence_out])
        print(sequence_in, '->', sequence_out)

    return np.array(dataX),np.array(dataY)

dataX, dataY = preprocessing_dataset(dataset)

dataX, dataY = preprocessing_dataset(dataset)
print("DataX_shape : ", dataX.shape)
print("Data_Y_shape : ", dataY.shape)

max_len = 5
def preprocess_X_Y(dataX, seq_length, dataY):
    print("Preprocessing dataX and dataY ....")
    # reshape X to be [samples, time steps, features]  --> Input Expected by LSTM/RNN/GRU Layers
    X = pad_sequences(dataX, maxlen=max_len, dtype='float32')
    # reshape X to be [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], max_len, 1))

    # noramlise
    #X = X / len(dataX)
    # making y a categorical prediction 1 of 26 characters as next prediction
    y = np_utils.to_categorical(dataY)
    print("---------------------------------------")
    print("X-shape : ", X.shape)  # [batch_size,timesteps_per_sample,features_per_sample]
    print("y-shape : ", y.shape)

    return X, y


print("---------------------------------")


X, y = preprocess_X_Y(dataX, 1, dataY)
X = tf.convert_to_tensor(X)
y = tf.convert_to_tensor(y)


def model_1(X, y):
    model = Sequential()
    batch_size = 1
    model.add(LSTM(50, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(y.shape[1], activation='softmax'))
    print(model.summary())
    return model


model_11 = model_1(X, y)

model_11.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model_11.fit(X, y, epochs=500, batch_size=1)
batch_size = 1
for i in range(300):
    model_11.fit(X, y, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
    model_11.reset_states()


def scores(model_11, batch_size=1):
    # summarize performance of the model
    scores = model_11.evaluate(tf.convert_to_tensor(X), tf.convert_to_tensor(y), batch_size=batch_size, verbose=0)
    model_11.reset_states()
    print("Model Accuracy: %.2f%%" % (scores[1] * 100))
    return scores


scores = scores(model_11)
print("--------------------------------------")


def predict(dataX, dataset, model_11):
    # demonstrate some model predictions
    for pattern in dataX:
        x = pad_sequences([pattern], maxlen=max_len, dtype='float32')
        x = np.reshape(x, (1, max_len, 1))
        x = x / float(len(dataset))

        prediction = model_11.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        print(seq_in, "->", result)
    model_11.reset_states()


predict(dataX, dataset, model_11)


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