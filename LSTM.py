import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt


DATA_PATH = "data.json"


def load_data(data_path):
    """Loads training dataset from json file.
    
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    # convert lists into numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print("Data successfully loaded!")

    return X, y


def plot_history(history):

    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error subplot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


def prepare_datasets(test_size, validation_size):

    # load data
    X, y = load_data(DATA_PATH)

    # create train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # create the train/validation split
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):
    # Generate RNN_LSTM model

    # create model
    model = keras.Sequential()

    # 2 LSTM layers
    model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(64))

    # dense layer
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    #output layer
    model.add(keras.layers.Dense(2, activation='softmax')) # 10 for the amount of categories (music genres)

    return model

def predict(model, X, y):

    X = X[np.newaxis, ...]

    # prediction = [ [0.1, 0.2, ...] ] values of all different genre scores
    prediction = model.predict(X) # X -> (1, 130, 13, 1)

    # exstract index with max value
    predicted_index = np.argmax(prediction, axis=1) # [4]
    print("Expected index: {}, Predicted index: {}".format(y, predicted_index))



if __name__ == "__main__":

    # create train, validation, and test sets
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2) # test size and validation size as a percentage of total data


    # build the RNN net
    input_shape = (X_train.shape[1], X_train.shape[2]) # 130, 13
    model = build_model(input_shape)
    
    # compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])
    
    model.summary()
    
    # train the RNN
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=64, epochs=30) # you can tweak batch size and number of epochs

    # plot accuracy/error for training and validation
    plot_history(history)

    # evaluate the RNN on the test set
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print("\nTest accuracy: {}".format(test_accuracy))
    
    # make a prediction on a sample
    X = X_test[100] # random sample from data
    y = y_test[100]
    predict(model, X, y)

    X = X_test[101] # random sample from data
    y = y_test[101]
    predict(model, X, y)

    X = X_test[102] # random sample from data
    y = y_test[102]
    predict(model, X, y)

    X = X_test[103] # random sample from data
    y = y_test[103]
    predict(model, X, y)

    # Save the model
    model.save("dementia_model.keras")
