from keras.models import load_model
import json
import numpy as np
import random


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

    # print("Data successfully loaded!")

    return X, y

def predict(model, X, y):

    X = X[np.newaxis, ...]

    # prediction = [ [0.1, 0.2, ...] ] values of all different genre scores
    prediction = model.predict(X) # X -> (1, 130, 13, 1)

    # exstract index with max value
    predicted_index = np.argmax(prediction, axis=1) # [4]
    # print("Expected index: {}, Predicted index: {}".format(y, predicted_index))
    if y == predicted_index:
        print("prediction correct!")

    return predicted_index
    

def predictions():
    test_amount = 100
    correct_predictions = 0
    
    for i in range(test_amount): 
        X, y = load_data(DATA_PATH)

        # make a prediction on a sample
        random_index = random.randint(0, len(X))
        X = X[random_index] # random sample from data
        y = y[random_index]
        predicted_index = predict(model, X, y)
        if y == predicted_index:
            correct_predictions += 1
        print("number of correct predictions : " + str(correct_predictions))

    percentage_correct = correct_predictions / test_amount

    print("test accuracy was : " + str(percentage_correct*100) + "%")
        
        



if __name__ == "__main__":

    # To load the model later for predictions
    model = load_model("dementia_model.keras")

    predictions()
    
    

    
