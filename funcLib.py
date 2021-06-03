import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json
import os


def runEclipse(filename):
    # This function runs Eclipse through a macro
    # Input is the name of the .DATA file

    os.system(r'C:\ecl\macros\eclrun.exe eclipse' + filename)

def latinHypercube(n):
    # This functino takes in an integer and generates a latin square
    # of size n. The samples will be between the range of 0 and 1
    # so it has to be sccaled up after using the function

    lowerLimits = np.arange(0,n)/n
    upperLimits = np.arange(1,n+1)/n
    points = np.random.uniform(low=lowerLimits, high=upperLimits, size=[2,n]).T
    np.random.shuffle(points[:,1])

    return points

def makeGaData(time, inj1, inj2):
    # This function takes in the time vector and values for both
    # of the injection rates and then it generates a dataset that
    # will be an input to the proxy model.

    gaData=np.array([[0,50,50]])
    for i in range(len(time)):
        gaData=np.append(gaData, [[time[i][0],inj1, inj2]], 0)

    gaData = np.delete(gaData, 0, 0)

    return gaData

def dataNormalization(x):
    # This function is used to normalize data.
    # The function adds a point at the beginning and end of each dataset
    # to force the data to be normalized within 0 and 150 Sm3/d

    lower_vals = [[0,0,0]]
    upper_vals = [[0,150,150]]
    scaling = MinMaxScaler(feature_range=(-1,1))
    x = np.insert(x,0,lower_vals,0)
    x = np.append(x,upper_vals,0)
    x = scaling.fit_transform(x)
    x = np.delete(x, 0, 0)
    x = np.delete(x, len(x)-1, 0)
    return x

def ANNmodel(train_data, test_data, train_labels, test_labels):
    # This function takes in the training and testing data.
    # The output is a trained ANN

    epc = 1000
    verbo = 1
    model = keras.Sequential([
        keras.layers.Dense(50, input_shape=(train_data.shape[1:]), activation="relu"),
        #keras.layers.Dropout(0.2),
        keras.layers.Dense(50, activation="relu"),
        #keras.layers.Dropout(0.2),
        keras.layers.Dense(50, activation="relu"),
        #keras.layers.Dropout(0.2),
        keras.layers.Dense(50, activation="relu"),
        #keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation="relu")
    ])
    opt = tf.keras.optimizers.Adam(learning_rate = 0.001)
    model.compile(optimizer=opt, loss="mean_squared_error", metrics=["mean_absolute_error"])
    model.fit(train_data, train_labels, epochs=epc, validation_data=(test_data, test_labels),verbose=verbo, batch_size=32)

    return model

def saveAnnModel(model, modelName):
    # This function saves the proxy model to a .json file

    weights = modelName + "_weights.h5"
    modelName = modelName + ".json"
    model_json = model.to_json()
    with open(modelName, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(weights)

def loadAnnModel(modelName):
    # This function takes in a model name and returns
    # a saved ANN
    
    weightsName = modelName + "_weights.h5"
    modelName = modelName + ".json"
    json_file = open(modelName, 'r')
    loaded_model = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model)
    model.load_weights(weightsName)
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error"])
    return model

def makeWCONINJE(inj1, inj2):
    # This function takes in two injection rates and creates a new schedule file for eclipse

    timestep = " TSTEP \n   1000*2 /\n"

    with open("ECL_test_folder/shedule_new.INC", 'w') as file:

        #Writing 8 lines for each of the injectors
        file.write("WCONINJE\n")
        for i in range(8):
            if(i<4):
                line = "'INJECT{}'   'WATER'	'OPEN'	'RATE'	{:.5f} 1* 420/ \n".format(i+1, inj1)
            else:
                line = "'INJECT{}'   'WATER'	'OPEN'	'RATE'	{:.5f} 1* 420/ \n".format(i+1, inj2)

            file.write(line)

        file.write("/\n \n")
        file.write(timestep)
        file.write(timestep)

def getFOPTfromRsm(filename):
    # This function finds the final value of FOPT from a .RSM-file
    # It requires that FOPT is the final output parameter of eclipse

    openfile = open(filename,'r')
    dummy = openfile.readline()
    dummy = openfile.readline()
    file = openfile.readlines()
    openfile.close()
    temp = []

    # Making an array of every datapoint from the .RSM file
    for lines in file:
        line = lines.split()
        for words in line:
            temp.append(words)

    # The last element of the array will be the final value for FOPT
    FOPT = float(temp[len(temp)-1])

    return FOPT

def GaToCsv(fopt, inj1, inj2):
    # This function saves the data from an optimization run with the GA
    # to a .csv-file such that the data can easily be imported in f.ex excel

    with open("GA_Output.csv", 'w') as file:
        file.write("ITERATION;FOPT;InjectorCluster1;InjectorCluster2\n")
        for i in range(len(fopt)):
            line = "{};{:.3f};{:.3f};{:.3f}\n".format(i+1,float(fopt[i]), float(inj1[i]), float(inj2[i]))
            file.write(line)

        file.close()