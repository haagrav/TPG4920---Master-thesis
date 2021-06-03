import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pandas as pd
import funcLib
import matplotlib.pyplot as plt


# Gathering data from.csv files
data            = pd.read_csv("LHS_30Samples.csv", sep=";")
blindTest       = pd.read_csv("blind2.csv", sep=";")
data            = data.fillna(0)
blindTest       = blindTest.fillna(0) # replace all NA with 0

# Specify which columns of data that is desired.
data            = data[["TIME","FOPR","WWIR_I1","WWIR_I8"]]
blindTest       = blindTest[["TIME","FOPR" ,"WWIR_I1","WWIR_I8"]]
time            = np.array(blindTest[["TIME"]])
time            = np.delete(time,0,0)       #Deleting the first instance for normalization purposes
blind_fopr      = np.array(blindTest[["FOPR"]])
blind_fopr      = np.delete(blind_fopr,0,0)
fopr_y          = np.array(data["FOPR"])

# Remove FOPT from the data
data.drop(columns=['FOPR'], axis=1)
del data["FOPR"]

blindTest.drop(columns=['FOPR'], axis=1)
del blindTest["FOPR"]


# Split the data into training and testing data
train_data, test_data, train_fopr, test_fopr = train_test_split(data, fopr_y, test_size=0.2, shuffle=False)


blindTest = np.array(blindTest)
blindTest = np.delete(blindTest,0,0)

# Normalizing data
train_data      = funcLib.dataNormalization(np.array(train_data))
test_data       = funcLib.dataNormalization(np.array(test_data))
blindTest       = funcLib.dataNormalization(blindTest)
train_data      = np.delete(train_data,0,0)
train_fopr      = np.delete(train_fopr,0,0)


#Setting up a model
model       = funcLib.ANNmodel(train_data, test_data,train_fopr, test_fopr)

#Saving the model
funcLib.saveAnnModel(model, "NN_model")

#Loading the model
#model = funcLib.loadAnnModel("AnotherGoodModel")


#Making model evaluations
fopr_train_pred = model.predict(train_data).flatten()
fopr_test_pred = model.predict(test_data).flatten()
blind_pred = model.predict(blindTest)

#The value of FOPT is equal to the sum of all the injection rates times the timestep
FOPT_Pred = 2*np.sum(blind_pred)
FOPT_blind = 2*np.sum(blind_fopr)

# ------------Calculating the ANN metrics--------------
print("R2 scores for train-, test- , and blind data: ")
print(r2_score(train_fopr, fopr_train_pred))
print(r2_score(test_fopr, fopr_test_pred))
print(r2_score(blind_fopr,blind_pred))
print("MSE scores for train-, test- , and blind data: ")
print(mean_squared_error(train_fopr, fopr_train_pred))
print(mean_squared_error(test_fopr, fopr_test_pred))
print(mean_squared_error(blind_fopr, blind_pred))
print("MAE scores for train-, test- , and blind data: ")
print(mean_absolute_error(train_fopr, fopr_train_pred))
print(mean_absolute_error(test_fopr, fopr_test_pred))
print(mean_absolute_error(blind_fopr, blind_pred))


plt.style.use('seaborn')

#Alt plot
time2 = time[:350]
blind_fopr2 = blind_fopr[:350]
blind_pred2 = blind_pred[:350]
print("Length of blind fopr: ", len(test_fopr), "\nLength of the test prediction: ", len(fopr_test_pred))

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()

ax1.plot(time, blind_fopr, 'r', label="ECLIPSE")
ax1.plot(time, blind_pred,color = 'b', linestyle = '--', dashes=(5,5), label="Proxy model")
ax1.set_xlabel("Time [d]")
ax1.set_ylabel("FOPR [Sm3/d]")
ax1.set_title("ECLIPSE vs ANN FOPR")
ax1.legend(loc="upper right")
#ax1.grid()


ax2.plot(time2, blind_fopr2, 'r', label="ECLIPSE")
ax2.plot(time2, blind_pred2,color = 'b', linestyle = '--', dashes=(5,5), label="Proxy model")
ax2.set_xlabel("Time [d]")
ax2.set_ylabel("FOPR [Sm3/d]")
ax2.set_title("ECLIPSE vs ANN FOPR")
ax2.legend(loc="upper right")
#ax1.grid()

ax3.plot(test_fopr-fopr_test_pred)


plt.show()
