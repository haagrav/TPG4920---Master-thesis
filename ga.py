import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ypstruct import structure
import ga_utils
import funcLib
import time

start = time.time()

# Load in the blind test in order to get the time vector. Could make a vector containing these values and achieve the same result.
blindTest = pd.read_csv("blind2.csv", sep=";")


def modelEvaluation(problem,x):
    # This is the objective function for the proxy model. The input is all problem parameters
    # and a chromosome. Then the function evaluates the FOPT.

    time = problem.time.copy()
    inj1 = problem.initInjValue + sum(x[:4])
    inj2 = problem.initInjValue + sum(x[4:])
    gaData = funcLib.makeGaData(time, inj1, inj2)
    gaData = funcLib.dataNormalization(gaData)
    model_prediction = problem.model.predict(gaData)
    FOPT = 2 * np.sum(model_prediction)

    return -FOPT

def runEclipse(problem,x):
    # This is the objective function for Eclipse.

    inj1 = problem.initInjValue + sum(x[:4])
    inj2 = problem.initInjValue + sum(x[4:])
    funcLib.makeWCONINJE(inj1, inj2)
    #Run eclipse

    FOPT = funcLib.getFOPTfromRsm("EGG_MODEL_ECL_GA.RSM")

    return -FOPT



# Problem defiinitions
problem               = structure()
problem.costFunc      = modelEvaluation # Defining the cost function
problem.nVar          = 8           # Length of chromosome. Try to keep it even
problem.initInjValue  = 50       # Initial value of 50
problem.varMin        = 0           # The values of the chromosome will vary between
problem.varMax        = 25          # 0 and 25 such that the maximum when added to the initial value will be within the proposed interval
problem.time          = np.array(blindTest[['TIME']])
problem.model         = funcLib.loadAnnModel("AnotherGoodModel")

# HEI HENRIK:
# Når du lager barn og sjekker mo de er innenfor varmin og varmax,
# sjekker vi bare at de er imellom 0 og 25 eller 50 og 150?

# GA parameters
params         = structure()
params.maxIt   = 50
params.nPop    = 16   # Initial population size
params.pC      = 0.75     # Prop of children in a population,
params.parents = 2     #only used for tournament selection
params.gamma   = 0.1
params.mu      = 0.75
params.sigma   = 0.1
params.beta    = 1



# Running the program
out = ga_utils.run(problem, params)
out.bestFopt = -out.bestCost

end = time.time()
print("Total elapsed time: ", end-start)


funcLib.GaToCsv(out.bestFopt, out.bestI1, out.bestI2)


plt.style.use('seaborn')

# plotting
#plt.semilogy(out.bestcost) #Logarithmic scale on y-axis
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

ax1.plot(out.bestFopt)
#ax1.xlim(0,params.maxIt)
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Best FOPT [Sm3]')
ax1.set_title('Genetic algorithm')
ax1.grid(True)


ax2.plot(out.bestI1, 'r', label="Injection rate 1")
ax2.plot(out.bestI2, 'b', label="Injection rate 2")
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Injection rate [Sm3/d]')
ax2.set_title('Injection rates')
ax2.legend(loc="lower right")
ax2.grid(True)


plt.show()
