# TPG4920---Master-thesis
This is the python code that has been used for the work on my masters thesis in reservoir engineering.

The main script is in the file nn_setup.py. This script includes the import of data from .csv file, preprocessing of data, and the training of the artificial neural network.
It alo includes some plotting at the bottom of the script. 

The code that is called ga.py is the main script for the genetic algorithm. It is in this file that problem specifications are done, such as objective function and population size etc.
The script that is called ga_utils.py containt the implementatino of the GA, as well as all the implementatino for the crossover operation, parent selection, and mutation.

The script that is called funcLib.py containt a lot of the function that are used in both ga.py and nn_setup.py.
It is a librairy of function such as the functino that generates an ANN, normalizes data, and generates the data for the GA.

For any questions, please contact me on haagrav@gmail.com.

