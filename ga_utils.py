import numpy as np
from ypstruct import structure

def run(problem, params):
    # Extract information
    costFunc  = problem.costFunc
    nVar      = problem.nVar
    varMin    = problem.varMin
    varMax    = problem.varMax
    maxIt     = params.maxIt
    nPop      = params.nPop
    pC        = params.pC
    nC        =  int(np.round(pC*nPop/2)*2) # Manipulating nC such that it is always an even number
    gamma     = params.gamma
    mu        = params.mu
    sigma     = params.sigma
    beta      = params.beta
    nparents   = params.parents

    # Empty idividual structure template
    emptyIndividual = structure()
    emptyIndividual.position = None
    emptyIndividual.cost = None


    # Keeping track of the best solution
    bestSol      = emptyIndividual.deepcopy() #.deepcopy makes sure that changes in bestSol doesnt affect emptyIndividual
    bestSol.cost = np.inf
    bestCost     = np.empty(maxIt)
    bestI1       = np.empty(maxIt)
    bestI2       = np.empty(maxIt)


    # Initializing a population
    pop = emptyIndividual.repeat(nPop)
    for i in range(0, nPop):
        pop[i].position = np.random.uniform(varMin, varMax, nVar)
        pop[i].cost = costFunc(problem, pop[i].position)
        #print("Cost: (THIS Is A TEST)   ",pop[i].cost)
        # Saving the best solution of every population
        if pop[i].cost < bestSol.cost:
            bestSol = pop[i].deepcopy()

    # Main loop for the iterations
    for it in range(maxIt):

        # Defining the probabilities of becoming a parent
        # probability will be the cost divided by cumulative cost
        costs   = np.array([x.cost for x in pop])
        costs_copy = costs.copy()
        avgCost = np.mean(costs)
        if avgCost !=0:
            costs = costs/avgCost
        probs = np.exp(-beta*costs)


        # Population of children
        popC = []
        for _ in range(nC//2): # //2 just to make sure that nC is an even number

            # Selecting parents
            # Roulette wheel selection
          # rndm = np.random.permutation(nPop)
          # parent1 = pop[rndm[0]]
          # parent2 = pop[rndm[1]]
            #Roulette wheel selection
            #parent1 = pop[rouletteWheel(probs)]
            #parent2 = pop[rouletteWheel(probs)]

            #Tournament selection
            indices = tournamentSelection(-costs_copy, nparents)
            parent1 = pop[indices[0]]
            parent2 = pop[indices[1]]

            # Making children and adding mutations
            child1, child2 = crossover(parent1, parent2, gamma)

            child1 = mutate(child1, mu, sigma)
            child2 = mutate(child2, mu, sigma)

            # Check boundaries of children
            applyBounds(child1, varMin, varMax)
            applyBounds(child2, varMin, varMax)

            #Evauation
            child1.cost = costFunc(problem, child1.position)
            if child1.cost < bestSol.cost:
                bestSol = child1.deepcopy()
            child2.cost = costFunc(problem, child2.position)
            if child2.cost < bestSol.cost:
                bestSol = child2.deepcopy()

            # Adding children to population
            popC.append(child1)
            popC.append(child2)


        # Merging populations and selecting the best individuals
        pop += popC
        pop = sorted(pop, key=lambda x: x.cost)
        pop = pop[0:nPop]

        # Adding the best cost solutions
        bestCost[it] = bestSol.cost
        bestI1[it] = problem.initInjValue + sum(bestSol.position[:4])
        bestI2[it] = problem.initInjValue + sum(bestSol.position[4:])

        # Printing to console
        print("Iteration {}:  Best Cost = {}  Best rates - Inj1: {}  Inj2: {} ".format(it+1,bestCost[it], bestI1[it], bestI2[it]))


    # Output structure
    out = structure()
    out.pop = pop
    out.bestSol = bestSol
    out.bestCost = bestCost
    out.bestI1 = bestI1
    out.bestI2 = bestI2
    return out

def crossover(p1,p2, gamma):
    c1 = p1.deepcopy()
    c2 = p1.deepcopy()
    alpha = np.random.uniform(-gamma,1+gamma,*c1.position.shape)
    c1.position = alpha*p1.position + (1-alpha)*p2.position
    c2.position = alpha * p2.position + (1 - alpha) * p1.position
    return c1,c2

def mutate(x, mu, sigma):
    y = x.deepcopy()
    temp = np.random.rand(*x.position.shape) <= mu
    temp = np.argwhere(temp)
    y.position[temp] += sigma*np.random.randn(*temp.shape)
    return y

def applyBounds(ind, varMin, varMax):


    ind.position = np.maximum(ind.position, varMin)
    ind.position = np.minimum(ind.position, varMax)


def rouletteWheel(prob):
    c   = np.cumsum(prob)
    rdm = sum(prob)*np.random.rand()
    indices = np.argwhere(rdm <= c)
    return indices[0][0]

def tournamentSelection(costs, parents):
    costs_copy = costs.copy()
    while len(costs)>parents:
        temp = []
        np.random.shuffle(costs)
        for i in range(len(costs)//2):
            if costs[i]>costs[len(costs)-1-i]:
                temp.append(costs[i])
            else:
                temp.append(costs[len(costs)-1-i])

        costs = temp

    indice1 = np.argwhere(costs_copy==costs[0])
    indice2 = np.argwhere(costs_copy==costs[1])
    indices = [indice1[0][0], indice2[0][0]]
    return indices