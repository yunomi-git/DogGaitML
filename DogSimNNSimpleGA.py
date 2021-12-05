# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 17:43:38 2021

@author: Evan Yu
"""

import numpy as np;
from GeneticOptimizer import SimpleGAOptimizer, SimpleGAParameters
from Optimizer import OptimizationEndConditions
from FootModelNeuralNet import NNFootModelSimplest
from Simulation import BatchSimulation, Simulation, Curriculum
from DogUtil import DogModel, State, TaskMotion
# import matplotlib.pyplot as plt;
# import pyglet
# from pyglet.window import mouse
# from VisualizerSimulation import VisualizerSimulation
import pickle
import os
import time


subFolderName = "GA_NNSimpleModel"
prefix = "12-4-2021_GA_NNSimpleModelNoConvergenceWeight"
suffix = "_0"
doRunOptimizer = True
# doRunOptimizer = False

# -----------------------------------------------------------------------
def generateInitialStatesList():
    dogModel = DogModel()
    initialCOM = np.array([-20.0, 0.01])
    initialFootState = dogModel.defaultFootState - initialCOM
    initialState = State(initialFootState, 0.)
    return [initialState]

def generateTaskMotionsList():
    return [TaskMotion(5., 0.1, 0.1)]
    # return [TaskMotion(5., 0.1, 0.1),
    #         TaskMotion(0.1, 4., 0.1),
    #         TaskMotion(0.1, 0.1, 2.)]

footModel = NNFootModelSimplest()
numParameters = footModel.getNumParameters()

scale = 200.
populationSize = 200

initialParameters = np.random.rand(populationSize, numParameters) * scale - scale/2
initialStatesList = generateInitialStatesList()
desiredMotionsList = generateTaskMotionsList()

costWeights = np.array([0.01,0.01,
                        0.01,0.01,
                        0.0,
                        300.,
                        100.])
curriculum = Curriculum()
curriculum.setVelocityCostPerFootstep(distVelocityErrorIncreasePerStep=0.5, 
                                      angVelocityErrorIncreasePerStep=0.5)
curriculum.setFootDistanceCostPerFootstep(footDistCostPerStep=5)
numSteps = 4

optimizationParameters = SimpleGAParameters(crossoverRatio=0.5, 
                                            mutationMagnitude=15.0,
                                            decreaseMutationMagnitudeEveryNSteps=50,
                                            mutationMagnitudeLearningRate=0.7,
                                            mutationChance=1.0,
                                            decreaseMutationChanceEveryNSteps=200,
                                            mutationChanceLearningRate=0.9,
                                            mutateWithNormalDistribution=False,
                                            mutationLargeCostScalingFactor=40.0,
                                            diversityChoiceRatio = 0.3,
                                            varianceMutationMaxMagnitude = 10.);  

optimizationEndConditions = OptimizationEndConditions(maxSteps=10000,
                                                      convergenceThreshold=0.0)

printEveryNSteps = 100

# ============================================================================

simulationName = prefix + suffix
path = ".\\data\\" + subFolderName + "\\"
filename =  path + simulationName + '.pickle'

def main():
    if os.path.exists(filename):
        overwrite = input("File path exists. Overwrite? <1> yes | <else> no: ")
        if (overwrite != '1'):
            print("Aborting")
            raise SystemExit
        else:
            print("Starting Simulation")
    if not os.path.isdir(path):
        os.mkdir(path)
        
    if doRunOptimizer:
        runOptimizer()

def runOptimizer():    
    costEvaluator = BatchSimulation(initialStatesList = initialStatesList, 
                                    footModel = footModel, 
                                    desiredMotionsList = desiredMotionsList, 
                                    numSteps = numSteps, 
                                    costWeights = costWeights)
    costEvaluator.setCurriculum(curriculum)
    optimizer = SimpleGAOptimizer(initialParameters, costEvaluator, optimizationParameters)
    optimizer.printEveryNSteps = printEveryNSteps
    optimizer.setOptimizationEndConditions(optimizationEndConditions)
    
    start_time = time.time()
    try:
        while (not optimizer.hasReachedEndCondition()):
            optimizer.step();
            if (optimizer.stepCount % optimizer.printEveryNSteps == 0):
                print("step: " + str(optimizer.stepCount))
                value, cost = optimizer.getCurrentStateAndCost()
                print("cost: " + str(cost))
    except KeyboardInterrupt:
        pass
    print("Time elapsed: " + str(time.time() - start_time))

    parameterHistory, costHistory = optimizer.getFullHistory();
    bestCostIndex = np.argmin(costHistory)
    finalParameters = parameterHistory[bestCostIndex, :]
    # finalParameters = parameterHistory[-1,:]
    
    print(finalParameters)
    
    simData = {}
    simData["parameterHistory"] = parameterHistory
    simData["costHistory"] = costHistory
    simData["optimizationParameters"] = optimizationParameters
    simData["optimizationPopulationSize"] = populationSize
    simData["optimizationEndConditions"] = optimizationEndConditions
    simData["simNumFootSteps"] = numSteps
    simData["simCostWeights"] = costWeights
    simData["simInitialStatesList"] = initialStatesList
    simData["simDesiredMotionsList"] = desiredMotionsList
    simData["finalParameters"] = finalParameters
    simData["footModel"] = footModel
    simData["curriculum"] = curriculum
    
    with open(filename, 'wb') as handle:
        pickle.dump(simData, handle)
    
    print("saved")
    
if __name__ == "__main__":
    main()
