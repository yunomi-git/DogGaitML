# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 18:57:23 2022

@author: Evan Yu
"""

import numpy as np;
from GeneticOptimizer import SimpleGAOptimizer, SimpleGAParameters
from Optimizer import OptimizationEndConditions
from FootModelNeuralNet import NNFootModelSimplest
from Simulation import Simulation, CostWeights
from BatchSimulation import StochasticBatchSimulationFromFile
from DogUtil import DogModel, State, TaskMotion
import pickle
import os
import time


subFolderName = "GA_DrStrange"
prefix = "02-20-2022"
suffix = "_01"
doRunOptimizer = True
# doRunOptimizer = False

startFromPreviousParameters = False
prevPrefix = "none"
prevSuffix = "_01"

datasetPath = ".\\data\\"
datasetName = "TomatoTraining.dat"
datasetFileName = datasetPath + datasetName


simulationName = prefix + suffix
path = ".\\data\\" + subFolderName + "\\"
filename =  path + simulationName + '.pickle'

prevSimulationName = prevPrefix + prevSuffix
prevPath = ".\\data\\" + subFolderName + "\\"
prevFilename =  prevPath + prevSimulationName + '.pickle'


# -----------------------------------------------------------------------


def generateInitialParametersAround(parametersCenter):
    initialParameters = np.random.rand(populationSize, numParameters) * scale - scale/2
    if parametersCenter is not None:
        initialParameters[0,numParameters] = np.zeros(numParameters)
        initialParameters += parametersCenter
    return initialParameters

def getPreviousParameters(filename):
    with open(filename, 'rb') as handle:
        simData = pickle.load(handle)
    parameterHistory = simData["parameterHistory"]
    costHistory = simData["costHistory"]    
        
    bestCostIndex = np.argmin(costHistory)
    finalParameters = parameterHistory[bestCostIndex, :]
    return finalParameters

# -----------------------------------------------------------------------

footModel = NNFootModelSimplest()
numParameters = footModel.getNumParameters()

numInitialStates = 10
numTasks = 10
maxTaskX = 20.
maxTaskY = 15.
maxTaskR = 10.

scale = 200.
populationSize = 50
prevParameters = None
if startFromPreviousParameters:
    prevParameters = getPreviousParameters(prevFilename)
    
initialParameters = generateInitialParametersAround(prevParameters)

batchStatesSize = 5
batchTasksSize = 6

costWeights = CostWeights(failureStepsAfterTermination=10000.,
                            failureSwingFootOutOfBounds=200.,
                            failureAnchoredFootOutOfBounds=200.,
                            failureComUnsupportedAtStart=200.,
                            failureComUnsupportedAtEnd=200.,
                            failureFootOutOfBoundsErrorFromIdeal=5.0,
                            failureComEndErrorFromCentroid=5.0,
                            
                            comNormTranslationErrorInitial = 2.,
                            comNormRotationErrorInitial = 2.,
                            comTranslationSmoothnessInitial= 0.1,
                            comRotationSmoothnessInitial = 0.1,
                            footNormErrorFromIdealInitial = 1.)

numSteps = 4

optimizationParameters = SimpleGAParameters(crossoverRatio=0.5, 
                                            mutationMagnitude=15.0,
                                            decreaseMutationMagnitudeEveryNSteps=50,
                                            mutationMagnitudeLearningRate=0.9,
                                            mutationChance=1.0,
                                            decreaseMutationChanceEveryNSteps=200,
                                            mutationChanceLearningRate=0.9,
                                            mutateWithNormalDistribution=False,
                                            mutationLargeCostScalingFactor=40.0,
                                            diversityChoiceRatio = 0.3,
                                            varianceMutationMaxMagnitude = 10.);  

optimizationEndConditions = OptimizationEndConditions(maxSteps=50000,
                                                      convergenceThreshold=0.0)

printEveryNSteps = 100

# ============================================================================


def main():
    print("Starting from old parameters: " + filename) 
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
    # costEvaluator = BatchSimulation(initialStatesList = initialStatesList, 
    #                                 footModel = footModel, 
    #                                 desiredMotionsList = desiredMotionsList, 
    #                                 numSteps = numSteps, 
    #                                 costWeights = costWeights)
    costEvaluator = StochasticBatchSimulationFromFile(filename = datasetFileName, 
                                                numSteps = numSteps, 
                                                costWeights = costWeights,
                                                batchStatesSize = batchStatesSize, 
                                                batchTasksSize = batchTasksSize)
    optimizer = SimpleGAOptimizer(initialParameters, costEvaluator, optimizationParameters)
    optimizer.printEveryNSteps = printEveryNSteps
    optimizer.setOptimizationEndConditions(optimizationEndConditions)
    
    start_time = time.time()
    try:
        while (not optimizer.hasReachedEndCondition()):
            optimizer.step();
            optimizer.printDebugOnInterval()

    except KeyboardInterrupt:
        pass
    print("Time elapsed: " + str(time.time() - start_time))

    parameterHistory, costHistory = optimizer.getFullHistory();
    bestCostIndex = np.argmin(costHistory)
    finalParameters = parameterHistory[bestCostIndex, :]
    
    print(finalParameters)
    
    simData = {}
    simData["parameterHistory"] = parameterHistory
    simData["costHistory"] = costHistory
    simData["optimizationParameters"] = optimizationParameters
    simData["optimizationPopulationSize"] = populationSize
    simData["optimizationEndConditions"] = optimizationEndConditions
    simData["simNumFootSteps"] = numSteps
    simData["datasetFileName"] = datasetFileName
    simData["finalParameters"] = finalParameters
    simData["footModel"] = footModel
    simData["simCostWeights"] = costWeights
    
    with open(filename, 'wb') as handle:
        pickle.dump(simData, handle)
    
    print("saved")
    
if __name__ == "__main__":
    main()
