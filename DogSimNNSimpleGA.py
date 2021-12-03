# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 17:43:38 2021

@author: Evan Yu
"""

import numpy as np;
from GeneticOptimizer import SimpleGAOptimizer, SimpleGAParameters
from Optimizer import OptimizationEndConditions
from FootModelNeuralNet import NNFootModelSimplest
from Simulation import BatchSimulation, Simulation
from DogUtil import DogModel, State, TaskMotion
# import matplotlib.pyplot as plt;
# import pyglet
# from pyglet.window import mouse
# from VisualizerSimulation import VisualizerSimulation
import pickle
import os
import time


subFolderName = "GA_NNSimpleModel"
prefix = "12-3-2021_GA_NNSimpleModel"
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

footModel = NNFootModelSimplest()
numParameters = footModel.getNumParameters()

scale = 200.
populationSize = 100

initialParameters = np.random.rand(populationSize, numParameters) * scale - scale/2
initialStatesList = generateInitialStatesList()
desiredMotionsList = generateTaskMotionsList()

costWeights = np.array([1.,1.,
                        1.,1.,
                        20.,
                        300.,
                        100.])
numSteps = 4

optimizationParameters = SimpleGAParameters(crossoverRatio=0.5, 
                                            mutationMagnitude=10.0,
                                            decreaseMutationMagnitudeEveryNSteps=75,
                                            mutationMagnitudeLearningRate=0.8,
                                            mutationChance=0.9,
                                            decreaseMutationChanceEveryNSteps=100,
                                            mutationChanceLearningRate=0.9,
                                            mutateWithNormalDistribution=False,
                                            mutationLargeCostScalingFactor=15.0,
                                            diversityChoiceRatio = 0.5,
                                            varianceMutationMaxMagnitude = 15.);  

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
    
    with open(filename, 'wb') as handle:
        pickle.dump(simData, handle)
    
    print("saved")
    
if __name__ == "__main__":
    main()
# def plotParameterHistory():
#     with open(filename, 'rb') as handle:
#         simData = pickle.load(handle)
#     parameterHistory = simData["parameterHistory"]
#     plotParameters(parameterHistory)

    
# def plotCostHistory():
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1)
#     with open(filename, 'rb') as handle:
#         simData = pickle.load(handle)
#     costHistory = simData["costHistory"]
#     ax.plot(range(0,np.size(costHistory)), costHistory)
#     ax.set_yscale('log')
#     ax.set_ylabel('Cost')
#     ax.set_xlabel('Optimization Step')
#     ax.set_title('Convergence Graph')

#     print(costHistory[-1])
    
# def drawSimulationVisualizer():
#     with open(filename, 'rb') as handle:
#         simData = pickle.load(handle)
#     parameterHistory = simData["parameterHistory"]
#     costHistory = simData["costHistory"]
#     bestCostIndex = np.argmin(costHistory)
#     finalParameters = parameterHistory[bestCostIndex, :]

#     simulation = Simulation(initialState=initialStatesList[0], 
#                             footModel=footModel, 
#                             desiredTaskMotion=desiredMotionsList[0], 
#                             numSteps=numSteps, 
#                             costWeights=costWeights)
    
#     simulation.getCost(finalParameters)
#     window = pyglet.window.Window(960, 540)
#     pyglet.gl.glClearColor(0.6, 0.6, 0.6, 1)
#     batch = pyglet.graphics.Batch()
    
#     visualizer = VisualizerSimulation(simulation.getSimulationHistory())
#     thingsToDraw = visualizer.visualizeCurrentItem(batch)
#     a = [thingsToDraw]
    
#     @window.event
#     def on_draw():
#         window.clear()
#         batch.draw()
        
#     @window.event
#     def on_mouse_press(x,y, button, modifier):
#         if button == mouse.LEFT:
#             for item in a[0]:
#                 item.delete()
#             visualizer.increment()
#             a[0] = visualizer.visualizeCurrentItem(batch)
#             window.clear()
#             batch.draw()
     
#         elif button == mouse.RIGHT:
#             for item in a[0]:
#                 item.delete()
#             visualizer.decrement()
#             a[0] = visualizer.visualizeCurrentItem(batch)
#             window.clear()
#             batch.draw()
    
#     pyglet.app.run() 
    

# def plotParameters(parameterHistory):
#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1)
#     r, c = parameterHistory.shape
#     numItems = r
#     length = c
#     for i in range(0,length):
#         ax.plot(range(0,numItems), parameterHistory[:,i]);
#     ax.set_ylabel('Parameter Value')
#     ax.set_xlabel('Optimization Step')
#     ax.set_title('Parameters')

