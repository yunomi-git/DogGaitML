# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 19:25:55 2021

@author: Evan Yu
"""

import numpy as np;
from GeneticOptimizer import SimpleGAOptimizer, SimpleGAParameters
from FootModel import SimpleFootModel
from Simulation import BatchSimulation, Simulation
from DogUtil import DogModel, State, TaskMotion
import matplotlib.pyplot as plt;
import pyglet
from pyglet.window import mouse
from VisualizerSimulation import VisualizerSimulation
import pickle
from pynput import keyboard


def generateInitialStatesList():
    dogModel = DogModel()
    initialCOM = np.array([-20.0, 0])
    initialFootState = dogModel.defaultFootState - initialCOM
    initialState = State(initialFootState, 0.)
    return [initialState]

def generateTaskMotionsList():
    return [TaskMotion(5, 5, 3)]


#TODO find a way to save these to file
footModel = SimpleFootModel()
numParameters = footModel.getNumParameters()

scale = 5.
populationSize = 100
initialParameters = np.random.rand(populationSize, numParameters) * scale - scale/2
initialStatesList = generateInitialStatesList()
desiredMotionsList = generateTaskMotionsList()

costWeights = np.array([1.,1.,
                        1.,1.,
                        20.,
                        300.,
                        100.])
numSteps = 5
simulationName = "GA3"
readSimulationName = "GA3"

endOptimizerEarly = False



def runOptimizer():    
    costEvaluator = BatchSimulation(initialStatesList = initialStatesList, 
                                    footModel = footModel, 
                                    desiredMotionsList = desiredMotionsList, 
                                    numSteps = numSteps, 
                                    costWeights = costWeights)
        
    optimizationParameters = SimpleGAParameters(crossoverRatio=0.7, 
                                                mutationChance=0.9, 
                                                mutationMagnitude=10,
                                                decreaseMutationMagnitudeEveryNSteps=100,
                                                mutationMagnitudeLearningRate=0.7,
                                                decreaseMutationChanceEveryNSteps=100,
                                                mutationChanceLearningRate=0.7);                    
    optimizer = SimpleGAOptimizer(initialParameters, costEvaluator, optimizationParameters)
    optimizer.optimizeUntilMaxCount(100000, 0);
    parameterHistory, costHistory = optimizer.getFullHistory();
    finalParameters = parameterHistory[-1,:]
    np.savetxt(".\\data\\" +simulationName + '_parameterHistory.dat', parameterHistory)
    np.savetxt(".\\data\\"+simulationName + '_costHistory.dat', costHistory)
    np.savetxt(".\\data\\"+simulationName + '_parameters.dat', finalParameters)

def main():
    runOptimizer()
    # plotParameterHistory()
    # plotCostHistory()
    # drawSimulationVisualizer()
    

def plotParameterHistory():
    parameterHistory = np.loadtxt(".\\data\\" +readSimulationName + '_parameterHistory.dat')
    plotParameters(parameterHistory)
    print(footModel.convertParametersToModel(parameterHistory[0,:] - parameterHistory[-1,:]))

    
def plotCostHistory():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    costHistory = np.loadtxt(".\\data\\" +readSimulationName + '_costHistory.dat')
    ax.plot(range(0,np.size(costHistory)), costHistory)
    ax.set_yscale('log')
    ax.set_ylabel('Cost')
    ax.set_xlabel('Optimization Step')
    ax.set_title('Convergence Graph')

    print(costHistory[-1])
    
def drawSimulationVisualizer():
    costHistory = np.loadtxt(".\\data\\" +readSimulationName + '_costHistory.dat')
    parameterHistory = np.loadtxt(".\\data\\" +readSimulationName + '_parameterHistory.dat')
    bestCostIndex = np.argmin(costHistory)
    finalParameters = parameterHistory[bestCostIndex, :]
    # finalParameters = np.loadtxt(".\\data\\" +readSimulationName + '_parameters.dat')
    simulation = Simulation(initialState=initialStatesList[0], 
                            footModel=footModel, 
                            desiredTaskMotion=desiredMotionsList[0], 
                            numSteps=numSteps, 
                            costWeights=costWeights)
    simulation.getCost(finalParameters)
    window = pyglet.window.Window(960, 540)
    pyglet.gl.glClearColor(0.6, 0.6, 0.6, 1)
    batch = pyglet.graphics.Batch()
    
    visualizer = VisualizerSimulation(simulation.getSimulationHistory())
    thingsToDraw = visualizer.visualizeCurrentItem(batch)
    a = [thingsToDraw]
    
    @window.event
    def on_draw():
        window.clear()
        batch.draw()
        
    @window.event
    def on_mouse_press(x,y, button, modifier):
        if button == mouse.LEFT:
            for item in a[0]:
                item.delete()
            visualizer.increment()
            a[0] = visualizer.visualizeCurrentItem(batch)
            window.clear()
            batch.draw()
     
        elif button == mouse.RIGHT:
            for item in a[0]:
                item.delete()
            visualizer.decrement()
            a[0] = visualizer.visualizeCurrentItem(batch)
            window.clear()
            batch.draw()
    
    pyglet.app.run() 
    
def generateInitialStatesList():
    dogModel = DogModel()
    initialCOM = np.array([-20.0, 0])
    initialFootState = dogModel.defaultFootState - initialCOM
    initialState = State(initialFootState, 0.)
    return [initialState]

def plotParameters(parameterHistory):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    r, c = parameterHistory.shape
    numItems = r
    length = c
    for i in range(0,length):
        ax.plot(range(0,numItems), parameterHistory[:,i]);
    ax.set_ylabel('Parameter Value')
    ax.set_xlabel('Optimization Step')
    ax.set_title('Parameters')

if __name__ == "__main__":
    main()