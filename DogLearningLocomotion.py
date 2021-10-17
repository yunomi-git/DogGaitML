# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 21:12:37 2021

@author: Evan Yu
"""
import numpy as np;
from Optimizer import Optimizer, OptimizationParameters, GradientDescentOptimizer
from FootModel import SimpleFootModel
from Simulation import BatchSimulation, Simulation
from DogUtil import DogModel, State, TaskMotion
import matplotlib.pyplot as plt;
import pyglet
from pyglet.window import mouse
from VisualizerSimulation import VisualizerSimulation
import pickle

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

scale = 1.
initialParameters = np.random.rand(numParameters) * scale - scale/2
initialStatesList = generateInitialStatesList()
desiredMotionsList = generateTaskMotionsList()

costWeights = np.array([1.,1.,
                        1.,1.,
                        1.,
                        300.,
                        100.])
numSteps = 10
simulationName = "n3"
readSimulationName = "n3"



def runOptimizer():
    costEvaluator = BatchSimulation(initialStatesList = initialStatesList, 
                                    footModel = footModel, 
                                    desiredMotionsList = desiredMotionsList, 
                                    numSteps = numSteps, 
                                    costWeights = costWeights)
        
    optimizationParameters = OptimizationParameters(optimizationStepSize = 0.1,
                                                    gradientStepSize = 0.001,
                                                    convergenceThreshold = 0.001,
                                                    optimizationStepSizeScaling = 0.95,
                                                    scaleEveryNSteps = 50)                       
    optimizer = GradientDescentOptimizer(initialParameters, costEvaluator, optimizationParameters)
    optimizer.optimizeUntilMaxCount(10000);
    parameterHistory, costHistory = optimizer.getFullHistory();
    finalParameters = parameterHistory[-1,:]
    np.savetxt(".\\data\\" +simulationName + '_parameterHistory.dat', parameterHistory)
    np.savetxt(".\\data\\"+simulationName + '_costHistory.dat', costHistory)
    np.savetxt(".\\data\\"+simulationName + '_parameters.dat', finalParameters)

def main():
    # runOptimizer()
    plotParameterHistory()
    plotCostHistory()
    # drawSimulationVisualizer()
    
    
def plotParameterHistory():
    parameterHistory = np.loadtxt(".\\data\\" +readSimulationName + '_parameterHistory.dat')
    plotParameters(parameterHistory)
    print(parameterHistory[0,:])
    print(parameterHistory[0,:] - parameterHistory[-1,:])
    
def plotCostHistory():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    costHistory = np.loadtxt(".\\data\\" +readSimulationName + '_costHistory.dat')
    ax.plot(range(0,np.size(costHistory)), costHistory)
    ax.set_yscale('log')

    print(costHistory[-1])
    
def drawSimulationVisualizer():
    finalParameters = np.loadtxt(".\\data\\" +readSimulationName + '_parameters.dat')
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
    r, c = parameterHistory.shape
    numItems = r
    length = c
    for i in range(0,length):
        plt.plot(range(0,numItems), parameterHistory[:,i]);

if __name__ == "__main__":
    main()