# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 13:16:47 2021

@author: Evan Yu
"""

import numpy as np;
from GeneticOptimizer import SimpleGAOptimizer, SimpleGAParameters
from Optimizer import OptimizationEndConditions
from FootModelNeuralNet import NNFootModelSimplest
from Simulation import BatchSimulation, Simulation
from DogUtil import DogModel, State, TaskMotion
import matplotlib.pyplot as plt;
import pyglet
from pyglet.window import mouse
from VisualizerSimulation import VisualizerSimulation
import pickle
import os
from FootModel import SimpleFootModel
import time



subFolderName = "GA_NNSimpleModel"
prefix = "01-01-2022_GA_NNSimpleModel_NewCostWeights"
suffix = "_04"

oldVisualization= False
oldSimulationName = "GA1"
oldSubFolderName = "LinearModel_GDandGA"
oldCostWeights = np.array([1.,1.,
                            1.,1.,
                            20.,
                            300.,
                            100.])
oldFootModel = SimpleFootModel()

numSteps = 4


def generateInitialState():
    dogModel = DogModel()
    initialCOM = np.array([-20.0, 0.01])
    initialFootState = dogModel.defaultFootState - initialCOM
    initialState = State(initialFootState, 0.)
    return initialState

def generateTaskMotion():
    return TaskMotion(5., 0.1, 0.1)

# -----------------------------------------------------------------------

simulationName = prefix + suffix
path = ".\\data\\" + subFolderName + "\\"
filename =  path + simulationName + '.pickle'

def main():
    plotParameterHistory()
    plotCostHistory()
    drawSimulationVisualizer()

def plotParameterHistory():
    if oldVisualization:
        parameterHistory = np.loadtxt(".\\data\\LinearModel_GDandGA\\" 
                                      + oldSimulationName 
                                      + '_parameterHistory.dat')
    else:
        with open(filename, 'rb') as handle:
            simData = pickle.load(handle)
        parameterHistory = simData["parameterHistory"]
    plotParameters(parameterHistory)
    
def plotCostHistory():
    if oldVisualization:
        costHistory = np.loadtxt(".\\data\\LinearModel_GDandGA\\" 
                                    + oldSimulationName 
                                    + '_costHistory.dat')
    else:
        with open(filename, 'rb') as handle:
            simData = pickle.load(handle)
        costHistory = simData["costHistory"]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(0,np.size(costHistory)), costHistory)
    ax.set_yscale('log')
    ax.set_ylabel('Cost')
    ax.set_xlabel('Optimization Step')
    ax.set_title('Convergence Graph')

    print(costHistory[-1])
    
def drawSimulationVisualizer():
    if oldVisualization:
        parameterHistory = np.loadtxt(".\\data\\LinearModel_GDandGA\\" 
                                      + oldSimulationName 
                                      + '_parameterHistory.dat')
        costHistory = np.loadtxt(".\\data\\LinearModel_GDandGA\\" 
                                  + oldSimulationName 
                                  + '_costHistory.dat')
        costWeights = oldCostWeights
        footModel = oldFootModel
    else:
        with open(filename, 'rb') as handle:
            simData = pickle.load(handle)
        parameterHistory = simData["parameterHistory"]
        costHistory = simData["costHistory"]    
        costWeights = simData["simCostWeights"]
        footModel = simData["footModel"]
        
    bestCostIndex = np.argmin(costHistory)
    print(costHistory[bestCostIndex])
    finalParameters = parameterHistory[bestCostIndex, :]
    
    initialState = generateInitialState()
    desiredMotion = generateTaskMotion()
    
    simulation = Simulation(initialState=initialState, 
                            footModel=footModel, 
                            desiredTaskMotion=desiredMotion, 
                            numSteps=numSteps, 
                            costWeights=costWeights)
    print(desiredMotion)
    print(initialState)
    cost = simulation.getCost(finalParameters)
    print(cost)
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