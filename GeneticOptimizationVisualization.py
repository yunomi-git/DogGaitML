# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 19:31:07 2021

@author: Evan Yu
"""

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from GeneticVisualizer import GeneticVisualizer, CostEvaluator2D, ParaboloidCostEvaluator, SmithCostEvaluator
from GeneticOptimizer import SimpleGAOptimizer, SimpleGAParameters
from Optimizer import OptimizationEndConditions
import numpy as np


# dataHistory is a list of np arrays with shape (numElements, 3)
def createPopulationHistory(costEvaluator):
    numData = 10
    numHistory = 100
    
    dataHistory = []
        
    optimizationParameters = SimpleGAParameters(crossoverRatio=0.5, 
                                                mutationChance=0.9, 
                                                mutationMagnitude=3.0,
                                                decreaseMutationMagnitudeEveryNSteps=10,
                                                mutationMagnitudeLearningRate=0.7,
                                                decreaseMutationChanceEveryNSteps=10,
                                                mutationChanceLearningRate=0.7);
    initialValue = (np.random.rand(numData, 2)-0.5) * 30;
    optimizer = SimpleGAOptimizer(initialValue, costEvaluator, optimizationParameters);
    endConditions = OptimizationEndConditions(maxSteps=numHistory, 
                                              convergenceThreshold=0.00)
    optimizer.setOptimizationEndConditions(endConditions)
    optimizer.stepCount = 0;
    appendOptimizationDataToHistory(optimizer, dataHistory)
    while (not optimizer.hasReachedEndCondition()):
        optimizer.step();
        appendOptimizationDataToHistory(optimizer, dataHistory)
    # optimizer.optimizeUntilEndCondition(endConditions);
    return dataHistory;

def appendOptimizationDataToHistory(optimizer, dataHistory):
    population = np.array(optimizer.population)
    costsList = np.array(optimizer.costsList)
    numPopulation = costsList.size
    data = np.empty((numPopulation, 3))
    data[:,0] = population[:,0]
    data[:,1] = population[:,1]
    data[:,2] = costsList
    dataHistory.append(data)


# ## Create a GL View widget to display data
app = QtGui.QApplication([])

xMax = 15
yMax = 15
# costEvaluator = ParaboloidCostEvaluator(a=0.01, b=0.01, c=0, d=0, e=0);
costEvaluator = SmithCostEvaluator(0.05, 10)
dataHistory = createPopulationHistory(costEvaluator)
    
visualizer = GeneticVisualizer(costEvaluator=costEvaluator,
                               dataHistory=dataHistory)
visualizer.setPlotRange(xMax = 15, yMax = 15)
visualizer.setResolution(resolution=200)
visualizer.visualize()


if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

    
    
