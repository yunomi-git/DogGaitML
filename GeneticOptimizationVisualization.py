# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 19:31:07 2021

@author: Evan Yu
"""

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from GeneticVisualizer import GeneticVisualizer
from GeneticOptimizer import SimpleGAOptimizer, SimpleGAParameters
from Optimizer import OptimizationEndConditions
import numpy as np
from CostFunction2D import *


# dataHistory is a list of np arrays with shape (numElements, 3)
def createPopulationHistory(costEvaluator):
    numData = 10
    numHistory = 100
    xMax, yMax = costEvaluator.getDefaultRange()
    
    dataHistory = []

    ## Full 
    # Ackley
    # optimizationParameters = SimpleGAParameters(crossoverRatio=0.5, 
    #                                             mutationMagnitude=3.0,
    #                                             decreaseMutationMagnitudeEveryNSteps=10,
    #                                             mutationMagnitudeLearningRate=0.9,
    #                                             mutationChance=0.9,
    #                                             decreaseMutationChanceEveryNSteps=10,
    #                                             mutationChanceLearningRate=0.9,
    #                                             mutateWithNormalDistribution=False,

    #                                             mutationLargeCostScalingFactor=3.0,
    #                                             diversityChoiceRatio = 0.7,
    #                                             varianceMutationMaxMagnitude = 3.0)

    # Eggholder
    optimizationParameters = SimpleGAParameters(crossoverRatio=0.5, 
                                                mutationMagnitude=3.0,
                                                decreaseMutationMagnitudeEveryNSteps=10,
                                                mutationMagnitudeLearningRate=0.9,
                                                mutationChance=0.9,
                                                decreaseMutationChanceEveryNSteps=10,
                                                mutationChanceLearningRate=0.9,
                                                mutateWithNormalDistribution=False,
                                                mutationLargeCostScalingFactor=40.0,
                                                diversityChoiceRatio = 0.7,
                                                varianceMutationMaxMagnitude = 50.)

    ## Nothing
    # optimizationParameters = SimpleGAParameters(crossoverRatio=0.5, 
    #                                             mutationMagnitude=3.0,
    #                                             decreaseMutationMagnitudeEveryNSteps=10,
    #                                             mutationMagnitudeLearningRate=0.9,
    #                                             mutationChance=0.9,
    #                                             decreaseMutationChanceEveryNSteps=10,
    #                                             mutationChanceLearningRate=0.9,
    #                                             mutateWithNormalDistribution=False,

    #                                             mutationLargeCostScalingFactor=0.0,
    #                                             diversityChoiceRatio = 0.0,
    #                                             varianceMutationMaxMagnitude = 0.0)

    # ## Diversity Only
    # optimizationParameters = SimpleGAParameters(crossoverRatio=0.5, 
    #                                             mutationMagnitude=3.0,
    #                                             decreaseMutationMagnitudeEveryNSteps=10,
    #                                             mutationMagnitudeLearningRate=0.9,
    #                                             mutationChance=0.9,
    #                                             decreaseMutationChanceEveryNSteps=10,
    #                                             mutationChanceLearningRate=0.9,
    #                                             mutateWithNormalDistribution=False,
    #                                             mutationLargeCostScalingFactor=0.0,
    #                                             diversityChoiceRatio = 0.7,
    #                                             varianceMutationMaxMagnitude = 0.)

    # ## Large mutation scale Only
    # optimizationParameters = SimpleGAParameters(crossoverRatio=0.5, 
    #                                             mutationMagnitude=3.0,
    #                                             decreaseMutationMagnitudeEveryNSteps=10,
    #                                             mutationMagnitudeLearningRate=0.9,
    #                                             mutationChance=0.9,
    #                                             decreaseMutationChanceEveryNSteps=10,
    #                                             mutationChanceLearningRate=0.9,
    #                                             mutateWithNormalDistribution=False,
    #                                             mutationLargeCostScalingFactor=3.0,
    #                                             diversityChoiceRatio = 0.0,
    #                                             varianceMutationMaxMagnitude = 0.)

    # ## Population Variance Only
    # optimizationParameters = SimpleGAParameters(crossoverRatio=0.5, 
    #                                             mutationMagnitude=3.0,
    #                                             decreaseMutationMagnitudeEveryNSteps=10,
    #                                             mutationMagnitudeLearningRate=0.9,
    #                                             mutationChance=0.9,
    #                                             decreaseMutationChanceEveryNSteps=10,
    #                                             mutationChanceLearningRate=0.9,
    #                                             mutateWithNormalDistribution=False,
    #                                             mutationLargeCostScalingFactor=0.0,
    #                                             diversityChoiceRatio = 0.0,
    #                                             varianceMutationMaxMagnitude = 3.)

    initialValue = (np.random.rand(numData, 2)-0.5) * (xMax + yMax)
    optimizer = SimpleGAOptimizer(initialValue, costEvaluator, optimizationParameters,
                               minBounds = [-xMax, -yMax],
                               maxBounds = [xMax, yMax])
    endConditions = OptimizationEndConditions(maxSteps=numHistory, 
                                              convergenceThreshold=0.00)
    optimizer.setOptimizationEndConditions(endConditions)
    optimizer.stepCount = 0
    appendOptimizationDataToHistory(optimizer, dataHistory)
    while (not optimizer.hasReachedEndCondition()):
        optimizer.step()
        appendOptimizationDataToHistory(optimizer, dataHistory)
    # optimizer.optimizeUntilEndCondition(endConditions);
    temp, convergenceHistory = optimizer.getFullHistory()
    return dataHistory, convergenceHistory

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

# costEvaluator = ParaboloidCostEvaluator(a=0.01, b=0.01, c=0, d=0, e=0);
# costEvaluator = SmithCostEvaluator(0.05, 10)
# costEvaluator = SixHumpCamelCostEvaluator()
costEvaluator = AckleyCostEvaluator()
# costEvaluator = EggHolderCostEvaluator()
xMax, yMax = costEvaluator.getDefaultRange()
dataHistory, convergenceHistory = createPopulationHistory(costEvaluator)
    
visualizer = GeneticVisualizer(costEvaluator=costEvaluator,
                               dataHistory=dataHistory,
                               convergenceHistory=convergenceHistory)
visualizer.setPlotRange(xMax = xMax, yMax = yMax)
visualizer.setResolution(resolution=200)
visualizer.visualize()


if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

    
    

