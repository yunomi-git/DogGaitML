# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 22:11:22 2021

@author: Evan Yu
"""
import numpy as np;
from abc import ABC, abstractmethod
from dataclasses import dataclass
import matplotlib.pyplot as mpl;
from CostEvaluator import ParabolicCostEvaluator, ParaboloidCostEvaluator
from mpl_toolkits.mplot3d import Axes3D


class Optimizer(ABC):
    # value: np array
    # costEvaluator: class
    # optimizationParameters:
    #       optimizationStepSize
    #       gradientStepSize
    #       convergenceThreshold
    #       optimizationStepSizeScaling
    #       scaleEveryNSteps
    def __init__(self, initialValue, costEvaluator, optimizationParameters):
        self.value = initialValue;
        self.costEvaluator = costEvaluator;
        self.optimizationStepSize = optimizationParameters.optimizationStepSize;
        self.gradientStepSize = optimizationParameters.gradientStepSize;
        self.convergenceThreshold = optimizationParameters.convergenceThreshold;
        self.optimizationStepSizeScaling = optimizationParameters.optimizationStepSizeScaling;
        self.scaleEveryNSteps = optimizationParameters.scaleEveryNSteps
        self.stepCount = 0;
        self.history = [];
        self.history.append([initialValue, costEvaluator.getCost(initialValue)]);
    
    def takeOptimizationStep(self):
        valueGradientDirection = self.findValueGradient()
        valueStepVector = self.optimizationStepSize * valueGradientDirection;
        self.value = self.value + valueStepVector
        self.history.append([self.value, self.costEvaluator.getCost(self.value)]);
        self.stepCount += 1;
        if (self.stepCount % self.scaleEveryNSteps == 0):
            self.optimizationStepSize *= self.optimizationStepSizeScaling
        
    def getCurrentStateAndCost(self):
        return self.history[-1]
    
    def getFullHistory(self):
        return self.history;
        
    @abstractmethod
    def findValueGradient(self):
        pass;
        
    def hasReachedMinimum(self):
        if len(self.history) < 2:
            return False;
        currentHistory = self.history[-1];
        lastHistory = self.history[-2];
        currentCost = currentHistory[1];
        lastCost = lastHistory[1];
        return abs(lastCost - currentCost) < self.convergenceThreshold;
    
    def optimizeUntilMaxCount(self, maxCount):
        currentCount = 0;
        while (~self.hasReachedMinimum() and currentCount < maxCount):
            self.takeOptimizationStep();
            currentCount += 1;

class EvolutionaryOptimizer(Optimizer):
    def __init__(self, initialValue, costEvaluator, optimizationParameters):
        super().__init__(initialValue, costEvaluator, optimizationParameters);
        
    def findValueGradient(self):
        numDim = self.value.size;
        valueGradient = np.zeros(numDim);
        currentHistory = self.history[-1]
        currentCost = currentHistory[1];
        #construct gradient by sampling in every direction
        for i in range(numDim):
            valueTemp = np.copy(self.value);
            valueTemp[i] += self.gradientStepSize;
            costDim = self.costEvaluator.getCost(valueTemp);
            valueGradient[i] = currentCost - costDim;
        gradientNorm = np.linalg.norm(valueGradient);
        valueGradient /= gradientNorm;
        return valueGradient;
    
    
@dataclass
class OptimizationParameters:
    optimizationStepSize : float
    gradientStepSize : float
    convergenceThreshold : float
    optimizationStepSizeScaling : float
    scaleEveryNSteps : int
    
    
    
    
    
    
def optimizeParabola() :
    optimizationParameters = OptimizationParameters(0.5, 0.5, 0.00, 0.9, 3);
    initialValue = np.array([5.0]);
    costEvaluator = ParabolicCostEvaluator(1.0, 0);
    optimizer = EvolutionaryOptimizer(initialValue, costEvaluator, optimizationParameters);
    optimizer.optimizeUntilMaxCount(100);
    history = optimizer.getFullHistory();
    
    count = len(history);
    xval = [];
    yval = [];
    for i in range(count):
        x = history[i][0];
        xval.append(x[0])
        yval.append(history[i][1])
    
    mpl.plot(xval, yval);
    print(xval[-1])
    
def optimizeParabaloid():
    a = 1.0
    b = 1.0
    optimizationParameters = OptimizationParameters(0.1, 0.001, 0.00, 0.95, 10);
    initialValue = np.array([5.0, 5.0]);
    costEvaluator = ParaboloidCostEvaluator(a, b);
    optimizer = EvolutionaryOptimizer(initialValue, costEvaluator, optimizationParameters);
    optimizer.optimizeUntilMaxCount(1000);
    history = optimizer.getFullHistory();
    
    count = len(history);
    xval = [];
    yval = [];
    zval = [];
    for i in range(count):
        x = history[i][0];
        xval.append(x[0])
        yval.append(x[1])
        zval.append(history[i][1])
    
    fig = mpl.figure()
    ax = fig.add_subplot(111, projection='3d')
    plotParaboloic(ax, a, b)
    ax.plot(xval, yval, zval);
    # print(zval) 
    print(xval[-1])
    print(yval[-1])
    
def plotParaboloic(ax, a, b):
    x = np.linspace(0,5,20);
    y = np.linspace(0,5,20);
    x, y = np.meshgrid(x, y);
    z = a * x * x + b * y * y;
    ax.plot_surface(x, y, z, alpha=0.4);
    
def main():
    optimizeParabaloid()
    
    
if __name__ == "__main__":
    main()