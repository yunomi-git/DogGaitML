# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 18:17:59 2021

@author: Evan Yu
"""
import matplotlib.pyplot as mpl;
from CostEvaluator import CostEvaluator
from mpl_toolkits.mplot3d import Axes3D
from GeneticOptimizer import SimpleGAOptimizer, SimpleGAParameters
from Optimizer import OptimizationEndConditions
import numpy as np
import matplotlib.pyplot as plt;


class ParabolicCostEvaluator(CostEvaluator):
    def __init__(self, a, b):
        self.a = a;
        self.b = b
        
    def getCost(self, value):
        x = value[0];
        return (self.a * x * x + self.b * x)
    
class ParaboloidCostEvaluator(CostEvaluator):
    def __init__(self, a, b):
        self.a = a;
        self.b = b
        
    def getCost(self, value):
        x = value[0];
        y = value[1];
        return (self.a * x * x + self.b * y * y)

def optimizeParabola() :
    optimizationParameters = SimpleGAParameters(0.3, 0.3, 1);
    initialValue = np.random.rand(10, 1) * 5;
    costEvaluator = ParabolicCostEvaluator(1.0, 0);
    optimizer = SimpleGAOptimizer(initialValue, costEvaluator, optimizationParameters);
    optimizer.optimizeUntilMaxCount(1000, 0.0);
    valueHistory, costHistory = optimizer.getFullHistory();
    
    count = len(costHistory);
    xval = [];
    yval = [];
    for i in range(count):
        x = valueHistory[i];
        
        xval.append(x[0])
        yval.append(costHistory[i])
    
    mpl.plot(xval, yval);
    print(xval[-1])
    
def optimizeParabaloid():
    a = 1.0
    b = 1.0
    optimizationParameters = SimpleGAParameters(crossoverRatio=0.5, 
                                                mutationChance=0.9, 
                                                mutationMagnitude=3.0,
                                                decreaseMutationMagnitudeEveryNSteps=10,
                                                mutationMagnitudeLearningRate=0.7,
                                                decreaseMutationChanceEveryNSteps=10,
                                                mutationChanceLearningRate=0.7);
    initialValue = np.random.rand(10, 2) * 5;
    costEvaluator = ParaboloidCostEvaluator(a, b);
    optimizer = SimpleGAOptimizer(initialValue, costEvaluator, optimizationParameters);
    endConditions = OptimizationEndConditions(100, 0.00)
    optimizer.optimizeUntilEndCondition(endConditions);
    valueHistory, costHistory = optimizer.getFullHistory();
    
    count = len(costHistory);
    xval = valueHistory[:,0];
    yval = valueHistory[:,1];
    zval = costHistory;
    
    fig = mpl.figure()
    ax = fig.add_subplot(111, projection='3d')
    plotParaboloic(ax, a, b)
    ax.plot(xval, yval, zval);
    print(zval[-1]) 
    print(xval[-1])
    print(yval[-1])
    
    plotCostHistory(costHistory)
    
def plotParaboloic(ax, a, b):
    x = np.linspace(0,2,20);
    y = np.linspace(0,2,20);
    x, y = np.meshgrid(x, y);
    z = a * x * x + b * y * y;
    ax.plot_surface(x, y, z, alpha=0.4);
    
def main():
    optimizeParabaloid()
    

def plotCostHistory(costHistory):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(0,np.size(costHistory)), costHistory)
    ax.set_yscale('log')
    ax.set_ylabel('Cost')
    ax.set_xlabel('Optimization Step')
    ax.set_title('Convergence Graph')

    print(costHistory[-1])
    
    
if __name__ == "__main__":
    main()