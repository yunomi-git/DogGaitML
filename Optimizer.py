# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 22:11:22 2021

@author: Evan Yu
"""
import numpy as np;
from abc import ABC, abstractmethod
from dataclasses import dataclass
import CostEvaluator


class Optimizer(ABC):
    def __init__(self, initialValue, costEvaluator):
        self.value = initialValue;
        self.costEvaluator = costEvaluator;

        self.stepCount = 0;
        self.valueHistory = np.array([initialValue])
        self.costHistory = np.array([costEvaluator.getCost(initialValue)])
        
        self.numFeatures = initialValue.size
    
    @abstractmethod
    def takeStepAndGetValue(self):
        pass
    
    def step(self):
        self.value = self.takeStepAndGetValue()
        
        self.valueHistory = np.append(self.valueHistory, [self.value], axis=0)
        self.costHistory = np.append(self.costHistory, self.costEvaluator.getCost(self.value))
        self.stepCount += 1;
        
    def getCurrentStateAndCost(self):
        return self.valueHistory[-1], self.costHistory[-1]
    
    def getFullHistory(self):
        return (self.valueHistory, self.costHistory)
        
    def hasReachedMinimum(self, convergenceThreshold):
        if len(self.costHistory) < 2:
            return False;
        currentCost = self.costHistory[-1];
        lastCost = self.costHistory[-2];
        return abs(lastCost - currentCost) < convergenceThreshold;
    
    def optimizeUntilMaxCount(self, maxCount, convergenceThreshold):
        self.stepCount = 0;
        while (~self.hasReachedMinimum(convergenceThreshold) and self.stepCount < maxCount):
            self.step();
            if (self.stepCount % 20 == 0):
                print("step: " + str(self.stepCount))
                value, cost = self.getCurrentStateAndCost()
                print("cost: " + str(cost))

class GradientDescentOptimizer(Optimizer):
    def __init__(self, initialValue, costEvaluator, optimizationParameters):
        super().__init__(initialValue, costEvaluator);
        self.optimizationStepSize = optimizationParameters.optimizationStepSize;
        self.gradientStepFactor = optimizationParameters.gradientStepFactor;
        self.optimizationStepSizeScaling = optimizationParameters.optimizationStepSizeScaling;
        self.scaleEveryNSteps = optimizationParameters.scaleEveryNSteps
        
    def findValueGradient(self):
        numDim = self.value.size;
        valueGradient = np.zeros(numDim);
        currentCost = self.costHistory[-1];
        #construct gradient by sampling in every direction
        for i in range(numDim):
            valueTemp = np.copy(self.value);
            valueTemp[i] += self.gradientStepFactor * self.optimizationStepSize;
            costDim = self.costEvaluator.getCost(valueTemp);
            valueGradient[i] = currentCost - costDim;
        gradientNorm = np.linalg.norm(valueGradient);
        valueGradient /= gradientNorm;
        return valueGradient;
    
    def takeStepAndGetValue(self):
        if ((self.stepCount + 1) % self.scaleEveryNSteps == 0):
            self.optimizationStepSize *= self.optimizationStepSizeScaling
            
        valueGradientDirection = self.findValueGradient()
        valueStepVector = self.optimizationStepSize * valueGradientDirection;
        value = self.value + valueStepVector
        return value

       
    
    
@dataclass
class OptimizationParameters:
    optimizationStepSize : float
    gradientStepFactor : float
    optimizationStepSizeScaling : float
    scaleEveryNSteps : int
    
    
    
