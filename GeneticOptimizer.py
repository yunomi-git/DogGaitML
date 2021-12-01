# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 16:10:47 2021

@author: Evan Yu
"""

import numpy as np;
from Optimizer import Optimizer
from dataclasses import dataclass
from abc import ABC, abstractmethod
from scipy.special import softmax
import random


class GeneticAlgorithmOptimizer(Optimizer):
    def __init__(self, initialPopulation, costEvaluator):
        initialValue = initialPopulation[0] # arbitraty: choose first value
        super().__init__(initialValue, costEvaluator);
        self.population = initialPopulation
        self.populationSize = np.ma.size(initialPopulation, 0)
        self.costsList = self.getCostListOfPopulation(self.population)
        
    def takeStepAndGetValue(self):
        self.population = self.getNextPopulation(self.population, self.costsList)
        self.costsList = self.getCostListOfPopulation(self.population)
    
        minCostIndex = self.costsList.index(min(self.costsList))
        value = self.population[minCostIndex, :]

        self.postStepActions()
        return value
    
    @abstractmethod
    def postStepActions(self):
        pass
    
    @abstractmethod
    def getNextPopulation(self, population):
        pass
    
    def getCostListOfPopulation(self, population):
        costsList = []
        for i in range(0, self.populationSize):
            cost = self.costEvaluator.getCost(population[i,:])
            costsList.append(cost)
        return costsList
    
        
@dataclass
class SimpleGAParameters:
    crossoverRatio: float
    mutationChance: float
    mutationMagnitude: float
    decreaseMutationMagnitudeEveryNSteps: int
    mutationMagnitudeLearningRate: float
    decreaseMutationChanceEveryNSteps: int
    mutationChanceLearningRate: float
    mutateWithNormalDistribution: bool
    
    
class SimpleGAOptimizer(GeneticAlgorithmOptimizer):
    def __init__(self, initialPopulation, costEvaluator, GAParameters):
        super().__init__(initialPopulation, costEvaluator);
        self.GAParameters = GAParameters
        
    def getNextPopulation(self, population, costsList):   
        eliteParent = self.value # smallest cost is always saved
        children = np.array([eliteParent])
        
        weightedChoiceList = self.getWeightedChoiceList(population, costsList)
        for i in range(self.populationSize - 1):
            parents, costs = self.choose2Parents(population, costsList, weightedChoiceList)
            parent1 = parents[0]
            parent2 = parents[1]
            child = self.getChildFromParents(parent1, parent2)
            children = np.append(children, np.array([child]), axis=0)
        
        return children
    
    def getWeightedChoiceList(self, population, costsList):
        invertedCosts = -np.array(costsList)
        normedWeights = softmax(invertedCosts)
        return normedWeights
        
    def choose2Parents(self, population, costsList, weightedChoiceList):
        indices = np.random.choice(self.populationSize, 
                                   size=2, 
                                   replace=False, 
                                   p=weightedChoiceList)
        
        parents = [population[indices[0],:], population[indices[1],:]]
        costs = [costsList[indices[0]], costsList[indices[1]]]
        return parents, costs
    
    
    
    def getChildFromParents(self, parent1, parent2):
        if (random.random() < self.GAParameters.crossoverRatio):
            child = self.generateCrossover(parent1, parent2);
        else:
            child = parent1
        
        child = self.generateMutations(child)
        return child
    
    def generateCrossover(self, parent1, parent2):
        crossoverMask = np.random.rand(self.numFeatures) < 0.5
        flippedCrossover = ~crossoverMask
        child = parent1 * crossoverMask + parent2 * flippedCrossover
        return child
    
    def generateMutations(self, child):
        mutationMask = np.random.rand(self.numFeatures) < self.GAParameters.mutationChance
        if (self.GAParameters.mutateWithNormalDistribution):
            mutationValues = mutationMask * (np.random.standard_normal(self.numFeatures)) * self.GAParameters.mutationMagnitude
        else:
            mutationValues = mutationMask * (2 * np.random.rand(self.numFeatures) - 1.0) * self.GAParameters.mutationMagnitude
        child += mutationValues
        return child
    
    def postStepActions(self):
        if ((self.stepCount + 1) % self.GAParameters.decreaseMutationMagnitudeEveryNSteps == 0):
            self.GAParameters.mutationMagnitude *= self.GAParameters.mutationMagnitudeLearningRate
            
        if ((self.stepCount + 1) % self.GAParameters.decreaseMutationChanceEveryNSteps == 0):
            self.GAParameters.mutationChance *= self.GAParameters.mutationChanceLearningRate

