# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 16:10:47 2021

@author: Evan Yu
"""

import numpy as np;
from Optimizer import Optimizer
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random


class GeneticAlgorithmOptimizer(Optimizer):
    def __init__(self, initialPopulation, costEvaluator):
        self.population = initialPopulation
        initialValue = initialPopulation[0] # arbitraty: choose first value
        super().__init__(initialValue, costEvaluator);
        self.populationSize = np.ma.size(initialPopulation, 0)
        
    def takeStepAndGetValue(self):
        costsList = self.getCostOfPopulation(self.population)
        nextPopulation = self.getNextPopulation(costsList, self.population)

        minCostIndex = costsList.index(min(costsList))
        value = self.population[minCostIndex, :]
        self.population = nextPopulation
        
        self.postStepActions()
        return value
    
    @abstractmethod
    def postStepActions(self):
        pass
    
    @abstractmethod
    def getNextPopulation(self, costsList, population):
        pass
    
    def getCostOfPopulation(self, population):
        costsList = []
        for value in population:
            cost = self.costEvaluator.getCost(value)
            costsList.append(cost)
        return costsList
        
@dataclass
class SimpleGAParameters:
    crossoverRatio: float
    mutationChance: float
    mutationMagnitude: float
    decreaseMutationEveryNSteps: int
    mutationLearningRation: float
    
class SimpleGAOptimizer(GeneticAlgorithmOptimizer):
    def __init__(self, initialPopulation, costEvaluator, simpleGAParameters):
        super().__init__(initialPopulation, costEvaluator);
        self.simpleGAParameters = simpleGAParameters
        
    def getNextPopulation(self, costsList, population):   
        eliteParent = self.value
        children = np.array([eliteParent])
        
        for i in range(self.populationSize - 1):
            parent1, parent2 = self.choose2Parents(population, costsList)
            child = self.getChildFromParents(parent1, parent2)
            children = np.append(children, np.array([child]), axis=0)
            
        return children
        
    def choose2Parents(self, population, costsList):
        normedCosts = costsList/sum(costsList)
        weights = 1.0/normedCosts
        normedWeights = weights/sum(weights)
        indices = np.random.choice(self.populationSize, size=2, replace=False, p=normedWeights)
        return population[indices[0],:], population[indices[1],:]
    
    def getChildFromParents(self, parent1, parent2):
        if (random.random() < self.simpleGAParameters.crossoverRatio):
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
        mutationMask = np.random.rand(self.numFeatures) < self.simpleGAParameters.mutationChance
        mutationValues = mutationMask * (np.random.standard_normal(self.numFeatures)) * self.simpleGAParameters.mutationMagnitude
        child += mutationValues
        return child
    
    def postStepActions(self):
        if ((self.stepCount + 1) % self.simpleGAParameters.decreaseMutationEveryNSteps == 0):
            self.simpleGAParameters.mutationMagnitude *= self.simpleGAParameters.mutationLearningRation
