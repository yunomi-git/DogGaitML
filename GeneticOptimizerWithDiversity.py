# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 14:04:09 2021

@author: Evan Yu
"""

from GeneticOptimizer import GeneticAlgorithmOptimizer, SimpleGAOptimizer
import numpy as np
from scipy.special import softmax


class SimpleGAOptimizerWithDiversity(SimpleGAOptimizer):
    def __init__(self, initialPopulation, costEvaluator, GAParameters):
        super().__init__(initialPopulation, costEvaluator, GAParameters)
        
    def getWeightedChoiceList(self, population):
        invertedCosts = -np.array(self.costsList) # lower cost = higher chance
        # should account for negative costs, but not empirically supported
        normedCostWeights = softmax(invertedCosts) 
        
        diversityList = self.getDiversityListOfPopulation(population)
        normedDiversity  = diversityList / np.linalg.norm(diversityList)
        
        weights = (normedCostWeights + normedDiversity) / 2.0
        
        return weights
    
    def getVarianceOfPopulation(population):
        covMat = np.cov(population)
        det = np.linalg.det(covMat)
        return det
    
    def getDiversityListOfPopulation(population):
        populationSize, numDim = population.shape
        data = population.reshape((populationSize,1,numDim))
        comparator = population.reshape((1,populationSize,numDim))
        
        axisData = 0
        axisComparator = 1
        axisDim = 2
        
        dataTens = np.repeat(data, populationSize, axisComparator)
        compTens = np.repeat(comparator, populationSize, axisData)
        errorTens = dataTens - compTens # shape (data, comparator, dim)
        
        errorNormMat = np.linalg.norm(errorTens, axis=axisDim) #norm squishes along dimension axis
        errorAvgList = np.sum(errorNormMat, axis=axisData) / (populationSize - 1) # combines along data axis
        return errorAvgList



# @dataclass
# class GAParametersWithDiversity(SimpleGAParameters):
    
    