# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 18:48:59 2022

@author: Evan Yu
"""


import numpy as np
from Dynamics import Dynamics
from FootModel import SimpleFootModel
from CostEvaluator import CostEvaluator
from DogUtil import DogModel
import MathUtil as mu
from abc import ABC, abstractmethod
from dataclasses import dataclass
from DebugMessage import DebugMessage
from Simulation import Simulation
from Dataset import Dataset
from GaitPerformanceJudge import GaitPerformanceJudge

class StochasticBatchSimulationFromFile(CostEvaluator):
    def __init__(self, filename, footModel, numSteps, gaitPerformanceJudge: GaitPerformanceJudge, batchStatesSize, batchTasksSize): 
        super().__init__()
        self.footModel = footModel
        self.dataSet = Dataset(filename)
        self.numSteps = numSteps
        self.gaitPerformanceJudge = gaitPerformanceJudge
        
        self.batchStatesSize = batchStatesSize
        self.batchTasksSize = batchTasksSize
                
    def getCost(self, parameters):
        totalCost = 0
        self.debugMessage = DebugMessage()  
        statesBatch, tasksBatch = self.dataset.getRandomBatch(self.batchStatesSize, 
                                                              self.batchTasksSize)
        
        i = 0
        for initialState in statesBatch:
            for desiredMotion in tasksBatch:
                i += 1
                simulation = Simulation(initialState, self.footModel, desiredMotion, self.numSteps, self.gaitPerformanceJudge)
                simulation.setOptimizerIteration(self.optimizerIteration)
                totalCost += simulation.getCost(parameters)
                self.debugMessage.appendMessage("sim" + str(i), simulation.getDebugMessage())
                
        return totalCost

class BatchSimulation(CostEvaluator):
    def __init__(self, initialStatesList, footModel, desiredMotionsList, numSteps, gaitPerformanceJudge: GaitPerformanceJudge): 
        super().__init__()
        self.footModel = footModel
        self.initialStatesList = initialStatesList
        self.desiredMotionsList = desiredMotionsList
        self.numSteps = numSteps
        self.gaitPerformanceJudge = gaitPerformanceJudge
                
    def getCost(self, parameters):
        totalCost = 0
        self.debugMessage = DebugMessage()
        i = 0

        for initialState in self.initialStatesList:
            for desiredMotion in self.desiredMotionsList:
                i += 1
                simulation = Simulation(initialState, self.footModel, desiredMotion, self.numSteps, self.gaitPerformanceJudge)
                simulation.setOptimizerIteration(self.optimizerIteration)
                totalCost += simulation.getCost(parameters)
                self.debugMessage.appendMessage("sim" + str(i), simulation.getDebugMessage())
                
        return totalCost
    
class StochasticBatchSimulation(CostEvaluator):
    def __init__(self, initialStatesList, footModel, desiredMotionsList, numSteps, gaitPerformanceJudge: GaitPerformanceJudge,
                 batchStatesSize, batchTasksSize): 
        super().__init__()
        self.footModel = footModel
        self.initialStatesList = initialStatesList
        self.desiredMotionsList = desiredMotionsList
        self.numSteps = numSteps
        self.gaitPerformanceJudge = gaitPerformanceJudge
        
        self.batchStatesSize = batchStatesSize
        self.batchTasksSize = batchTasksSize
                
    def getCost(self, parameters):
        totalCost = 0
        self.debugMessage = DebugMessage()  
        statesBatch, tasksBatch = self.chooseRandomBatch()
        i = 0


        for initialState in statesBatch:
            for desiredMotion in tasksBatch:
                i += 1
                simulation = Simulation(initialState, self.footModel, desiredMotion, self.numSteps, self.gaitPerformanceJudge)
                simulation.setOptimizerIteration(self.optimizerIteration)
                totalCost += simulation.getCost(parameters)
                self.debugMessage.appendMessage("sim" + str(i), simulation.getDebugMessage())
                
        return totalCost
    
    def chooseRandomBatch(self):
        statesBatch = np.random.choice(self.initialStatesList, 
                                   size=self.batchStatesSize, 
                                   replace=False)
        tasksBatch = np.random.choice(self.desiredMotionsList, 
                                   size=self.batchTasksSize, 
                                   replace=False)
        return statesBatch.tolist(), tasksBatch.tolist()
        