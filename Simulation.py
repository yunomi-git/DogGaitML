# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 19:22:23 2021

@author: Evan Yu
"""

import numpy as np
from Dynamics import Dynamics
from FootModel import FootModel
from CostEvaluator import CostEvaluator
from DogUtil import DogModel, State, TaskMotion
import MathUtil as mu
from abc import ABC, abstractmethod
from dataclasses import dataclass
from DebugMessage import DebugMessage
from GaitPerformanceJudge import GaitPerformanceJudge
from SimulationHistoryItem import SimulationHistoryItem

    
class Simulation(CostEvaluator):
    def __init__(self, initialState: State, footModel: FootModel, desiredTaskMotion: TaskMotion, 
                 numSteps, gaitPerformanceJudge: GaitPerformanceJudge):
        super().__init__()
        self.simulationHistory = [];

        self.currentCOMInWorldFrame = np.array([0.,0.]);
        self.currentState = initialState;
        self.lastCommand = None
        
        self.footModel = footModel
        self.numSteps = numSteps
        
        self.desiredTaskMotion = desiredTaskMotion
        
        self.terminated = False
        self.currentRunningCost = 0
        
        self.simulationHistory.append(SimulationHistoryItem(self.currentState, 
                                              self.currentCOMInWorldFrame, 
                                              self.currentRunningCost, 
                                              None,
                                              None))
        self.numSuccessfulStepsTaken = 0
        self.gaitPerformanceJudge = gaitPerformanceJudge
            
    def takeSimulationStep(self):
        desiredCommand = self.lastCommand
        if (not self.hasTerminated()):
            dynamics = Dynamics(self.currentState)
            desiredCommand = self.footModel.computeCommandFromState(self.currentState, self.desiredTaskMotion);
            dynamics.applyCommand(desiredCommand)
            originalState = self.currentState
            
            currentState = dynamics.getCurrentState()
            currentCOMInWorldFrame = self.currentCOMInWorldFrame + desiredCommand.comTranslation
            
            if not dynamics.hasFailed():
                self.currentState = currentState
                self.currentCOMInWorldFrame = currentCOMInWorldFrame
                self.numSuccessfulStepsTaken += 1
                self.gaitPerformanceJudge.applyPostStepCostActions()
            else:
                self.terminate()

            # cost = self.gaitPerformanceJudge.computeCostBeforeTermination(desiredTaskMotion=self.desiredTaskMotion, 
            #                     command=desiredCommand, 
            #                     originalState=originalState, 
            #                     currentState=currentState, 
            #                     failureMessage=dynamics.getFailureMessage()); 
            
            self.lastCommand = desiredCommand
            failureMessage = dynamics.getFailureMessage()
        else:
            # cost = self.computeCostAfterTermination()
            failureMessage = None #TODO should change this to "last error message"
        
        
            
        simulationHistoryItem = SimulationHistoryItem(self.currentState, 
                                                      self.currentCOMInWorldFrame, 
                                                      self.currentRunningCost, 
                                                      desiredCommand,
                                                      failureMessage)
        
        cost = self.gaitPerformanceJudge.getPerformanceOfStep(simulationHistoryItem)
        self.currentRunningCost += cost
        self.simulationHistory.append(simulationHistoryItem)
    
    def terminate(self):
        self.terminated = True
        
    def hasTerminated(self):
        return self.terminated
    
    def getCost(self, modelParameters):
        self.footModel.setParameters(modelParameters)
        for i in range(self.numSteps):
            self.takeSimulationStep()
        self.debugMessage.appendMessage("numStepsSuccess", self.numSuccessfulStepsTaken)
        return self.currentRunningCost
    
    def getSimulationHistory(self):
        return self.simulationHistory

# def convertOldCostWeightsToNew(costArray):
#         self.initialCostWeights = np.array([costWeights[0], costWeights[1],
#                                      costWeights[2], costWeights[3],
#                                      costWeights[4]])
#         self.costWeights = np.copy(self.initialCostWeights)
#         self.failureWeights = np.array([costWeights[5], costWeights[5], costWeights[5], costWeights[5]])
#         self.failedWeights = costWeights[6]
#     costWeights = CostWeights();
#     return costWeights

# costWeights = np.array([20.,20.0,
#                         0.0,0.0,
#                         0.1,
#                         300.,
#                         100.])


    
