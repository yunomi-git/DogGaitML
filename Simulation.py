# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 19:22:23 2021

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




class BatchSimulation(CostEvaluator):
    def __init__(self, initialStatesList, footModel, desiredMotionsList, numSteps, costWeights): 
        super().__init__()
        self.footModel = footModel
        self.initialStatesList = initialStatesList
        self.desiredMotionsList = desiredMotionsList
        self.numSteps = numSteps
        self.costWeights = costWeights
                
    def getCost(self, parameters):
        totalCost = 0
        self.debugMessage = DebugMessage()
        i = 0

        for initialState in self.initialStatesList:
            for desiredMotion in self.desiredMotionsList:
                i += 1
                simulation = Simulation(initialState, self.footModel, desiredMotion, self.numSteps, self.costWeights)
                simulation.setOptimizerIteration(self.optimizerIteration)
                totalCost += simulation.getCost(parameters)
                self.debugMessage.appendMessage("sim" + str(i), simulation.getDebugMessage())
                
        return totalCost
    
class StochasticBatchSimulation(CostEvaluator):
    def __init__(self, initialStatesList, footModel, desiredMotionsList, numSteps, costWeights,
                 batchStatesSize, batchTasksSize): 
        super().__init__()
        self.footModel = footModel
        self.initialStatesList = initialStatesList
        self.desiredMotionsList = desiredMotionsList
        self.numSteps = numSteps
        self.costWeights = costWeights
        
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
                simulation = Simulation(initialState, self.footModel, desiredMotion, self.numSteps, self.costWeights)
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
        
    
class Simulation(CostEvaluator):
    def __init__(self, initialState, footModel, desiredTaskMotion, 
                 numSteps, costWeights):
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
        self.costWeights = costWeights
        
        #simulation should calculate the values here.
    
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
                self.applyPostStepCostActions()
            else:
                self.terminate()

            cost = self.computeCostBeforeTermination(desiredTaskMotion=self.desiredTaskMotion, 
                                command=desiredCommand, 
                                originalState=originalState, 
                                currentState=currentState, 
                                failureMessage=dynamics.getFailureMessage()); 
            
            self.lastCommand = desiredCommand
            failureMessage = dynamics.getFailureMessage()
        else:
            cost = self.computeCostAfterTermination()
            failureMessage = None #TODO should change this to "last error message"
        
        self.currentRunningCost += cost
            
        simulationHistoryItem = SimulationHistoryItem(self.currentState, 
                                                      self.currentCOMInWorldFrame, 
                                                      self.currentRunningCost, 
                                                      desiredCommand,
                                                      failureMessage)
        self.simulationHistory.append(simulationHistoryItem)
        
    def computeCostBeforeTermination(self, desiredTaskMotion, command, originalState, currentState, failureMessage):
        if failureMessage.failureHasOccurred():
            # "stability" ability to take all n steps
            cost = self.computeCostFromInitialFailure(desiredTaskMotion, 
                                                        command, 
                                                        originalState, 
                                                        currentState, 
                                                        failureMessage); 
        else:
            # cost if motion is successful
            cost = self.computeCostFromMotion(desiredTaskMotion, 
                                                command, 
                                                originalState, 
                                                currentState, 
                                                failureMessage); 
        return cost
        
    def applyPostStepCostActions(self):
        pass
        # self.costWeights.comNormTranslationError += self.costWeights.comNormTranslationErrorCostIncPerStep
        # self.costWeights.comNormRotationError += self.costWeights.comNormRotationErrorCostIncPerStep
        # self.costWeights.footNormErrorFromIdeal += self.costWeights.footNormErrorFromIdealCostIncPerStep
            
    def computeCostFromMotion(self, desiredTaskMotion, command, originalState, currentState, failureMessage):
        cost = 0
        
        # "speed" error in command motion from input motion
        desiredTranslation = np.array([desiredTaskMotion.translationX, desiredTaskMotion.translationY])
        normDistErr = np.linalg.norm(desiredTranslation - command.comTranslation) / DogModel.maximumCOMTranslationDistance
        normAngErr = 1 - np.dot(mu.getUnitVectorFromAngle(desiredTaskMotion.relativeRotation),
                                mu.getUnitVectorFromAngle(command.comRelativeRotation))
        
        # "smoothness" difference between current and last motion
        if self.lastCommand is None:
            normDDist = 0
            normDAng = 0
                
        else:
            normDDist = np.linalg.norm(self.lastCommand.comTranslation - command.comTranslation) / DogModel.maximumCOMTranslationDistance
            normDAng = 1 - np.dot(mu.getUnitVectorFromAngle(self.lastCommand.comRelativeRotation),
                                  mu.getUnitVectorFromAngle(command.comRelativeRotation))
    
        # "convergence" distance of feet from ideal
        currentFootState = currentState.footState
        dogModel = DogModel(originalState)
        distances = dogModel.getPostMotionFootDistancesFromIdeal(currentFootState, command.getTaskMotion())
        normFootErr = np.sum(distances) / DogModel.maximumCOMTranslationDistance
        
        cost += (self.costWeights.comNormTranslationErrorInitial * normDistErr +
                 self.costWeights.comNormRotationErrorInitial * normAngErr + 
                 self.costWeights.comTranslationSmoothnessInitial * normDDist + 
                 self.costWeights.comRotationSmoothnessInitial * normDAng +
                 self.costWeights.footNormErrorFromIdealInitial * normFootErr)
        
        return cost
    
    def computeCostAfterTermination(self):
        return self.costWeights.failureStepsAfterTermination
    
    def computeCostFromInitialFailure(self, desiredTaskMotion, command, originalState, currentState, failureMessage):
        cost = (self.costWeights.failureComUnsupportedAtStart * failureMessage.comIsNotContainedAtStart +
                     self.costWeights.failureComUnsupportedAtEnd * failureMessage.comIsNotContainedAtEnd +
                     self.costWeights.failureSwingFootOutOfBounds * failureMessage.swingFootPlacementOutOfBounds +
                     self.costWeights.failureAnchoredFootOutOfBounds * failureMessage.anchoredFootPlacementsOutOfBounds 
                                                                     * failureMessage.numAnchoredFootPlacementsOutOfBounds)
    
        # distance of foot placements from maximum foot range
        if (failureMessage.swingFootPlacementOutOfBounds or failureMessage.anchoredFootPlacementsOutOfBounds):
            currentFootState = currentState.footState
            dogModel = DogModel(originalState)
            distances = dogModel.getPostMotionFootDistancesFromIdeal(currentFootState, command.getTaskMotion())
            
            outOfBoundDistances = distances - DogModel.maximumFootDistanceFromIdeal
            outOfBoundDistances *= (outOfBoundDistances > 0) # only erroneous feet are considered
            
            normFootErr = np.sum(outOfBoundDistances) / DogModel.maximumCOMTranslationDistance
            self.debugMessage.appendMessage("normFootErr", normFootErr)
            self.debugMessage.appendMessage("numBadFeet", failureMessage.numAnchoredFootPlacementsOutOfBounds + failureMessage.swingFootPlacementOutOfBounds)
            cost += normFootErr * self.costWeights.failureFootOutOfBoundsErrorFromIdeal
            
        # distance of com from centroid
        if (failureMessage.comIsNotContainedAtEnd):
            dogModel = DogModel(currentState)
            footToMove = command.footToMove
            supportingFeet = dogModel.getEveryFootExcept(footToMove)
            centroid = np.mean(supportingFeet, axis = 0)
            comDistanceFromCentroid = np.linalg.norm(centroid)
            normComError = comDistanceFromCentroid / DogModel.maximumCOMTranslationDistance
            cost += normComError * self.costWeights.failureComEndErrorFromCentroid
        
        return cost
    
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
    
        
        
    
        
        
class SimulationHistoryItem():
    def __init__(self, state, comInWorldFrame, runningCost, command, failureMessage):
        self.state = state;
        self.comInWorldFrame = comInWorldFrame
        self.runningCost = runningCost
        self.command = command
        self.failureMessage = failureMessage
        
@dataclass 
class CostWeights():
    failureStepsAfterTermination : float = 0.0
    failureSwingFootOutOfBounds : float = 0.0
    failureAnchoredFootOutOfBounds : float = 0.0
    failureComUnsupportedAtStart : float = 0.0
    failureComUnsupportedAtEnd : float = 0.0
    failureFootOutOfBoundsErrorFromIdeal : float = 0.0
    failureComEndErrorFromCentroid : float = 0.0
    
    comNormTranslationErrorInitial : float = 0.0
    comNormRotationErrorInitial : float = 0.0
    comTranslationSmoothnessInitial : float = 0.0
    comRotationSmoothnessInitial : float = 0.0
    footNormErrorFromIdealInitial : float = 0.0
    
    # comNormTranslationErrorFinalStep : float = 0.0
    # comNormRotationErrorFinalStep : float = 0.0
    # footNormErrorFromIdealFinalStep : float = 0.0
    
    # comNormTranslationErrorMatureInNIterations : float = 0.0
    # comNormRotationErrorMatureInNIterations : float = 0.0
    # footNormErrorFromIdealMatureInNIterations : float = 0.0
    
    
    
    
    
    
    
    
    
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


    
