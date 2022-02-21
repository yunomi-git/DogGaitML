# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 19:55:25 2022

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
from SimulationHistoryItem import SimulationHistoryItem
from DebugMessage import DebugMessage


@dataclass 
class CostWeights():
    pass
    
class GaitPerformanceJudge(ABC):
    def __init__(self, costWeights: CostWeights):
        self.costWeights = costWeights
        self.debugMessage = DebugMessage()
        
    def setDebugMessage(self, debugMessage: DebugMessage):
        self.debugMessage = debugMessage
        
    def setIterationNum(self, iterationNum):
        self.iterationNum = iterationNum
        
    @abstractmethod
    def getPerformanceOfStep(self, simulationHistoryItem: SimulationHistoryItem):
        if (not simulationHistoryItem.hasTerminated()):
            # if not SimulationHistoryItem.failedOnThisStep():
            #     self.applyPostStepCostActions()
            cost = self.computeCostBeforeTermination(desiredTaskMotion=simulationHistoryItem.desiredTaskMotion, 
                                                    command=simulationHistoryItem.desiredCommand, 
                                                    originalState=simulationHistoryItem.originalState, 
                                                    currentState=simulationHistoryItem.currentState, 
                                                    failureMessage=simulationHistoryItem.getFailureMessage()); 
        else:
            cost = self.computeCostAfterTermination()
            
        return cost
        
    
    @abstractmethod
    def applyPostStepCostActions(self):
        pass
    
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
    
    @abstractmethod
    def computeCostFromMotion(self, desiredTaskMotion, command, originalState, currentState, failureMessage):
        pass
    
    @abstractmethod
    def computeCostAfterTermination(self):
        pass
    
    @abstractmethod
    def computeCostFromInitialFailure(self, desiredTaskMotion, command, originalState, currentState, failureMessage):
        pass
    
@dataclass 
class CostWeightsJarvis(CostWeights):
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
    
class GaitJudgeJarvis(GaitPerformanceJudge):
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