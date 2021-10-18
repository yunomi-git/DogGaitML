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

class BatchSimulation(CostEvaluator):
    def __init__(self, initialStatesList, footModel, desiredMotionsList, numSteps, costWeights): 
        self.footModel = footModel
        self.initialStatesList = initialStatesList
        self.desiredMotionsList = desiredMotionsList
        self.numSteps = numSteps
        self.costWeights = costWeights
                
    def getCost(self, parameters):
        totalCost = 0
        for initialState in self.initialStatesList:
            for desiredMotion in self.desiredMotionsList:
                simulation = Simulation(initialState, self.footModel, desiredMotion, self.numSteps, self.costWeights)
                totalCost += simulation.getCost(parameters)
                
        return totalCost
    
class Simulation(CostEvaluator):
    def __init__(self, initialState, footModel, desiredTaskMotion, numSteps, costWeights):
        self.simulationHistory = [];

        self.currentCOMInWorldFrame = np.array([0.,0.]);
        self.currentState = initialState;
        self.lastCommand = None
        
        self.footModel = footModel
        self.numSteps = numSteps
        self.costWeights = np.array([costWeights[0], costWeights[1],
                                     costWeights[2], costWeights[3],
                                     costWeights[4]])
        self.failureWeights = np.array([costWeights[5], costWeights[5], costWeights[5], costWeights[5]])
        self.failedWeights = costWeights[6]
        
        self.desiredTaskMotion = desiredTaskMotion
        
        self.terminated = False
        self.currentRunningCost = 0
        
        self.simulationHistory.append(SimulationHistoryItem(self.currentState, 
                                              self.currentCOMInWorldFrame, 
                                              self.currentRunningCost, 
                                              None,
                                              None))

        
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
                # self.currentState = dynamics.getCurrentState()
                # self.currentCOMInWorldFrame = self.currentCOMInWorldFrame + desiredCommand.comTranslation
                self.currentState = currentState
                self.currentCOMInWorldFrame = currentCOMInWorldFrame
            else:
                self.terminate()
            # self.currentState = dynamics.getCurrentState()
            # self.currentCOMInWorldFrame = self.currentCOMInWorldFrame + desiredCommand.comTranslation
            # if dynamics.hasFailed():
            #     self.terminate()
                
            self.currentRunningCost += self.computeCostFromMotion(desiredTaskMotion=self.desiredTaskMotion, 
                                                              command=desiredCommand, 
                                                              originalState=originalState, 
                                                              currentState=currentState, 
                                                              failureMessage=dynamics.getFailureMessage()); 
            self.lastCommand = desiredCommand
            failureMessage = dynamics.getFailureMessage()
        else:
            self.currentRunningCost += self.computeCostAfterFail()
            failureMessage = None #TODO should change this to "last error message"
            
        simulationHistoryItem = SimulationHistoryItem(self.currentState, 
                                                      self.currentCOMInWorldFrame, 
                                                      self.currentRunningCost, 
                                                      desiredCommand,
                                                      failureMessage)
        self.simulationHistory.append(simulationHistoryItem)
        
        
            
            
    def computeCostFromMotion(self, desiredTaskMotion, command, originalState, currentState, failureMessage):
        cost = 0
        if failureMessage.failureHasOccurred():
            # "stability" ability to take all n steps
            failures = np.array([failureMessage.comIsNotContainedAtStart, 
                                 failureMessage.comIsNotContainedAtEnd, 
                                 failureMessage.swingFootPlacementOutOfBounds, 
                                 failureMessage.anchoredFootPlacementsOutOfBounds * failureMessage.numAnchoredFootPlacementsOutOfBounds])
            failureCost = np.dot(failures, self.failureWeights)
            cost += failureCost

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
        
        errors = np.array([normDistErr, normAngErr, 
                           normDDist, normDAng, 
                           normFootErr])
        cost += np.dot(self.costWeights, np.square(errors))
        return cost
    
    def computeCostAfterFail(self):
        return self.failedWeights
    
    def terminate(self):
        self.terminated = True
        
    def hasTerminated(self):
        return self.terminated
    
    def getCost(self, modelParameters):
        self.footModel.setParameters(modelParameters)
        for i in range(self.numSteps):
            self.takeSimulationStep()
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
