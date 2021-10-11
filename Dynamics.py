# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 18:51:00 2021

@author: Evan Yu
"""
import numpy as np
from DogUtil import Command, State, DesiredMotion, DogModel
from Polygon import Triangle2D

class Dynamics():
    def __init__(self, initialState):
        self.state = initialState #should I hold state if dogmodel already holds the state?
        self.dogModel = DogModel(initialState) 
        
        self.failureMessage = DynamicsFailureMessage()
    
    def applyCommand(self, command):
        if (self.hasNotFailed()):
            self.dogModel.setState(self.state)
            footToMove = command.footToMove
            nextFootState = np.copy(self.state)
            nextFootState[footToMove, :] = command.footPosition
            
            nextAbsoluteRotation = self.currentRelativeRotation + command.comRelativeRotation
            
            nextState = State(nextFootState, nextAbsoluteRotation)
            
            self.checkFailureAfterMotion(nextState, command)
            if (self.hasNotFailed()):
                self.state = nextState
                            
    
    def checkFailureAfterMotion(self, nextState, command):
        failureMessage = DynamicsFailureMessage()
        footToMove = command.footToMove
        
        # each foot needs to be within x distance from default (at ideal)
        idealFootState = self.dogModel.getIdealFootState(command.getDesireMotion())
        nextFootState = np.copy(nextState.footState);
        nextFootState[footToMove, :] += command.footTranslation
        footDistanceFromIdealFoots = np.linalg.norm(nextFootState - idealFootState, axis=1)
        distanceViolations = footDistanceFromIdealFoots > DogModel.maximumFootDistanceFromIdeal
        
        if (distanceViolations[footToMove]):
            failureMessage.setSwingFootPlacementOutOfBoundsFailure()
        
        nonMovingFeet = self.dogModel.getOtherFeetOrderedIndices(footToMove)
        numViolations = np.count_nonzero(distanceViolations[nonMovingFeet])
        if numViolations > 0:
            failureMessage.setAnchoredFootPlacementsOutOfBoundsFailure(numViolations)
        
        # com must be within poylygon at beginning of motion
        # com must be within polygon at end of motion
        finalComTranslation = command.comTranslation
        supportPolygon = Triangle2D(self.dogModel.getEveryFootExceptUnordered(footToMove))
        beginningIsEnclosed = supportPolygon.isEnclosed(np.array([0,0]))
        endIsEnclosed = supportPolygon.isEnclosed(finalComTranslation)
        if not beginningIsEnclosed:
            failureMessage.setCOMIsNotContainedAtStartFailure()
        if not endIsEnclosed:
            failureMessage.setCOMIsNotContainedAtEndFailure()
    
    def hasFailed(self):
        return self.dynamicsFailureMessage.failureHasOccurred()
    
    def getFailureMessage(self):
        return self.failureMessage
    
    def getCurrentState(self):
        return self.state
    
     
class DynamicsFailureMessage:
    def __init__(self):
        self.swingFootPlacementOutOfBounds = False
        self.anchoredFootPlacementsOutOfBounds = False
        self.numAnchoredFootPlacementsOutOfBounds = 0
        self.comIsNotContainedAtStart = False
        self.comIsNotContainedAtEnd = False
        
    def setSwingFootPlacementOutOfBoundsFailure(self):
        self.swingFootPlacementOutofBounds = True
        
    def setAnchoredFootPlacementsOutOfBoundsFailure(self, num):
        self.swingFootPlacementOutofBounds = True
        self.numAnchoredFootPlacementsOutOfBounds = num
        
    def setCOMIsNotContainedAtStartFailure(self):
        self.comIsNotContainedAtStart = True
        
    def setCOMIsNotContainedAtEndFailure(self):
        self.comIsNotContainedAtEnd = True
          
    def failureHasOccurred(self):
        return (self.swingFootPlacementOutOfBounds or
                self.anchoredFootPlacementsOutOfBounds or
                self.comIsNotContainedAtEnd or
                self.comIsNotContainedAtStart)
              