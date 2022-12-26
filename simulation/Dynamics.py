# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 18:51:00 2021

@author: Evan Yu
"""
import numpy as np
from model.DogUtil import Command, State, TaskMotion, DogModel
from util.Polygon import Triangle2D

class Dynamics():
    def __init__(self, initialState):
        self.state = initialState #should I hold state if dogmodel already holds the state?
        self.dogModel = DogModel(initialState) 
        
        self.failureMessage = DynamicsFailureMessage()
    
    def applyCommand(self, command):
        if (not self.hasFailed()):
            self.dogModel.setState(self.state)
            footToMove = command.footToMove
            nextFootState = np.copy(self.state.footState)
            nextFootState[footToMove, :] += command.footTranslation
            nextFootState -= command.comTranslation
            
            nextAbsoluteRotation = self.state.absoluteRotation + command.comRelativeRotation
            
            nextState = State(nextFootState, nextAbsoluteRotation)
            
            self.checkFailureAfterMotion(nextState, command)
            # if (not self.hasFailed()):
            self.state = nextState #saves the broken state
                            
    
    def checkFailureAfterMotion(self, nextState, command):
        failureMessage = DynamicsFailureMessage()
        footToMove = command.footToMove
        
        # each foot needs to be within x distance from default (at ideal)
        footDistanceFromIdealFoots = self.dogModel.getPostMotionFootDistancesFromIdeal(nextState.footState, 
                                                                                       command.getTaskMotion())
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
        supportPolygon = Triangle2D(self.dogModel.getEveryFootExcept(footToMove))
        beginningIsEnclosed = supportPolygon.isPointEnclosed(np.array([0,0]))
        endIsEnclosed = supportPolygon.isPointEnclosed(finalComTranslation)
        if not beginningIsEnclosed:
            failureMessage.setCOMIsNotContainedAtStartFailure()
        if not endIsEnclosed:
            failureMessage.setCOMIsNotContainedAtEndFailure()
            
        self.failureMessage = failureMessage
    
    def hasFailed(self):
        return self.failureMessage.failureHasOccurred()
    
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
        self.swingFootPlacementOutOfBounds = True
        
    def setAnchoredFootPlacementsOutOfBoundsFailure(self, num):
        self.anchoredFootPlacementsOutOfBounds = True
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
    
    def __str__(self):
        output = ""
        if (self.swingFootPlacementOutOfBounds):
            output += "Swing foot placement out of bounds\n"
        if (self.anchoredFootPlacementsOutOfBounds):
            output += "Anchored foot placement out of bounds, "
            output += str(self.numAnchoredFootPlacementsOutOfBounds) + "\n"
        if (self.comIsNotContainedAtStart):
            output += "COM Not Contained At Start\n"
        if (self.comIsNotContainedAtEnd):
            output += "COM Not Contained At End\n"
        return output