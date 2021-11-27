# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 19:26:13 2021

@author: Evan Yu
"""

from dataclasses import dataclass
import numpy as np
import MathUtil as mu
from Polygon import Triangle2D

@dataclass
class TaskMotion:
    translationX : float
    translationY : float
    relativeRotation : float
    
class State:
    def __init__(self, footState, absoluteRotation):
        self.footState = footState
        self.absoluteRotation = absoluteRotation
        
class Command:
    def __init__(self, footToMove, footTranslation, comTranslation, comRelativeRotation):
        self.footToMove = footToMove
        self.footTranslation = footTranslation
        self.comTranslation = comTranslation
        self.comRelativeRotation = comRelativeRotation
        
    def getTaskMotion(self):
        taskMotion = TaskMotion(translationX=self.comTranslation[0], 
                                      translationY=self.comTranslation[1], 
                                      relativeRotation=self.comRelativeRotation)
        return taskMotion
    


class DogModel():
    # in groups of 4,feet always listed in order UR, BR, BL, UL
    # in groups of 3, feet listed as Opposing, Horiz, Vert
    maximumFootDistanceFromIdeal = 100.0
    maximumCOMTranslationDistance = maximumFootDistanceFromIdeal/4.0
    ordered3FootMaps = {0:[2,3,1], 1:[3,2,0], 2:[0,1,3], 3:[1,0,2]}
    
    def __init__(self, state=None):
        self.halfLength = 112.5;
        self.halfWidth = 60.0 + 13.97;
        self.defaultFootState = np.array([[ self.halfLength,-self.halfWidth],
                                          [-self.halfLength,-self.halfWidth],
                                          [-self.halfLength, self.halfWidth],
                                          [ self.halfLength, self.halfWidth]]);
        if (state is not None):
            self.setState(state)
        else:
            self.setState(State(self.defaultFootState, 0.0))
        
       
        
    def setState(self, state):
        self.footState = state.footState
        self.absoluteRotation = state.absoluteRotation
        
    def getIdealFootStateFromOriginalCom(self, taskMotion):
        translation = np.array([taskMotion.translationX, taskMotion.translationY])
        fullRotation = self.absoluteRotation + taskMotion.relativeRotation
        fullRotationMat = mu.getRotationMatrix(fullRotation)
        desiredFootState = np.matmul(fullRotationMat, (self.defaultFootState).T).T+translation
        return desiredFootState
    
    def getPostMotionFootDistancesFromIdeal(self, postMotionFootState, taskMotion):
        idealFootState = self.getIdealFootStateFromOriginalCom(taskMotion)
        postMotionFootStateFromOriginalCOM = postMotionFootState + np.array([taskMotion.translationX, 
                                                                             taskMotion.translationY])
        distances = np.linalg.norm(postMotionFootStateFromOriginalCOM - idealFootState, axis=1)
        return distances
    
    def getPreMotionFootDistancesFromIdeal(self, taskMotion):
        idealFootState = self.getIdealFootStateFromOriginalCom(taskMotion)
        distances = np.linalg.norm(self.footState - idealFootState, axis=1)
        return distances
    
    def getFeetThatCanMove(self):
        feetThatCanMove = []
        
        for i in range(4):
            supportTriangle = Triangle2D(self.getEveryFootExcept(i));
            if supportTriangle.isPointEnclosed(np.array([0.,0.])):
                feetThatCanMove.append(i)
                if len(feetThatCanMove) == 2:
                    break
        
        return feetThatCanMove
    
    def getEveryFootExcept(self, foot):
        return self.footState[self.getOtherFeetOrderedIndices(foot)]        
    
    def getOtherFeetOrderedIndices(self, i):
        orderedIndices = self.ordered3FootMaps[i]
        return orderedIndices