# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 21:12:00 2021

@author: Evan Yu
"""

from abc import ABC, abstractmethod
import numpy as np
from Polygon import Triangle2D
from DogUtil import Command, State, TaskMotion, DogModel
import MathUtil as mu


class FootModel(ABC):
    def __init__(self, parameters=None):
        pass
    
    @abstractmethod
    def computeCommandFromState(self, state, desiredMotion):
        pass
    
    @abstractmethod
    def setParameters(self, parameters):
        pass
    
    @abstractmethod
    def getNumParameters(self):
        pass
    
    
class SimpleFootModel(FootModel):
    def __init__(self, parameters=None):
        self.expandedStateDims = 31;
        self.outputDims = 6;
        if (parameters is not None):
            self.parameterModel = self.convertParametersToModel(parameters)
        self.dogModel = DogModel()
        
    def setParameters(self, parameters):
        self.parameterModel = self.convertParametersToModel(parameters)
    
    def computeCommandFromState(self, state, desiredTaskMotion):
        footState = state.footState
        self.dogModel.setState(state)
        feetThatCanMove = self.dogModel.getFeetThatCanMove();
        
        outputList = [];
        for foot in feetThatCanMove:
            expandedState = self.getExpandedState(state, desiredTaskMotion, foot);
            output = np.matmul(self.parameterModel, expandedState);
            outputList.append(output)

        i = self.getBestOutputIndexFromList(outputList);
        bestOutput = outputList[i];
        bestFoot = feetThatCanMove[i];
        
        # construct desired motion from output
        desiredCommand = Command(bestFoot, 
                                 np.array([bestOutput[1], bestOutput[2]]),
                                 np.array([bestOutput[3], bestOutput[4]]),
                                 bestOutput[5])
        return desiredCommand;
    
    def getExpandedState(self, state, desiredTaskMotion, footToMove):
        footState = state.footState;
        self.dogModel.setState(state) # state is technically already set. Should I do it again here?
        ofi = [footToMove] + self.dogModel.getOtherFeetOrderedIndices(footToMove)
        idealFootState = self.dogModel.getIdealFootStateFromOriginalCom(desiredTaskMotion)
        footStateInIdealCOM = footState - np.array([desiredTaskMotion.translationX, desiredTaskMotion.translationY])
        origFootDistanceFromOrigCOMs = np.linalg.norm(footState, axis=1)
        origFootDistanceFromIdealFoots = self.dogModel.getPreMotionFootDistancesFromIdeal(desiredTaskMotion)
        desiredRotationFromDefault = state.absoluteRotation + desiredTaskMotion.relativeRotation
        expandedState = np.array([
            1,  # 1
            footState[ofi[0],0],  #foot i orig x
            footState[ofi[0],1],  #foot i orig y
            footState[ofi[1],0],  #foot op orig x
            footState[ofi[1],1],  #foot op orig y
            footState[ofi[2],0],  #foot hz orig x
            footState[ofi[2],1],  #foot hz orig y
            footState[ofi[3],0],  #foot vt orig x
            footState[ofi[3],1],  #foot vt orig y
            idealFootState[ofi[0],0],  #foot i ideal x
            idealFootState[ofi[0],1], #foot i ideal y
            desiredTaskMotion.translationX, #desired COM x
            desiredTaskMotion.translationY, #desired COM y
            np.dot(footState[ofi[1],:], footState[ofi[2],:]), #orig com,feet op dot hz
            np.dot(footState[ofi[1],:], footState[ofi[3],:]), #orig com,feet op dot vt
            np.dot(footState[ofi[2],:], footState[ofi[3],:]), #orig com,feet hz dot vt
            np.dot(footStateInIdealCOM[ofi[1],:], footStateInIdealCOM[ofi[2],:]), #ideal com orig feet op dot hz
            np.dot(footStateInIdealCOM[ofi[1],:], footStateInIdealCOM[ofi[3],:]), #ideal com orig feet op dot vt
            np.dot(footStateInIdealCOM[ofi[2],:], footStateInIdealCOM[ofi[3],:]), #ideal com orig feet hz dot vt
            origFootDistanceFromOrigCOMs[ofi[0]], #orig foot vs com dist i
            origFootDistanceFromOrigCOMs[ofi[1]], #orig foot vs com dist op
            origFootDistanceFromOrigCOMs[ofi[2]], #orig foot vs com dist hz
            origFootDistanceFromOrigCOMs[ofi[3]], #orig foot vs com dist vt
            origFootDistanceFromIdealFoots[ofi[0]], #orig foot vs ideal foot i
            origFootDistanceFromIdealFoots[ofi[1]], #orig foot vs ideal foot op
            origFootDistanceFromIdealFoots[ofi[2]], #orig foot vs ideal foot hz
            origFootDistanceFromIdealFoots[ofi[3]], #orig foot vs ideal foot vt
            desiredTaskMotion.relativeRotation, #desired angle
            mu.getSignedVectorAngleFromRotation(footStateInIdealCOM[ofi[1],:], desiredRotationFromDefault), #orig foot-desired com angle from desired angle op
            mu.getSignedVectorAngleFromRotation(footStateInIdealCOM[ofi[2],:], desiredRotationFromDefault), #orig foot-desired com angle from desired angle hz
            mu.getSignedVectorAngleFromRotation(footStateInIdealCOM[ofi[3],:], desiredRotationFromDefault) #orig foot-desired com angle from desired angle vt
            ])
        return expandedState

    def convertParametersToModel(self, parameters):
        model = parameters.reshape([self.outputDims, self.expandedStateDims])
        return model
    
    # chooses highest score
    def getBestOutputIndexFromList(self, outputList):
        if outputList[0][0] > outputList[1][0]:
            return 0
        else:
            return 1
        # return bestIndex
            
    def getNumParameters(self):
        return (self.expandedStateDims * self.outputDims)

        

    
    
