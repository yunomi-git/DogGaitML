# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 21:12:00 2021

@author: Evan Yu
"""

from abc import ABC, abstractmethod
import numpy as np
from Polygon import Triangle2D
from SimulationDataStructures import Command, State, DesiredMotion, getRotationMatrix


class FootModel(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def computeCommandFromState(self, state, desiredMotion, parameters):
        pass
    
    @abstractmethod
    def getNumParameters(self):
        pass
    
    
class SimpleFootModel(FootModel):
    # in groups of 4,feet always listed in order UR, BR, BL, UL
    # in groups of 3, feet listed as Opposing, Horiz, Vert
    # state is 4x2 numpy array
    def __init__(self, parameters):
        self.expandedStateDims = 31;
        self.outputDims = 6;
        self.halfLength = 112.5;
        self.halfWidth = 60.0 + 13.97;
        self.defaultFootState = np.array([[ self.halfLength,-self.halfWidth],
                                          [-self.halfLength,-self.halfWidth],
                                          [-self.halfLength, self.halfWidth],
                                          [ self.halfLength, self.halfWidth]]);
        self.ordered3FootMaps = {0:[2,3,1], 1:[3,2,0], 2:[0,1,3], 3:[1,0,2]}
        self.parameterModel = self.convertParametersToModel(parameters)
    
    def computeCommandFromState(self, state, desiredMotion):
        footState = state.footState
        feetThatCanMove = self.getFeetThatCanMove(footState);
        
        outputList = [];
        for foot in feetThatCanMove:
            expandedState = self.getExpandedState(state, desiredMotion, foot);
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
            
    def getFeetThatCanMove(self, footState):
        feetThatCanMove = []
        
        for i in range(4):
            supportTriangle = Triangle2D(self.getEveryFootExcept(footState, i));
            if supportTriangle.isPointEnclosed(np.array([0,0])):
                feetThatCanMove.append(i)
                if len(feetThatCanMove) == 2:
                    break
        
        return feetThatCanMove
    
    def getEveryFootExcept(self, footState, foot):
        feet = np.copy(footState);
        feet = feet[self.getOtherFeetOrderedIndices(foot)]        
        return feet
    
    def getOtherFeetOrderedIndices(self, i):
        orderedIndices = self.ordered3FootMaps[i]
        return orderedIndices
    
    def getExpandedState(self, state, desiredMotion, footToMove):
        footState = state.footState;
        ofi = [footToMove] + self.getOtherFeetOrderedIndices(footToMove)
        desiredFootState = self.getDesiredFootState(state, desiredMotion)
        footStateInIdealCOM = footState - np.array([desiredMotion.translationX, desiredMotion.translationY])
        origFootDistanceFromOrigCOMs = np.linalg.norm(footState, axis=1)
        origFootDistanceFromIdealFoots = np.linalg.norm(footState - desiredFootState, axis=1)
        desiredRotationFromDefault = state.rotation + desiredMotion.rotation
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
            desiredFootState[ofi[0],0],  #foot i ideal x
            desiredFootState[ofi[0],1], #foot i ideal y
            desiredMotion.translationX, #desired COM x
            desiredMotion.translationY, #desired COM y
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
            desiredMotion.rotation, #desired angle
            self.getSignedVectorAngleFromRotation(footStateInIdealCOM[ofi[1],:], desiredRotationFromDefault), #orig foot-desired com angle from desired angle op
            self.getSignedVectorAngleFromRotation(footStateInIdealCOM[ofi[2],:], desiredRotationFromDefault), #orig foot-desired com angle from desired angle hz
            self.getSignedVectorAngleFromRotation(footStateInIdealCOM[ofi[3],:], desiredRotationFromDefault) #orig foot-desired com angle from desired angle vt
            ])
        return expandedState
    
    def getDesiredFootState(self, origState, desiredMotion):
        translation = np.array([desiredMotion.translationX, desiredMotion.translationY])
        fullRotation = origState.rotation + desiredMotion.rotation
        fullRotationMat = getRotationMatrix(fullRotation)
        desiredFootState = np.matmul(fullRotationMat, (self.defaultFootState).T).T+translation
        return desiredFootState
    
    def convertParametersToModel(self, parameters):
        model = parameters.reshape([self.outputDims, self.expandedStateDims])
        return model
    
    def getBestOutputIndexFromList(self, outputList):
        firstOutput = outputList[0]
        bestValue = firstOutput[0]
        bestIndex = 0;
        i = 0;
        for output in outputList:
            value = output[0]
            if value > bestValue:
                bestIndex = i
                bestValue = value
            i += 1
        return bestIndex
            
    def getNumParameters(self):
        return (self.expandedStateDims * self.outputDims)
    
    def getSignedVectorAngleFromRotation(self, v, rotation):
        rotationMat = getRotationMatrix(-rotation)
        v_rotated = rotationMat @ v
        angle = np.degrees(np.arctan2(v_rotated[1], v_rotated[0]))
        return angle
        

    
    
def testFeetThatCanMove():
    footModel = SimpleFootModel(np.ones(6*31))
    state = np.array([[1.0,1.0],[1.0,-1.0],[-1.0,-1.0],[-1.0,1.0]])
    COMPositions = np.array([[0., 0.5], [0.5,0],[0,-0.5],[-0.5,0]])
    correctAnswers = [[1,2],[2,3],[0,3],[0,1]]
    for i in range(4):
        COMPosition = COMPositions[i,:]
        tempState = state - COMPosition;
        movableFeet = footModel.getFeetThatCanMove(tempState)
        if (not (sorted(movableFeet) == sorted(correctAnswers[i][:]))):
            print("\n----")
            print(i)
            print("error in testFeetThatCanMove\n----")
        else:
            print(".")
    
def testGetFeetExcept():
    state = np.array([[0,0],[1,1],[2,2],[3,3]])
    footModel = SimpleFootModel(np.ones(6*31))
    for i in range(4):
        allFeetExcepti = footModel.getEveryFootExcept(state, i)
        if (any((allFeetExcepti[:]==state[i,:]).all(1)) != False):
            print("\n----")
            print(i)
            print("error in getFeetExcept\n----")
        else:
            print(".")
            
def testGetOrderedIndices():
    footModel = SimpleFootModel(np.ones(6*31))
    accurateList = [[2,3,1],[3, 2, 0], [0,1,3],[1,0,2]]
    for i in range(4):
        if (footModel.getOtherFeetOrderedIndices(i) != accurateList[i][:]):
            print("\n----")
            print(i)
            print("error in getOrderedIndices\n----")
        else:
            print(".")
        
def testGetExpandedStates():
    footModel = SimpleFootModel(np.ones(6*31))
    defaultFootState = np.array([[ 112.5 ,  -73.97],
                                   [-112.5 ,  -73.97],
                                   [-112.5 ,   73.97],
                                   [ 112.5 ,   73.97]])
    rotation = 5
    originalState = State(defaultFootState, rotation)
    footToMove = 0
    desiredMotion = DesiredMotion(10, 20, 25)
    expandedState = footModel.getExpandedState(originalState, desiredMotion, footToMove)
    correctExpandedState = np.array([
            1,  # 1
            112.5,  #foot i orig x
            -73.97,  #foot i orig y
            -112.5,  #foot op orig x
            73.97,  #foot op orig y
            112.5,  #foot hz orig x
            73.97,  #foot hz orig y
            -112.5,  #foot vt orig x
            -73.97,  #foot vt orig y
            144.41285793,  #foot i ideal x
            12.19010088, #foot i ideal y
            10, #desired COM x
            20, #desired COM y
            -7184.6891000000005, #orig com,feet op dot hz
            7184.6891000000005, #orig com,feet op dot vt
            -18127.8109, #orig com,feet hz dot vt
            -9643.4891, #ideal com orig feet op dot hz
            9934.6891, #ideal com orig feet op dot vt
            -17627.8109, #ideal com orig feet hz dot vt
            134.63955919, #orig foot vs com dist i
            134.63955919, #orig foot vs com dist op
            134.63955919, #orig foot vs com dist hz
            134.63955919, #orig foot vs com dist vt
            91.88032153, #orig foot vs ideal foot dist i
            47.67254029, #orig foot vs ideal foot dist op
            78.54798161, #orig foot vs ideal foot dist hz
            67.415719, #orig foot vs ideal foot dist vt
            25, #desired angle
            126.22308128821268, #orig foot-desired com angle from desired angle op
            -2.2315538083380275, #orig foot-desired com angle from desired angle hz
            -172.50811304048392 #orig foot-desired com angle from desired angle vt
            ])
    if (not np.all(np.isclose(correctExpandedState, expandedState))):
        print("\n----")
        print(expandedState)
        print(np.isclose(correctExpandedState, expandedState))
        print("error in testGetExpandedStates\n----")
    else:
        print(".")
    
    
def main():
    # testGetFeetExcept()
    # testGetOrderedIndices()
    # testFeetThatCanMove()
    testGetExpandedStates()
    
    
if __name__ == "__main__":
    main()