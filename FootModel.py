# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 21:12:00 2021

@author: Evan Yu
"""

from abc import ABC, abstractmethod
import numpy as np
from Polygon import Triangle2D

class FootModel(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def getDesiredMotionFromState(self, state, desiredMotion, parameters):
        pass
    
    @abstractmethod
    def getNumParameters(self):
        pass
    
    
class SimpleFootModel(FootModel):
    # in groups of 4,feet always listed in order UR, BR, BL, UL
    # in groups of 3, feet listed as Opposing, Horiz, Vert
    # state is 4x2 numpy array
    def __init__(self):
        self.expandedStateDims = 31;
        self.outputDims = 6;
        self.defaultFeetPositions = np.array([[1,1],[1,2],[2,3],[1,4]]) #TODO
        self.oppositeFootMap = {0:2, 1:3, 2:0, 3:1}
        self.horizontalFootMap = {0:3, 1:2, 2:1, 3:0}
        self.verticalFootMap = {0:1, 1:0, 2:3, 3:2}
    
    def getDesiredMotionFromState(self, state, desiredMotion, parameters):
        feetThatCanMove = self.getFeetThatCanMove(state);
        parameterModel = self.convertParametersToModel(parameters);
        
        outputList = [];
        for foot in feetThatCanMove:
            expandedState = self.getExpandedState(state, desiredMotion, foot);
            output = np.matmul(parameterModel, expandedState);
            outputList.append(output)

        i = self.getBestOutputIndexFromList(outputList);
        bestOutput = outputList[i];
        bestFoot = feetThatCanMove[i];
        
        # construct desired motion from output
        desiredMotion = bestOutput;
        desiredMotion[0] = bestFoot;
        return bestOutput;
    
    def getOtherFeetOrderedIndices(self, i):
        orderedIndices = [self.oppositeFootMap[i],
                          self.horizontalFootMap[i],
                          self.verticalFootMap[i]]
        return orderedIndices
            
    def getFeetThatCanMove(self, state):
        feetThatCanMove = []
        
        for i in range(4):
            supportTriangle = Triangle2D(self.getEveryFootExcept(state, i));
            if supportTriangle.isPointEnclosed(np.array([0,0])):
                feetThatCanMove.append(i)
                if len(feetThatCanMove) == 2:
                    break
        
        return feetThatCanMove
    
    def getEveryFootExcept(self, state, foot):
        feet = np.copy(state);
        feet = feet[self.getOtherFeetOrderedIndices(foot)]        
        return feet
    
    #TODO
    def getExpandedState(self, state, desiredMotion, footToMove):
        expandedState = np.array([1,  # 1
                                  1,  #foot i orig x
                                  2,  #foot i orig y
                                  3,  #foot op orig x
                                  4,  #foot op orig y
                                  5,  #foot hz orig x
                                  6,  #foot hz orig y
                                  7,  #foot vt orig x
                                  8,  #foot vt orig y
                                  9,  #foot i ideal x
                                  10, #foot i ideal y
                                  11, #desired COM x
                                  12, #desired COM y
                                  13, #orig feet op dot hz
                                  14, #orig feet op dot vt
                                  15, #orig feet hz dot vt
                                  16, #ideal feet op dot hz
                                  17, #ideal feet op dot vt
                                  18, #ideal feet hz dot vt
                                  19, #orig foot vs com dist i
                                  20, #orig foot vs com dist op
                                  21, #orig foot vs com dist hz
                                  22, #orig foot vs com dist vt
                                  23, #orig foot vs ideal foot i
                                  24, #orig foot vs ideal foot op
                                  25, #orig foot vs ideal foot hz
                                  26, #orig foot vs ideal foot vt
                                  27, #desired angle
                                  28, #ideal foot angle from desired angle op
                                  29, #ideal foot angle from desired angle hz
                                  30, #ideal foot angle from desired angle vt
                                  ])
        return expandedState
    
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
    
    
    
    
    
    
    
def testFeetThatCanMove():
    footModel = SimpleFootModel()
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
    footModel = SimpleFootModel()
    for i in range(4):
        allFeetExcepti = footModel.getEveryFootExcept(state, i)
        if (any((allFeetExcepti[:]==state[i,:]).all(1)) != False):
            print("\n----")
            print(i)
            print("error in getFeetExcept\n----")
        else:
            print(".")
            
def testGetOrderedIndices():
    footModel = SimpleFootModel()
    accurateList = [[2,3,1],[3, 2, 0], [0,1,3],[1,0,2]]
    for i in range(4):
        if (footModel.getOtherFeetOrderedIndices(i) != accurateList[i][:]):
            print("\n----")
            print(i)
            print("error in getOrderedIndices\n----")
        else:
            print(".")
        
def main():
    # testGetFeetExcept()
    # testGetOrderedIndices()
    testFeetThatCanMove()
    
    
if __name__ == "__main__":
    main()