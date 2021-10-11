# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 12:56:06 2021

@author: Evan Yu
"""

import numpy as np
from DogUtil import Command, State, DesiredMotion, DogModel

def testFeetThatCanMove():
    footState = np.array([[1.0,1.0],[1.0,-1.0],[-1.0,-1.0],[-1.0,1.0]])
    COMPositions = np.array([[0., 0.5], [0.5,0],[0,-0.5],[-0.5,0]])
    correctAnswers = [[1,2],[2,3],[0,3],[0,1]]
    for i in range(4):
        COMPosition = COMPositions[i,:]
        tempState = footState - COMPosition;
        dog = DogModel()
        dog.setState(State(tempState, 0))
        movableFeet = dog.getFeetThatCanMove()
        if (not (sorted(movableFeet) == sorted(correctAnswers[i][:]))):
            print("\n----")
            print(i)
            print("error in testFeetThatCanMove\n----")
        else:
            print(".")
    
def testGetFeetExcept():
    footState = np.array([[0,0],[1,1],[2,2],[3,3]])
    state = State(footState, 0)
    dog = DogModel()
    dog.setState(state)
    for i in range(4):
        allFeetExcepti = dog.getEveryFootExcept(i)
        if (any((allFeetExcepti[:]==footState[i,:]).all(1)) != False):
            print("\n----")
            print(i)
            print("error in getFeetExcept\n----")
        else:
            print(".")
            
def testGetOrderedIndices():
    dog = DogModel()
    accurateList = [[2,3,1],[3, 2, 0], [0,1,3],[1,0,2]]
    for i in range(4):
        if (dog.getOtherFeetOrderedIndices(i) != accurateList[i][:]):
            print("\n----")
            print(i)
            print("error in getOrderedIndices\n----")
        else:
            print(".")
            
def main():
    testGetFeetExcept()
    testGetOrderedIndices()
    testFeetThatCanMove()
    
    
if __name__ == "__main__":
    main()