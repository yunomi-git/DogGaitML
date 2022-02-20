# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 19:12:09 2022

@author: Evan Yu
"""

from FootModel import FootModel
import MathUtil as mu
from DogUtil import TaskMotion, Command, State, DogModel
from StateSymmetry import StateRotTaskSymmetry
import numpy as np
import random
from datetime import datetime



class  SymmetricFootModel(FootModel):
    def computeCommandFromState(self, state, desiredMotion):
        forwardDir = mu.getUnitVectorFromAngle(state.absoluteRotation)
        leftDir = mu.getUnitVectorFromAngle(state.absoluteRotation + 90.)
        
        commandedTranslation = desiredMotion.translation()
        commandedRotation = desiredMotion.relativeRotation
        
        commandingForward = np.dot(forwardDir, commandedTranslation) > 0
        commandingLeft = np.dot(leftDir, commandedTranslation) > 0
        
        footToMoveSet = [3, 0] if commandingForward else [2, 1]
        footToMove = footToMoveSet[0] if commandingLeft else footToMoveSet[1]
        
        desiredCommand = Command(footToMove, 
                 commandedTranslation,
                 commandedTranslation,
                 commandedRotation)
        return desiredCommand
    
    def setParameters(self, parameters):
        pass
    
    def getNumParameters(self):
        return 0
    
def doesSymmetryProduceSameCommandAsOriginal(state, task):
    symmetryHandler = StateRotTaskSymmetry()
    symState, symTask = symmetryHandler.simplifyInputFrameAndSave(state, task)
    
    footModel = SymmetricFootModel()
    
    # this is original frame
    command = footModel.computeCommandFromState(state, task)
    # this is the frame 0, mirrored
    symCommand = footModel.computeCommandFromState(symState, symTask)
    revertedCommand = symmetryHandler.returnOutputToOriginalFrame(symCommand)
    
    passed = command.equals(revertedCommand)
    
    # if not passed:
    #     print(state.absoluteRotation)
    #     print(symState.absoluteRotation)
    #     print(symTask.translation())
    #     print(task.translation())
    #     print(command)
    #     print(revertedCommand)
    
    return passed
    
def testCases():
    testCase(TaskMotion(1., 2., 30), TaskMotion(-2., 3., 12.), 20.)
    testCase(TaskMotion(1., -2., -30), TaskMotion(-2., 3., 12.), -20.)
    random.seed(datetime.now())
    for i in range(20):
        testCase(TaskMotion(random.randrange(-20, 20, 1) * 1.01 + 0.001,
                            random.randrange(-20, 20, 1) * 1.01 + 0.001,
                            random.randrange(-200, 200, 1) * 1.01 + 0.001), 
                 TaskMotion(random.randrange(-20, 20, 1) * 1.01 + 0.001,
                            random.randrange(-20, 20, 1) * 1.01 + 0.001,
                            random.randrange(-200, 200, 1) * 1.01 + 0.001), 
                            random.randrange(-200, 200, 1) * 1.01 + 0.001)
        
def testCase(taskMotion, footStateTaskMotion, stateComAngle):
    dogModel = DogModel()
    footState = dogModel.getIdealFootStateFromOriginalCom(footStateTaskMotion)
    state = State(footState, stateComAngle)
    print(doesSymmetryProduceSameCommandAsOriginal(state, taskMotion))
    
    
testCases()