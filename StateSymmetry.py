# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 13:00:38 2022

@author: Evan Yu
"""
from abc import ABC, abstractmethod
import numpy as np
import MathUtil as mu
from DogUtil import TaskMotion, State, Command


class StateSymmetry(ABC):
    @abstractmethod
    def simplifyInputFrameAndSave(self, state, task):
        pass
    
    @abstractmethod
    def returnOutputToOriginalFrame(self, command):
        pass
    
class StateRotTaskSymmetry(ABC):
    MYFootMap = {0:3, 1:2, 2:1, 3:0}
    MXFootMap = {0:1, 1:1, 2:3, 3:2}
    
    def __init__(self):
        self.inputSaved = False
    
    def simplifyInputFrameAndSave(self, origState, origTask):
        state = origState.copy()
        task = origTask.copy()
        
        # rotate -> mirrorY -> mirrorX
        self.frameRotation = state.absoluteRotation
        StateRotTaskSymmetry.simplifyRotation(state, task, self.frameRotation)
        
        self.mirrorY = (task.translationY < 0)
        if self.mirrorY:
            StateRotTaskSymmetry.simplifyMirrorY(state, task)
            
        self.mirrorX = (task.translationX < 0)
        if self.mirrorX:
            StateRotTaskSymmetry.simplifyMirrorX(state, task)
            
        self.inputSaved = True
        
        return state, task
    
    def simplifyRotation(state, task, frameRotation):
        rotationMatrix = mu.getRotationMatrix(-frameRotation)
        
        state.footState = (rotationMatrix @ state.footState.T).T
        state.absoluteRotation = 0.0
        task.setTranslation((rotationMatrix @ task.translation().T).T)
        # task.relativeRotation -= frameRotation
        # if (task.relativeRotation < -180.):
        #     task.relativeRotation += 360.0
        
    def simplifyMirrorY(state, task):
        state.footState = np.array([state.footState[StateRotTaskSymmetry.MYFootMap[0]],
                                  state.footState[StateRotTaskSymmetry.MYFootMap[1]],
                                  state.footState[StateRotTaskSymmetry.MYFootMap[2]],
                                  state.footState[StateRotTaskSymmetry.MYFootMap[3]]])
        task.translationY *= -1.
        task.relativeRotation *= -1
        
    def simplifyMirrorX(state, task):
        state.footState = np.array([state.footState[StateRotTaskSymmetry.MXFootMap[0]],
                                  state.footState[StateRotTaskSymmetry.MXFootMap[1]],
                                  state.footState[StateRotTaskSymmetry.MXFootMap[2]],
                                  state.footState[StateRotTaskSymmetry.MXFootMap[3]]])
        task.translationX *= -1.
        task.relativeRotation *= -1
            
    def returnOutputToOriginalFrame(self, originalCommand):
        if not self.inputSaved:
            raise AssertionError("returning output before input has been applied")
            
        command = originalCommand.copy()
        
        # mirrorX -> mirrorY -> rotate
        if self.mirrorX:
            StateRotTaskSymmetry.revertMirrorX(command)
        if self.mirrorY:
            StateRotTaskSymmetry.revertMirrorY(command)
        
        StateRotTaskSymmetry.revertRotation(command, self.frameRotation)
            
        return command

            
    def revertMirrorX(command):
        command.footToMove = StateRotTaskSymmetry.MXFootMap[command.footToMove]
        command.footTranslation[0] *= -1
        command.comTranslation[0] *= -1
        command.comRelativeRotation *= -1
        
    def revertMirrorY(command):
        command.footToMove = StateRotTaskSymmetry.MYFootMap[command.footToMove]
        command.footTranslation[1] *= -1
        command.comTranslation[1] *= -1
        command.comRelativeRotation *= -1
        
    def revertRotation(command, frameRotation):
        rotationMatrix = mu.getRotationMatrix(frameRotation)
        
        command.footTranslation = (rotationMatrix @ command.footTranslation.T).T
        command.comTranslation = (rotationMatrix @ command.comTranslation.T).T
        # command.comRelativeRotation += frameRotation
        # if command.comRelativeRotation > 180.:
        #     command.comRelativeRotation -= 360.
        

        
        