# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 18:51:00 2021

@author: Evan Yu
"""
import numpy as np
from SimulationDataStructures import Command, State, DesiredMotion, getRotationMatrix

class Dynamics():
    def __init__(self):
        self.hasNotFailed = True
        self.state = None
    
    def computeStateAfterCommand(self, state, command):
        if (self.hasNotFailed):
            footToMove = command.footToMove
            desiredTranslation = command.comTranslation;
            nextFootState = state.footState - desiredTranslation #edit
            nextRotation = self.currentRotation + command.comRotation
            nextState = State(nextFootState, nextRotation)
            if (self.isStateTransitionValid(footToMove, state.footState, nextFootState, nextRotation)):
                self.comInWorldFrame += desiredTranslation
                self.state = State(nextState, nextRotation)
            else:
                self.hasNotFailed = False
            
    def isStateTransitionValid(self, footToMove, originalFootState, finalFootState, finalRotation):
        # each foot needs to be within x distance from default (at ideal)
        # com must be within poylygon at beginning of motion
        # com must be within polygon at end of motion
        return False
    
    def hasNotFailed(self):
        return self.hasNotFailed
    
    def getStateAfterCommand(self):
        return self.state
            