# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 20:06:09 2022

@author: Evan Yu
"""

class SimulationHistoryItem():
    def __init__(self, state, comInWorldFrame, runningCost, command, failureMessage):
        self.state = state;
        self.comInWorldFrame = comInWorldFrame
        self.runningCost = runningCost
        self.command = command
        self.failureMessage = failureMessage

