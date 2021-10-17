# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 19:34:21 2021

@author: Evan Yu
"""

from VisualizerState import VisualizerState
from Simulation import SimulationHistoryItem
import pyglet

class VisualizerSimulation:
    def __init__(self, simulationHistory):
        self.simulationHistory = simulationHistory
        self.numItems = len(self.simulationHistory)
        self.index = 0
        self.visualizerState = VisualizerState()
    
    def increment(self):
        if self.index < self.numItems-1:
            self.index += 1

            
    def decrement(self):
        if self.index > 0:
            self.index -= 1
            
    def visualizeCurrentItem(self, batch):
        currentHistoryItem = self.simulationHistory[self.index]
        command = currentHistoryItem.command
        currentState = currentHistoryItem.state
        failureMessage = currentHistoryItem.failureMessage
        cost = currentHistoryItem.runningCost
        
        if self.index == 0:
            originalCOMInWorld = currentHistoryItem.comInWorldFrame
            
            thingsToDraw = self.visualizerState.drawSingleState(currentState, 
                                                                  originalCOMInWorld, 
                                                                  cost,
                                                                  batch)
            

        else:
            lastHistoryItem = self.simulationHistory[self.index - 1]
            originalState = lastHistoryItem.state
            originalCOMInWorld = lastHistoryItem.comInWorldFrame

        
            thingsToDraw = self.visualizerState.drawStateWithPrevious(originalState, 
                                                                  currentState, 
                                                                  command, 
                                                                  originalCOMInWorld, 
                                                                  failureMessage, 
                                                                  cost,
                                                                  batch)
        label = pyglet.text.Label("Step: " + str(self.index),
                      color=(255,255,255,255),
                      font_name='Times New Roman',
                      font_size=15,
                      x=0, y=25,
                      anchor_x='left', anchor_y='bottom', batch=batch)
        thingsToDraw.append(label)
        
        return thingsToDraw
     