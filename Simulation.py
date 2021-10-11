# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 19:22:23 2021

@author: Evan Yu
"""

import numpy as np
import Dynamics
from FootModel import SimpleFootModel

class Simulation():
    def __init__(self, initialState, footModel, modelParameters, desiredMotion, numSteps):
        self.simulationHistory = [];
        self.currentCOMInWorldFrame = np.array([0,0]);
        self.currentState = initialState;
        self.currentRunningCost = 0
        self.footModel = footModel(modelParameters)
        self.stepsLeftToTake = numSteps
        
    def takeSimulationStep(self):
        if (not self.hasTerminated()):
            dynamics = Dynamics()
            desiredCommand = self.footModel.computeCommandFromState(self.currentState);
            dynamics.computeNextStateAfterCommand(self.currentState, desiredCommand)
            if dynamics.hasNotFailed():
                self.currentState = dynamics.getStateAfterCommand()
                self.currentCOMInWorldFrame += desiredCommand.comTranslation
                self.currentRunningCost += self.computeCostAtStep();
                simulationHistoryItem = SimulationHistoryItem(self.currentState, self.currentCOMInWorldFrame, self.currentRunningCost)
                self.simulationHistory.append(simulationHistoryItem)
                self.stepsLeftToTake -= 1
            else:
                self.currentRunningCost += self.computeCostAtStep();
                simulationHistoryItem = SimulationHistoryItem(self.currentState, self.currentCOMInWorldFrame, self.currentRunningCost)
                self.simulationHistory.append(simulationHistoryItem)
                self.terminate()
            
    def computeCostAtStep(self):
        pass
    
    def terminate(self):
        self.stepsLeftToTake = 0
        
    def hasNotTerminated(self):
        return self.stepsLeftToTake > 0
    
        
def SimulationHistoryItem():
    def __init__(self, state, comInWorldFrame, runningCost):
        self.state = state;
        self.comInWorldFrame = comInWorldFrame
        self.runningCost = runningCost