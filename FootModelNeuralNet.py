# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 20:32:25 2021

@author: Evan Yu
"""

from FootModel import FootModel
from NeuralNetwork import NeuralNetworkSingleLayer as nnsl
import numpy as np
from DogUtil import TaskMotion, State, Command, DogModel

class NNFootModelDualStage(FootModel):
    def __init__(self, parameters=None):
        FCNNLayerDimensions = [8, 16, 4]
        CNNLayerDimensions = [8, 16, 5]
        FCNNActivationFunction = nnsl.softSignActivation
        CNNActivationFunction = nnsl.softSignActivation
        self.footChoiceNN = nnsl(FCNNLayerDimensions, FCNNActivationFunction, parameters)
        self.commandNN = nnsl(CNNLayerDimensions, CNNActivationFunction, parameters)
    
    def computeCommandFromState(self, state, desiredMotion):
        pass
    
    def setParameters(self, parameters):
        self.footChoiceNN.setParameters(parameters)
        self.commandNN.setParameters(parameters)
    
    def getNumParameters(self):
        pass
    
    def chooseFootToMove(self):
        state = 0
        
class NNFootModelSingleStage(FootModel):
    def __init__(self, parameters=None):
        LayerDimensions = [8, 36, 5]
        ActivationFunction = nnsl.softSignActivation
        self.NN = nnsl(LayerDimensions, ActivationFunction, parameters)
    
    def computeCommandFromState(self, state, desiredMotion):
        pass
    
    def setParameters(self, parameters):
        self.footChoiceNN.setParameters(parameters)
        self.commandNN.setParameters(parameters)
    
    def getNumParameters(self):
        pass
    
    def chooseFootToMove(self):
        state = 0
        
class NNFootModelSimplest(FootModel):
    def __init__(self, parameters=None):
        LayerDimensions = [12, 21, 9]
        ActivationFunction = nnsl.softSignActivation
        self.NN = nnsl(LayerDimensions, ActivationFunction, parameters)
    
    def computeCommandFromState(self, state, desiredMotion):
        inputVect = NNFootModelSimplest.generateStateVector(state, desiredMotion)
        output = self.NN.inference(inputVect)
        footChoiceOutput = output[0:4]

        bestFoot = np.argmin(footChoiceOutput)
        desiredCommand = Command(bestFoot, 
                         np.array([output[4], output[5]]),
                         np.array([output[6], output[7]]),
                         output[8])
        return desiredCommand;
    
    def generateStateVector(state, desiredMotion):
        footState = state.footState
        absoluteRotation = state.absoluteRotation
        vector = np.array([footState[0,0],
                           footState[0,1],
                           footState[1,0],
                           footState[1,1],
                           footState[2,0],
                           footState[2,1],
                           footState[3,0],
                           footState[3,1],
                           absoluteRotation,
                           desiredMotion.translationX, 
                           desiredMotion.translationY,
                           desiredMotion.relativeRotation
                           ])
        return vector
    
    def setParameters(self, parameters):
        self.NN.setParameters(parameters)
    
    def getNumParameters(self):
        return self.NN.numParameters
    
        
        