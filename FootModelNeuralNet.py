# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 20:32:25 2021

@author: Evan Yu
"""

from FootModel import FootModel
from NeuralNetwork import NeuralNetworkSingleLayer as nnsl

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
        
        