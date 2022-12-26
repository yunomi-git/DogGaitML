# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 18:55:27 2021

@author: Evan Yu
"""

import numpy as np

class NeuralNetworkSingleLayer:
    def __init__(self, layerDimensions, activationFunction, parameters):
        self.activationFunction = activationFunction
        if len(layerDimensions) != 3:
            raise ValueError('more than 3 layers in single layer perceptron')
        self.layerDimensions = layerDimensions;
        self.numLayerParameters = [(layerDimensions[0] + 1) * layerDimensions[1], 
                                   (layerDimensions[1] + 1) * layerDimensions[2]]
        self.numParameters = np.sum(self.numLayerParameters)
        if parameters is not None:
            self.setParameters(parameters)
        
    def setParameters(self, parameters):
        newNumParameters = parameters.size
        if (newNumParameters != self.numParameters):
            raise ValueError('passed incorrect number of parameters')
        layer1Parameters = np.array(parameters[0:self.numLayerParameters[0]])
        layer2Parameters = np.array(parameters[self.numLayerParameters[0]:])
        layer1Matrix = layer1Parameters.reshape((self.layerDimensions[1],
                                                 self.layerDimensions[0] + 1))
        layer2Matrix = layer2Parameters.reshape((self.layerDimensions[2],
                                                 self.layerDimensions[1] + 1))
        self.layerMatrices = [layer1Matrix, layer2Matrix]
        
    def inference(self, value):
        if (value.size != self.layerDimensions[0]):
            raise ValueError('input to inference incorrect dimensions')
        value = np.append(np.array([1]), value)
        layer1Output = self.layerMatrices[0] @ value
        layer1Output = self.activationFunction(layer1Output)
        layer1Output = np.append(np.array([1]), layer1Output)
        output = self.layerMatrices[1] @ layer1Output
        return output
    
    
    def reLUActivation(value):
        return np.maximum(value, np.zeros(value.shape))
    
    def heavisideActivation(value):
        return 1 * (value > 0)
    
    def tanhActivation(value):
        posE = np.exp(value)
        negE = np.exp(-value)
        return (posE - negE) / (posE + negE)
    
    def softSignActivation(value):
        return value / (1 + np.abs(value))