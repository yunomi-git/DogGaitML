# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 19:36:12 2021

@author: Evan Yu
"""

import numpy as np
from NeuralNetwork import NeuralNetworkNLayer as nn

def testInferenceSimple():
    layerDimensions = [2,3,2]
    activationFunction = nn.reLUActivation
    parameters = np.array([1.,2.,3.,
                           -1.,-2.,-3.,
                           -1.,1.,-1.,
                           1.,2.,-2.,-1.,
                           -1.,-1.,2.,2.])
    net = nn(layerDimensions, activationFunction, parameters)
    
    nnInput = np.array([2., -1.])
    expectedOutput = np.array([3, 1])
    actualOutput = net.inference(nnInput)
    if ((np.equal(expectedOutput, actualOutput)).all()):
        print(".")
    else:
        print("\n----")
        print("received: ")
        print(actualOutput)
        print("expected: ")
        print(expectedOutput)
        print("error in basic nn inference\n----") 
        
    parameters = np.array([3.,-1.,-3.,
                           1.,-4.,0.,
                           -2.,1.,1.,
                           3.,1.,1.,-2.,
                           -3.,1.,2.,-1.])
    net.setParameters(parameters)
    nnInput = np.array([2., -1.])
    expectedOutput = np.array([7, 1])
    actualOutput = net.inference(nnInput)
    if ((np.equal(expectedOutput, actualOutput)).all()):
        print(".")
    else:
        print("\n----")
        print("received: ")
        print(actualOutput)
        print("expected: ")
        print(expectedOutput)
        print("error in new parameters nn inference\n----") 

def testActivations():
    reluInput = np.array([1.0, 5.0, 0.0, -1.0])
    expectedOutput = np.array([1.0, 5.0, 0.0, 0.0])
    actualOutput = nn.reLUActivation(reluInput)
    if ((np.equal(expectedOutput, actualOutput)).all()):
        print(".")
    else:
        print("\n----")
        print("error in relu activation\n----") 
        
    hsInput = np.array([1.0, 5.0, 0.0, -1.0, -5.0, -0.5])
    expectedOutput = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    actualOutput = nn.heavisideActivation(hsInput)
    if (np.equal(expectedOutput, actualOutput)).all():
        print(".")
    else:
        print("\n----")
        print("error in heaviside activation\n----") 
        
    tanhInput = np.array([1.0, 5.0, 0.0, -1.0, -5.0, -0.5])
    expectedOutput = np.array([0.7615941559557648881195, 
                               0.999909204262595131211, 
                               0.0, 
                               -0.7615941559557648881195, 
                               -0.999909204262595131211, 
                               -0.4621171572600097585023])
    actualOutput = nn.tanhActivation(tanhInput)
    if (np.isclose(expectedOutput, actualOutput)).all():
        print(".")
    else:
        print("\n----")
        print("error in tanh activation\n----") 
        
    ssInput = np.array([1.0, 5.0, 0.0, -1.0, -5.0, -0.5])
    expectedOutput = np.array([0.5, 
                               0.833333, 
                               0.0, 
                               -0.5, 
                               -0.833333, 
                               -0.333333])
    actualOutput = nn.softSignActivation(ssInput)
    if (np.isclose(expectedOutput, actualOutput)).all():
        print(".")
    else:
        print("\n----")
        print("error in softsign activation\n----") 

def main():
    # test = Test()
    # test.drawIt()
    
    testInferenceSimple()
    testActivations()
    
    
    
if __name__ == "__main__":
    main()
