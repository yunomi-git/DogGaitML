# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 11:15:02 2021

@author: Evan Yu
"""

import numpy as np
from DogUtil import Command, State, DesiredMotion
import FootModel as fmodel
import time

def testGetExpandedStates():
    footModel = fmodel.SimpleFootModel(np.ones(6*31))
    defaultFootState = np.array([[ 112.5 ,  -73.97],
                                   [-112.5 ,  -73.97],
                                   [-112.5 ,   73.97],
                                   [ 112.5 ,   73.97]])
    rotation = 5
    originalState = State(defaultFootState, rotation)
    footToMove = 0
    desiredMotion = DesiredMotion(10, 20, 25)
    expandedState = footModel.getExpandedState(originalState, desiredMotion, footToMove)
    correctExpandedState = np.array([
            1,  # 1
            112.5,  #foot i orig x
            -73.97,  #foot i orig y
            -112.5,  #foot op orig x
            73.97,  #foot op orig y
            112.5,  #foot hz orig x
            73.97,  #foot hz orig y
            -112.5,  #foot vt orig x
            -73.97,  #foot vt orig y
            144.41285793,  #foot i ideal x
            12.19010088, #foot i ideal y
            10, #desired COM x
            20, #desired COM y
            -7184.6891000000005, #orig com,feet op dot hz
            7184.6891000000005, #orig com,feet op dot vt
            -18127.8109, #orig com,feet hz dot vt
            -9643.4891, #ideal com orig feet op dot hz
            9934.6891, #ideal com orig feet op dot vt
            -17627.8109, #ideal com orig feet hz dot vt
            134.63955919, #orig foot vs com dist i
            134.63955919, #orig foot vs com dist op
            134.63955919, #orig foot vs com dist hz
            134.63955919, #orig foot vs com dist vt
            91.88032153, #orig foot vs ideal foot dist i
            47.67254029, #orig foot vs ideal foot dist op
            78.54798161, #orig foot vs ideal foot dist hz
            67.415719, #orig foot vs ideal foot dist vt
            25, #desired angle
            126.22308128821268, #orig foot-desired com angle from desired angle op
            -2.2315538083380275, #orig foot-desired com angle from desired angle hz
            -172.50811304048392 #orig foot-desired com angle from desired angle vt
            ])
    if (not np.all(np.isclose(correctExpandedState, expandedState))):
        print("\n----")
        print(expandedState)
        print(np.isclose(correctExpandedState, expandedState))
        print("error in testGetExpandedStates\n----")
    else:
        print(".")

def testGetBestOutput():
    outputList = [];
    bestOutput = np.array([1., 4., 3., 5., 2., 4.])
    outputList.append(np.array([0.5, 0., 5., 5., 3., 4.]))
    outputList.append(bestOutput)
    outputList.append(np.array([-1., 4., 3., 5., 2., 4.]))
    
    footModel = fmodel.SimpleFootModel(np.ones(6*31))
    
    if (not (footModel.getBestOutputIndexFromList(outputList) == 1)):
        print("\n----")
        print(footModel.getBestOutputIndexFromList(outputList))
        print("error in testGetBestOutput\n----")
    else:
        print(".")
            
    
def main():
    start_time = time.time()
    testGetExpandedStates()
    testGetBestOutput()
    print((time.time() - start_time))
    
    
if __name__ == "__main__":
    main()