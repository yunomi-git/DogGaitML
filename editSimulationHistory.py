# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 13:41:54 2021

@author: Evan Yu
"""

import pickle

from FootModelNeuralNet import NNFootModelSimplest

subFolderName = "GA_LinearModel"
prefix = "12-2-2021_GA_LinearModel"
suffix = "_0"

simulationName = prefix + suffix
path = ".\\data\\" + subFolderName + "\\"
filename =  path + simulationName + '.pickle'

with open(filename, 'rb') as handle:
    simData = pickle.load(handle)
    
footModel = SimpleFootModel()

simData["footModel"] = footModel
with open(filename, 'wb') as handle:
    pickle.dump(simData, handle)