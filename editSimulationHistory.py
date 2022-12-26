# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 13:41:54 2021

@author: Evan Yu
"""

import pickle

from model.FootModelNeuralNet import NNFootModelSimplest
from model.FootModel import SimpleFootModel


subFolderName = "GA_NNSimpleModel"
prefix = "12-3-2021_GA_NNSimpleModelCurriculum"
suffix = "_1"

simulationName = prefix + suffix
path = ".\\data\\" + subFolderName + "\\"
filename =  path + simulationName + '.pickle'

with open(filename, 'rb') as handle:
    simData = pickle.load(handle)
    
footModel = NNFootModelSimplest()

simData["footModel"] = footModel
with open(filename, 'wb') as handle:
    pickle.dump(simData, handle)