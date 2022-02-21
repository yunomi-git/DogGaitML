# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 17:22:34 2022

@author: Evan Yu
"""
import numpy as np
from DogUtil import DogModel, State, TaskMotion
import MathUtil as mu
import pickle
import os

class Dataset:
    def __init__(self, filename: str):
        with open(filename, 'rb') as handle:
            dataset = pickle.load(handle)
        self.states = dataset["states"]
        self.tasks = dataset["tasks"]    

    def getRandomBatch(self, numStates, numTasks):
        statesBatch = np.random.choice(self.states, 
                                   size=numStates, 
                                   replace=False)
        tasksBatch = np.random.choice(self.tasks, 
                                   size=self.numTasks, 
                                   replace=False)
        return statesBatch.tolist(), tasksBatch.tolist()
    
class DatasetGenerator:
    def __init__(self, dogModel: DogModel,
                 numStates, 
                 numTasks,
                 footMaxRangeFromIdeal,
                 stateAngleRange,
                 taskTransMag, taskTransAngleRange, taskRotRange):
        self.dogModel = dogModel
        self.numStates = numStates
        self.numTasks = numTasks
        self.footMaxRangeFromIdeal = footMaxRangeFromIdeal
        self.stateAngleRange = stateAngleRange
        self.taskTransMag = taskTransMag
        self.taskTransAngleRange = taskTransAngleRange
        self.taskRotRange = taskRotRange    
    
    def generateDataset(self):
        self.states = self.generateInitialStatesList()
        self.tasks = self.generateTaskMotionsList()
    
    def generateDatasetToFile(self, filename):
        self.generateDataset()
        simData = {}
        simData["states"] = self.states
        simData["tasks"] = self.tasks
        
        with open(filename, 'wb') as handle:
            pickle.dump(simData, handle)
    
    def generateInitialStatesList(self):
        statesList = []
        for i in range(self.numStates-1):
            statesList.append(self.generateRandomInitialState())
        return statesList
    
    def generateTaskMotionsList(self):
        tasksList = []
        for i in range(self.numTasks-1):
            tasksList.append(self.generateRandomTask())
        return tasksList
    
    def generateRandomInitialState(self):
        radiiSamples = np.sqrt(np.random.rand(4)) * self.footMaxRangeFromIdeal
        angleSamples = np.random.rand(4) * 2 * np.pi
        xSamples = radiiSamples * np.cos(angleSamples)
        ySamples = radiiSamples * np.sin(angleSamples)
        circleSamples = np.stack((xSamples.T, ySamples.T), axis=1)
        perturbedFootState = self.dogModel.defaultFootState + circleSamples
        
        comAngle = mu.getRandomValueOnRange(self.stateAngleRange)
        state = State(perturbedFootState, comAngle)
        
        return state
    
    def generateRandomTask(self):
        radius = np.sqrt(np.random.rand(1)[0]) * self.taskTransMag
        angle = mu.getRandomValueOnRange(self.taskTransAngleRange) * np.pi / 180.
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        rot = mu.getRandomValueOnRange(self.taskRotRange)
        
        return TaskMotion(x, y, rot)


def main():
    path = '.\\datasets\\'
    filename = 'TomatoTesting.dat'
    footMaxRangeFromIdeal = DogModel.maximumFootDistanceFromIdeal
    numStates = 15
    numTasks = 15
    stateAngleRange = [0, 0]
    taskTransMag = 40.
    taskTransAngleRange = [0., 90.]
    taskRotRange = [-10., 10.]
    generator = DatasetGenerator(DogModel(), 
                                 numStates, 
                                 numTasks, 
                                 footMaxRangeFromIdeal, 
                                 stateAngleRange, 
                                 taskTransMag, 
                                 taskTransAngleRange, 
                                 taskRotRange)
    if os.path.exists(path + filename):
        print('File Exists. Aborting')
        raise SystemExit
    else:
        if not os.path.isdir(path):
            os.mkdir(path)
        generator.generateDatasetToFile(path + filename)

        
if __name__ == "__main__":
    main()