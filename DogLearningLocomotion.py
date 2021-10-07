# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 21:12:37 2021

@author: Evan Yu
"""
import numpy as np;
import Optimizer.py

def runOptimizer(optimizer, parameters):
    optimizer.setParameters();
    optimizer.getGradient();
    optimizer.stepInDirection();
    

def main():
    initialValue = np.zeros(3);
    optimizer = Optimizer(initialValue, costEvaluator, optimizationParameters)
    runOptimizer()

if __name__ == "__main__":
    main()