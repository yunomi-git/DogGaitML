# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 01:17:36 2021

@author: Evan Yu
"""

from Simulation import Simulation
from FootModel import FootModel
from DogUtil import DogModel, Command, TaskMotion, State
import numpy as np
import time
from VisualizerSimulation import VisualizerSimulation
from VisualizerState import VisualizerState
import pyglet
from pyglet import shapes
from pyglet.window import mouse
from pyglet.graphics.vertexdomain import VertexList


class TestFootModel(FootModel):
    def __init__(self, parameters=None):
        self.parameters = parameters
    
    def computeCommandFromState(self, state, desiredTaskMotion):
        dogModel = DogModel(state)
        feetThatCanMove = dogModel.getFeetThatCanMove()
        footDistances = dogModel.getPreMotionFootDistancesFromIdeal(desiredTaskMotion)
        
        # choose the foot with the largest distance
        if (footDistances[feetThatCanMove[0]] <= footDistances[feetThatCanMove[1]]):
            footToMove = feetThatCanMove[0]
        else:
            footToMove = feetThatCanMove[1]
                        
        footScalingDistance = self.parameters[0]
        comScalingDistance = self.parameters[1]
        
        desiredTranslation = np.array([desiredTaskMotion.translationX, desiredTaskMotion.translationY])
        
        command = Command(footToMove, 
                          footTranslation=desiredTranslation * footScalingDistance, 
                          comTranslation=desiredTranslation * comScalingDistance, 
                          comRelativeRotation = 0)
        return command
    
    def getNumParameters(self):
        return 2
    
    def setParameters(self, parameters):
        self.parameters = parameters

def testSingleSimulation():
    footModel = TestFootModel()
    
    costWeights = np.array([1.,1.,
                            1.,1.,
                            1.,
                            1.,
                            1.])
    desiredTaskMotion = TaskMotion(20.0, 0.0, 0.0)
    numSteps = 10
    dogModel = DogModel()
    initialCOM = np.array([-20.0, 5])
    initialFootState = dogModel.defaultFootState - initialCOM
    initialState = State(initialFootState, 0.)
    simulation = Simulation(initialState, footModel, desiredTaskMotion, numSteps, costWeights)
    
    parameters = np.array([1.0, 0.5])
    cost = simulation.getCost(parameters)
    
    window = pyglet.window.Window(960, 540)
    pyglet.gl.glClearColor(0.6, 0.6, 0.6, 1)
    batch = pyglet.graphics.Batch()
    

        
    visualizer = VisualizerSimulation(simulation.getSimulationHistory())
    thingsToDraw = visualizer.visualizeCurrentItem(batch)
    a = [thingsToDraw]
    
    @window.event
    def on_draw():
        window.clear()
        batch.draw()
        
    @window.event
    def on_mouse_press(x,y, button, modifier):
        if button == mouse.LEFT:
            for item in a[0]:
                item.delete()
            visualizer.increment()
            a[0] = visualizer.visualizeCurrentItem(batch)
            window.clear()
            batch.draw()
     
        elif button == mouse.RIGHT:
            for item in a[0]:
                item.delete()
            visualizer.decrement()
            a[0] = visualizer.visualizeCurrentItem(batch)
            window.clear()
            batch.draw()
    
    pyglet.app.run()  
    
    
    
    
        
def main():
    start_time = time.time()
    testSingleSimulation()
    print((time.time() - start_time))
    
    
if __name__ == "__main__":
    main()