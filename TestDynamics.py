# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 17:31:49 2021

@author: Evan Yu
"""
from Dynamics import Dynamics, DynamicsFailureMessage
import numpy as np
from DogUtil import Command, State, TaskMotion, DogModel
from Polygon import Triangle2D
from VisualizerState import VisualizerState
import pyglet
from pyglet import shapes

def testFailureChecks():
    window = pyglet.window.Window(960, 540)
    pyglet.gl.glClearColor(0.6, 0.6, 0.6, 1)
    batch = pyglet.graphics.Batch()
    visualizer = VisualizerState()

    @window.event
    def on_draw():
        window.clear()
        batch.draw()
        
    dogModel = DogModel()
    startingFootState = dogModel.defaultFootState
    startingCOM = np.array([-20.,-20.])
    startingState = State(startingFootState - startingCOM, 5)
    
    startingCOMInWorld = np.array([0., 0.])
    goodCommand = Command(3, np.array([20., 20.]), np.array([10.,10.]), 10)
    badCommandBadStart = Command(1, np.array([20., 20.]), np.array([25.,25.]), 10)
    badCommandBadSome = Command(3, np.array([20., 20.]), np.array([10.,10.]), 40)
    badCommandBadEverything = Command(3, np.array([20., 20.]), np.array([25.,25.]), 50)
    
    dynamics = Dynamics(startingState)
    dynamics.applyCommand(goodCommand)
    failureMessage = dynamics.getFailureMessage()
    if (failureMessage.failureHasOccurred() == False):
        print(".")
    else:
        print("\n----")
        print(failureMessage)
        print("good command check")
        print("error in testFailureChecks\n----") 
        
    # thingsToDraw = visualizer.drawStateWithPrevious(startingState, dynamics.getCurrentState(), 
    #                                                 goodCommand,
    #                                                 startingCOMInWorld,
    #                                                 failureMessage,
    #                                                 0,
    #                                                 batch)
        
    dynamics = Dynamics(startingState)
    dynamics.applyCommand(badCommandBadStart)
    failureMessage = dynamics.getFailureMessage()
    expectedFailureMessage = DynamicsFailureMessage()
    expectedFailureMessage.setCOMIsNotContainedAtStartFailure()
    if (failureMessagesAreEqual(failureMessage, expectedFailureMessage)):
        print(".")
    else:
        print("\n----")
        print("got:")
        print(failureMessage)
        print("expected:")
        print(expectedFailureMessage)
        print("bad start command check")
        print("error in testFailureChecks\n----") 
    # thingsToDraw = visualizer.drawStateWithPrevious(startingState, dynamics.getCurrentState(), 
    #                                             badCommandBadStart,
    #                                             startingCOMInWorld,
    #                                             failureMessage,
    #                                             batch)
        
    dynamics = Dynamics(startingState)
    dynamics.applyCommand(badCommandBadSome)
    failureMessage = dynamics.getFailureMessage()
    expectedFailureMessage = DynamicsFailureMessage()
    expectedFailureMessage.setAnchoredFootPlacementsOutOfBoundsFailure(2)
    expectedFailureMessage.setSwingFootPlacementOutOfBoundsFailure()
    if (failureMessagesAreEqual(failureMessage, expectedFailureMessage)):
        print(".")
    else:
        print("\n----")
        print("got:")
        print(failureMessage)
        print("expected:")
        print(expectedFailureMessage)
        print("bad some command check")
        print("error in testFailureChecks\n----") 
        
    # thingsToDraw = visualizer.drawStateWithPrevious(startingState, dynamics.getCurrentState(), 
    #                                             badCommandBadSome,
    #                                             startingCOMInWorld,
    #                                             failureMessage,
    #                                             batch)
        
    dynamics = Dynamics(startingState)
    dynamics.applyCommand(badCommandBadEverything)
    failureMessage = dynamics.getFailureMessage()
    expectedFailureMessage = DynamicsFailureMessage()
    expectedFailureMessage.setCOMIsNotContainedAtEndFailure()
    expectedFailureMessage.setAnchoredFootPlacementsOutOfBoundsFailure(3)
    expectedFailureMessage.setSwingFootPlacementOutOfBoundsFailure()
    if (failureMessagesAreEqual(failureMessage, expectedFailureMessage)):
        print(".")
    else:
        print("\n----")
        print("got:")
        print(failureMessage)
        print("expected:")
        print(expectedFailureMessage)
        print("bad every command check")
        print("error in testFailureChecks\n----") 
    
    thingsToDraw = visualizer.drawStateWithPrevious(startingState, dynamics.getCurrentState(), 
                                                badCommandBadEverything,
                                                startingCOMInWorld,
                                                failureMessage,
                                                0,
                                                batch)

    pyglet.app.run()    
    
    
def testDynamicsBasicFunctions():
    # i should fully validate that dynamics mathematically computes the correct state
    # should also verify the state when dynamics fails
    pass

def failureMessagesAreEqual(msg1, msg2):
    return ((msg1.swingFootPlacementOutOfBounds == msg2.swingFootPlacementOutOfBounds) and
            (msg1.anchoredFootPlacementsOutOfBounds == msg2.anchoredFootPlacementsOutOfBounds) and
            (msg1.numAnchoredFootPlacementsOutOfBounds == msg2.numAnchoredFootPlacementsOutOfBounds) and
            (msg1.comIsNotContainedAtStart == msg2.comIsNotContainedAtStart) and
            (msg1.comIsNotContainedAtEnd == msg2.comIsNotContainedAtEnd))

        
def main():
    # test = Test()
    # test.drawIt()
    
    testFailureChecks()
    
    
    
if __name__ == "__main__":
    
    main()
    # pyglet.app.run()