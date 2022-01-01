# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 20:52:54 2021

@author: Evan Yu
"""

from DebugMessage import DebugMessage

parentMessage = DebugMessage()
parentMessage.appendMessage("item1", 2)
parentMessage.appendMessage("item2", 4.0)

childMessage = DebugMessage()
childMessage.appendMessage("child1", 3.4)
childMessage.appendMessage("child2", 5.4)

parentMessage.appendMessage("item3", childMessage)
parentMessage.combineDebugMessage(childMessage)


print(str(parentMessage))
