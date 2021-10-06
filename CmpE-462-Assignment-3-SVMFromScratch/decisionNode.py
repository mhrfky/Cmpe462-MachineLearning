# -*- coding: utf-8 -*-
"""
Created on Sun May 30 13:54:25 2021

@author: mhrfk
"""

class decision_node:
    def __init__(self,threshold,column,val):
        self.left = None
        self.right = None
        self.column = column
        self.val = val
        self.threshold = threshold