# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 19:17:41 2025

@author: diete
"""

import numpy as np


class LinearKF:
    def __init__(self, dim_x, dim_z, dim_u=0):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u
        
        
        self.F = np.zeros((self.dim_x,self.dim_x))
        self.B = np.zeros((self.dim_x,self.dim_u))
        self.P = np.zeros((self.dim_x,self.dim_x))
        self.Q = np.zeros((self.dim_x,self.dim_x))
        
        self.u = np.array([0])
        
        self.H = np.zeros((self.dim_z,self.dim_x))
        self.R = np.zeros((self.dim_z,self.dim_z))
        
        self.X0 = np.zeros(self.dim_x)
        
    def Predict(self,u):
        
        self.X0 = self.F @ self.X0 + self.B@u
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.X0,self.P
        
    def Update(self, z):
        
        y = z - self.H @ self.X0
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        
        self.X0 = self.X0 + K @ y
        self.P = (np.eye(self.dim_x)-K@self.H) @ self.P
        
        return self.X0, self.P
    