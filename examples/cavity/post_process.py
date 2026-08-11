#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 19:21:27 2022

@author: alberto
"""

import numpy as np 


def output_fields(flow,q):
    
    xu, yu = np.meshgrid(0.5*(flow.x[0:-1] + flow.x[1:]),flow.y)
    u = q[0:flow.szu].reshape((flow.rowsu,flow.colsu))
    
    xv, yv = np.meshgrid(flow.x,0.5*(flow.y[0:-1] + flow.y[1:]))
    v = q[flow.szu:flow.szu+flow.szv].reshape((flow.rowsv,flow.colsv))
    
    xvort, yvort = np.meshgrid(0.5*(flow.x[0:-1] + flow.x[1:]),\
                               0.5*(flow.y[0:-1] + flow.y[1:]))
    colsvort = len(flow.x)-1
    rowsvort = len(flow.y)-1
    vort = np.zeros((rowsvort,colsvort))
    for i in range (len(flow.y)-1):
        for j in range (len(flow.x)-1):
            ku = i*flow.colsu + j
            kv = i*flow.colsv + j + flow.szu
            vort[i,j] = (q[kv+1] - q[kv])/flow.dx - (q[ku+flow.colsu] - q[ku])/flow.dy
    
    
    if len(q) > flow.szu + flow.szv:
        xw, yw = np.meshgrid(flow.x,flow.y)
        w = q[flow.szu+flow.szv:].reshape((flow.rowsw,flow.colsw))
        
        X = [xu,xv,xw]
        Y = [yu,yv,yw]
        fields = [u,v,w]
        
        
    else:
        
        X = [xu,xv,xvort]
        Y = [yu,yv,yvort]
        fields = [u,v,vort]
        
    
    return X, Y, fields
            

def compute_energy(data,q0,tsave):
    
    N = data.shape[0]
    energy = np.zeros(len(tsave))
    for k in range (len(tsave)):
        energy[k] = np.dot(data[:,k]-q0,data[:,k]-q0)/N
        
    return energy


