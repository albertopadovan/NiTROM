#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 16:57:06 2022

@author: alberto
"""


import scipy.sparse as ssparse
import numpy as np


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# ---------- Operators for 2D simulations ----------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
def augment_pressure_laplacian(flow,L):
    
    val = 1./np.sqrt(flow.szp)
    rows = np.arange(0,flow.szp,1)
    cols = flow.szp*np.ones(flow.szp)
    data = val*np.ones(flow.szp)
    
    Mat1 = ssparse.csr_matrix((data,(rows,cols)), shape=(flow.szp+1,flow.szp+1))
    Mat2 = ssparse.csr_matrix((data,(cols,rows)), shape=(flow.szp+1,flow.szp+1))
    Laug = L + Mat1 + Mat2
    
    return Laug 


def gradient_2D(flow):
    
    rows = []
    cols = []
    data = []
    
    # --------------------------------------------------------
    # ---------- X momentum ----------------------------------
    # --------------------------------------------------------
    
    for i in range (0,flow.rowsu):
        for j in range (0,flow.colsu):
            ku = i*flow.colsu + j
            kp = i*flow.colsp + j
            
            if i == flow.rowsu-1 and j == flow.colsu-1:
                rows.extend([ku])
                cols.extend([kp])
                data.extend([-1/flow.dx])
            else:
                rows.extend([ku,ku])
                cols.extend([kp,kp+1])
                data.extend([-1/flow.dx,1/flow.dx])
                
    
    # --------------------------------------------------------
    # ---------- Y momentum ----------------------------------
    # --------------------------------------------------------
    
    for i in range (0,flow.rowsv):
        for j in range (0,flow.colsv):
            kv = i*flow.colsv + j + flow.szu
            kp = i*flow.colsp + j
            
            if i == flow.rowsv-1 and j == flow.colsv-1:
                rows.extend([kv])
                cols.extend([kp])
                data.extend([-1/flow.dy])
            else:
                rows.extend([kv,kv])
                cols.extend([kp,kp+flow.colsp])
                data.extend([-1/flow.dy,1/flow.dy])
    
    G = ssparse.csr_matrix((data,(rows,cols)), shape=(flow.szu+flow.szv,flow.szp-1))
    G.eliminate_zeros()
    G.sort_indices()
                
    return G


def laplacian_2D(flow):
    
    rows = []
    cols = []
    data = []
    
    val_jpm1 = 1.0/(flow.dx*flow.dx*flow.Re)
    val_ipm1 = 1.0/(flow.dy*flow.dy*flow.Re)
    val_ij   = -2.0/(flow.dx*flow.dx*flow.Re) - 2.0/(flow.dy*flow.dy*flow.Re)
    
    for M in [0,1]:
        
        if M == 0:                  # X Momentum
            rowsM = flow.rowsu
            colsM = flow.colsu
        else:                       # Y Momentum
            rowsM = flow.rowsv
            colsM = flow.colsv
    
        for i in range (1,rowsM-1):
            for j in range (1,colsM-1):
                k = i*colsM + j + M*flow.szu
                
                rows.extend([k,k,k,k,k])
                cols.extend([k-colsM,k-1,k,k+1,k+colsM])
                data.extend([val_ipm1,val_jpm1,val_ij,val_jpm1,val_ipm1])
                
        for i in [0,rowsM-1]:
            
            if i == 0:  sgni = 1
            else:       sgni = -1
            
            for j in range (1,colsM-1):
                k = i*colsM + j + M*flow.szu
                
                rows.extend([k,k,k,k])
                cols.extend([k+sgni*colsM,k-1,k,k+1])
                data.extend([val_ipm1,val_jpm1,val_ij,val_jpm1])
                
                
        for j in [0,colsM-1]:
            
            if j == 0:  sgnj = 1
            else:       sgnj = -1
            
            for i in range (1,rowsM-1):
                k = i*colsM + j + M*flow.szu
                
                rows.extend([k,k,k,k])
                cols.extend([k-colsM,k,k+colsM,k+sgnj])
                data.extend([val_ipm1,val_ij,val_ipm1,val_jpm1])
                
        
        for i in [0,rowsM-1]:
            if i == 0:  sgni = 1
            else:       sgni = -1
            
            for j in [0,colsM-1]:
                if j == 0:  sgnj = 1
                else:       sgnj = -1
                
                k = i*colsM + j + M*flow.szu
                
                rows.extend([k,k,k])
                cols.extend([k,k+sgnj,k+sgni*colsM])
                data.extend([val_ij,val_jpm1,val_ipm1])
                
                
    
    L = ssparse.csr_matrix((data,(rows,cols)), shape=(flow.szu+flow.szv,flow.szu+flow.szv))
    L.eliminate_zeros()
    L.sort_indices()
    
    return L


            
            