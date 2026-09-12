#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 18:34:37 2022

@author: alberto
"""

import numpy as np
import numba 


@numba.njit("f8[:](f8,f8[:],f8[:],f8[:],f8[:])",cache=True) 
def laplacian_boundary_conditions_2D(Re,x,y,q,qbc):
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    rowsu = len(y)
    colsu = len(x) - 1
    szu = rowsu*colsu
    
    rowsv = len(y) - 1
    colsv = len(x) 
    szv = rowsv*colsv
    
    qout = np.zeros(szu+szv,dtype=np.float64)
    
    fx = 1./(dx*dx*Re)
    fy = 1./(dy*dy*Re)
    
    # --------------------------------------------------------
    # ---------- X momentum ----------------------------------
    # --------------------------------------------------------
    
    # top/bottom walls
    for i in [0,rowsu-1]:
        ibc = 0 if i == 0 else rowsu + 1
        for j in range (colsu):
            ku = i*colsu + j
            kubc = ibc*(colsu + 2) + (j+1)
            qout[ku] += fy*qbc[kubc]
            
    # left/right walls
    for j in [0,colsu-1]:
        jbc = 0 if j == 0 else colsu + 1
        for i in range (rowsu):
            ku = i*colsu + j
            kubc = (i+1)*(colsu + 2) + jbc
            qout[ku] += fx*qbc[kubc]
        
    # --------------------------------------------------------
    # ---------- Y momentum ----------------------------------
    # --------------------------------------------------------
    
    # top/bottom walls
    for i in [0,rowsv-1]:
        ibc = 0 if i == 0 else rowsv + 1
        for j in range (colsv):
            kv = i*colsv + j + szu
            kvbc = ibc*(colsv+2) + (j+1) + (colsu + 2)*(rowsu + 2)
            qout[kv] += fy*qbc[kvbc] 
    
    # right/left walls
    for j in [0,colsv-1]:
        jbc = 0 if j == 0 else colsv + 1
        for i in range (rowsv):
            kv = i*colsv + j + szu
            kvbc = (i+1)*(colsv+2) + jbc + (colsu + 2)*(rowsu + 2)
            qout[kv] += fx*qbc[kvbc] 
        
    
    return qout 


@numba.njit("f8[:](f8,f8[:],f8[:],f8[:])",cache=True) 
def divergence_boundary_conditions_2D(Re,x,y,qbc):
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    rowsu = len(y)
    colsu = len(x) - 1
    
    rowsv = len(y) - 1
    colsv = len(x) 
    
    rowsp = len(y)
    colsp = len(x)
    
    qout = np.zeros(rowsp*colsp,dtype=np.float64)
    
    fx = 1./dx
    fy = 1./dy
    
    # --------------------------------------------------------
    # ---------- X contribution ------------------------------
    # --------------------------------------------------------
    for j in [0,colsp-1]:
        
        sgn = -1 if j == 0 else 1        
        jbc = 0 if j == 0 else colsu + 1
        
        for i in range (rowsp):
            kp = i*colsp + j
            kubc = (i+1)*(colsu+2) + jbc
            qout[kp] += sgn*fx*qbc[kubc]
            
    # --------------------------------------------------------
    # ---------- Y contribution ------------------------------
    # --------------------------------------------------------
    for i in [0,rowsp-1]:
        
        sgn = -1 if i == 0 else 1
        ibc = 0 if i == 0 else rowsv + 1
        
        for j in range (colsp):
            kp = i*colsp + j
            kvbc = ibc*(colsv+2) + (j+1) + (colsu + 2)*(rowsu + 2)
            qout[kp] += sgn*fy*qbc[kvbc]
    
    
    return qout[:-1]


@numba.njit("f8[:](f8[:],f8[:],f8[:],f8[:],f8[:],f8[:])",cache=True) 
def evaluate_bilinearity_2D(x,y,q1,q2,q1bc,q2bc):
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    rowsu = len(y)
    colsu = len(x) - 1
    szu = rowsu*colsu
    
    rowsv = len(y) - 1
    colsv = len(x) 
    szv = rowsv*colsv
    
    qnl = np.zeros(szu+szv)
    # -----------------------------------------------
    # ---------- X momentum: x derivatives ----------
    # -----------------------------------------------
    for i in range (rowsu):
        for j in range (1,colsu-1):
            
            ku = i*colsu + j
            
            duu_dx = (0.5*(q1[ku]+q1[ku+1])*0.5*(q2[ku]+q2[ku+1])  \
                    - 0.5*(q1[ku]+q1[ku-1])*0.5*(q2[ku]+q2[ku-1]))/dx
                
            qnl[ku] += duu_dx
    
    for j in [0,colsu-1]:
        
        sgn = 1 if j == 0 else -1
        jbc = 0 if j == 0 else colsu + 1
        
        for i in range (rowsu):
            
            ku = i*colsu + j
            kubc = (i+1)*(colsu + 2) + jbc
            
            duu_dx = sgn*(0.5*(q1[ku]+q1[ku+sgn])*0.5*(q2[ku]+q2[ku+sgn])  \
                    - 0.5*(q1[ku]+q1bc[kubc])*0.5*(q2[ku]+q2bc[kubc]))/dx
        
            qnl[ku] += duu_dx
            
    
    # -----------------------------------------------
    # ---------- X momentum: y derivatives ----------
    # -----------------------------------------------
    for i in range (1,rowsu-1):
        for j in range (colsu):
            
            ku = i*colsu + j
            kv = i*colsv + j + szu
            
            dvu_dy = (0.5*(q1[kv]+q1[kv+1])*0.5*(q2[ku]+q2[ku+colsu]) \
                    -0.5*(q1[kv-colsv]+q1[kv-colsv+1])*0.5*(q2[ku]+q2[ku-colsu]))/dy
            
            qnl[ku] += dvu_dy
            

    for i in [0,rowsu-1]:
        
        sgn = 1 if i == 0 else -1
        off = 0 if i == 0 else -colsv
        ibcu = 0 if i == 0 else rowsu + 1
        ibcv = 0 if i == 0 else rowsv + 1
        
        
        for j in range (colsu):
            
            ku = i*colsu + j
            kv = i*colsv + j + szu
            kubc = ibcu*(colsu + 2) + (j+1)
            kvbc = ibcv*(colsv + 2) + (j+1) + (colsu + 2)*(rowsu + 2)
            
            dvu_dy = sgn*(0.5*(q1[kv+off]+q1[kv+off+1])*0.5*(q2[ku]+q2[ku+sgn*colsu]) \
                         -0.5*(q1bc[kvbc]+q1bc[kvbc+1])*0.5*(q2[ku]+q2bc[kubc]))/dy
            
            qnl[ku] += dvu_dy
            
    # -----------------------------------------------
    # ---------- Y momentum: x derivatives ----------
    # -----------------------------------------------
    for i in range (rowsv):
        for j in range (1,colsv-1):
            
            ku = i*colsu + j
            kv = i*colsv + j + szu
            
            duv_dx = (0.5*(q1[ku]+q1[ku+colsu])*0.5*(q2[kv]+q2[kv+1]) \
                    - 0.5*(q1[ku-1]+q1[ku-1+colsu])*0.5*(q2[kv]+q2[kv-1]))/dx
                
            qnl[kv] += duv_dx
                
    for j in [0,colsv-1]:
        
        sgn = 1 if j == 0 else -1
        off = 0 if j == 0 else -1
        jbcu = 0 if j == 0 else colsu + 1
        jbcv = 0 if j == 0 else colsv + 1
        
        
        for i in range (rowsv):
            
            ku = i*colsu + j
            kv = i*colsv + j + szu
            kubc = (i+1)*(colsu + 2) + jbcu
            kvbc = (i+1)*(colsv + 2) + jbcv + (colsu + 2)*(rowsu + 2)
            
            duv_dx = sgn*(0.5*(q1[ku+off]+q1[ku+off+colsu])*0.5*(q2[kv]+q2[kv+sgn]) \
                         -0.5*(q1bc[kubc]+q1bc[kubc+(colsu+2)])*0.5*(q2[kv]+q2bc[kvbc]))/dx 
            
            qnl[kv] += duv_dx
            
    
    # -----------------------------------------------
    # ---------- Y momentum: y derivatives ----------
    # -----------------------------------------------
    for i in range (1,rowsv-1):
        for j in range (colsv):
            
            ku = i*colsu + j
            kv = i*colsv + j + szu
            
            dvv_dy = (0.5*(q1[kv]+q1[kv+colsv])*0.5*(q2[kv]+q2[kv+colsv]) \
                    - 0.5*(q1[kv]+q1[kv-colsv])*0.5*(q2[kv]+q2[kv-colsv]))/dy
                
            qnl[kv] += dvv_dy
                
    for i in [0,rowsv-1]:
        sgn = 1 if i == 0 else -1
        ibc = 0 if i == 0 else rowsv + 1
        
        for j in range (colsv):
            
            ku = i*colsu + j
            kv = i*colsv + j + szu
            kvbc = ibc*(colsv + 2) + (j+1) + (colsu + 2)*(rowsu + 2)
            
            dvv_dy = sgn*(0.5*(q1[kv]+q1[kv+sgn*colsv])*0.5*(q2[kv]+q2[kv+sgn*colsv]) \
                            - 0.5*(q1[kv]+q1bc[kvbc])*0.5*(q2[kv]+q2bc[kvbc]))/dy
                    
            qnl[kv] += dvv_dy
    
    
    return qnl


