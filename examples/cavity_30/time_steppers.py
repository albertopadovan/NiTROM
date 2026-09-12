#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 12:02:38 2022

@author: alberto
"""
import numpy as np
import numba_operators as numbaops
import time as tlib


def solver_2D(flow,lops,qic,time,n,bc_coefs_,*argv,**kwargs):
    
    bc_coefs = bc_coefs_.copy()
    
    flag = 0
    if len(argv) > 0:
        which_coefs = argv[0]
        scipy_intrp = argv[1]
        forcn_period = argv[2]
        flag = 1
    
    vol_forcing = kwargs.get('vol_forcing',None)
    flag_vf = 1
    try:
        if vol_forcing == None:     
            vol_forcing = np.zeros(len(qic))
            flag_vf = 0
    except:
        vol_forcing = vol_forcing.copy()
    
    q = qic.copy()
    tsave = time[::n]
    data = np.zeros((len(q),len(tsave)))
    data[:,0] =  q
    
    # Initialize vectors for time stepping
    qrhs = np.zeros(len(q))
    qfwd = qrhs.copy()
    qkm1 = qrhs.copy()
    qs = qrhs.copy()
    
    
    idx_save, ttot = 1, 0
    for k in range (1,len(time)):
        
        t0 = tlib.perf_counter()
        
        
        if flag == 1:
            for (count,i) in enumerate (which_coefs):
                bc_coefs[i] = scipy_intrp[count](np.mod(time[k-1],forcn_period[count]))
                
        if flag_vf == 1:
            
            val = 0
            for (count,i) in enumerate (which_coefs):
                val += scipy_intrp[count](np.mod(time[k-1],forcn_period[count]))
            
            ff = vol_forcing*val
        else:
            ff = vol_forcing.copy()
        
        
        qbc = flow.populate_boundary_conditions(q,*bc_coefs)
        qlap_bc = numbaops.laplacian_boundary_conditions_2D(flow.Re,flow.x,flow.y,q,qbc)
        qdiv_bc = numbaops.divergence_boundary_conditions_2D(flow.Re,flow.x,flow.y,qbc)
        
        qrhs = qlap_bc - numbaops.evaluate_bilinearity_2D(flow.x,flow.y,q,q,qbc,qbc) + lops.L.dot(q) + ff
        
        
        if k == 0: 
            qfwd[:] = qrhs
            qkm1[:] = qrhs
        else:
            qfwd[:] = 1.5*qrhs - 0.5*qkm1
            qkm1[:] = qrhs
        
        qs[:] = lops.LL.dot(lops.LR.dot(q) + qfwd)
        q[:] = qs - lops.G.dot(lops.luP.solve(lops.D.dot(qs) - qdiv_bc))

        
        t1 = tlib.perf_counter() - t0
        ttot += t1
        # print("Time step: %d/%d,\t Iter time: %1.3e [s],\t Tot. time: %1.3e [s]"%(k,len(time),t1,ttot))
        
        if np.mod(k,n) == 0: 
            data[:,idx_save] = q
            idx_save += 1
            
            if np.isnan(np.linalg.norm(q)):
                raise ValueError ("Code blew up.")
            
            
    return data, tsave
