#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 15:38:17 2022

@author: alberto
"""
import numpy as np
import torch
import scipy.sparse
import linear_operators as lops
from scipy.sparse.linalg import splu
import scipy.linalg as sciplin
import numba_operators as numbaops


class flow_class:
    
    def __init__(self,Lx,Ly,Nx,Ny,Re):
        
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        
        self.dx = self.Lx/self.Nx
        self.dy = self.Ly/self.Ny
        self.x = self.dx*np.arange(0,self.Nx,1) + self.dx/2 - 0
        self.y = self.dy*np.arange(0,self.Ny,1) + self.dy/2 - 0
        
        self.Re = Re
        
        self.rowsu = self.Ny
        self.colsu = self.Nx - 1
        self.rowsv = self.Ny - 1
        self.colsv = self.Nx 
        self.rowsp = self.Ny
        self.colsp = self.Nx
        
        self.szu = self.colsu*self.rowsu
        self.szv = self.colsv*self.rowsv
        self.szp = self.colsp*self.rowsp
        
        self.q_sbf = np.zeros(self.szu+self.szv)

        self.ul = np.zeros(self.rowsu) + 0
        self.ur = np.zeros(self.rowsu) + 0
        self.ub = np.zeros(self.colsu) + 1
        self.ut = np.zeros(self.colsu) + 0
        
        self.vl = np.zeros(self.rowsv) + 0
        self.vr = np.zeros(self.rowsv) + 0
        self.vb = np.zeros(self.colsv) + 0
        self.vt = np.zeros(self.colsv) + 0
        
        self.Ubc = np.zeros((self.rowsu+2,self.colsu+2))
        self.Vbc = np.zeros((self.rowsv+2,self.colsv+2))
        

    def populate_boundary_conditions(self,q,cul,cur,cub,cut,cvl,cvr,cvb,cvt):
        
        U = q[:self.szu].reshape((self.rowsu,self.colsu))
        
        self.Ubc *= 0
        
        self.Ubc[1:-1,0] += self.ul*cul
        self.Ubc[1:-1,-1] += self.ur*cur
        self.Ubc[0,1:-1] += 2*self.ub*cub - U[0,]
        self.Ubc[-1,1:-1] += 2*self.ut*cut - U[-1,]
        
        
        V = q[self.szu:].reshape((self.rowsv,self.colsv))
        self.Vbc *= 0
        
        self.Vbc[1:-1,0] += 2*self.vl*cvl - V[:,0]
        self.Vbc[1:-1,-1] += 2*self.vr*cvr - V[:,-1]
        self.Vbc[0,1:-1] += self.vb*cvb 
        self.Vbc[-1,1:-1] += self.vt*cvt 
        
        return np.concatenate((self.Ubc.reshape(-1),self.Vbc.reshape(-1)))
    
    
    def create_side_wall_forcing_profile(self,y0,yf,attr):
        
        l = yf - y0
        for i in range (self.rowsu):
            if self.y[i] >= y0 and self.y[i] <= yf:
                getattr(self,attr)[i] = np.sin((2*np.pi/l)*(self.y[i] - y0))
        
class linear_operators_2D:
    
    def __init__(self,flow,dt):
        
        self.L = lops.laplacian_2D(flow)
        self.I = scipy.sparse.identity(self.L.shape[0])
        self.LR = (1/dt)*self.I 
        self.LL = dt*self.I
        self.G = lops.gradient_2D(flow)
        self.D = self.G.T
        self.luP = splu((self.D.dot((self.G)).tocsc()))
        

class fom_class:
    
    def __init__(self,flow_params,lops):
        
        self.flow_params = flow_params
        self.lops = lops
        
    
    
    def compute_output(self,q):
        return q
    
    def compute_output_derivative(self,q):
        if torch.is_tensor(q):
            return torch.eye(q.shape[1], device=q.device, dtype=q.dtype)
        else:
            return np.eye(q.shape[1], dtype=q.dtype)

    def apply_output_adjoint(self,e,q):
        # compute_output is the identity, so its adjoint is the identity too;
        # this skips ever forming the dense (N, N) output-derivative matrix.
        return e

    def assemble_forcing_profile(self,xc,yc):
        
        dx = self.flow_params.x[1] - self.flow_params.x[0] 
        f = np.zeros(self.flow_params.szu + self.flow_params.szv)
        
        for i in range (self.flow_params.rowsu): 
            
            yi = self.flow_params.y[i]
            
            for j in range (self.flow_params.colsu): 
                
                xj = self.flow_params.x[j] + dx/2 
                f[i*self.flow_params.colsu+j] = np.exp(-5000*((xj - xc)**2 + (yi - yc)**2))
                
        f -= self.lops.G.dot(self.lops.luP.solve(self.lops.D.dot(f))) 
        self.f = f/np.linalg.norm(f)
    
    def assemble_petrov_galerkin_tensors(self,Phi,Psi,B,bc_coefs):
        
        x = self.flow_params.x
        y = self.flow_params.y
        Re = self.flow_params.Re
        q_sbf = self.flow_params.q_sbf
        
        qbc_sbf = self.flow_params.populate_boundary_conditions(q_sbf,*bc_coefs)
        bc_coefs_pert = np.zeros_like(bc_coefs)
        
        PhiF = Phi@sciplin.inv(Psi.T@Phi)
        r = Phi.shape[-1]
        
        A2 = np.zeros((r,r))
        A3 = np.zeros((r,r,r))
        
        for k in range (r): 
            
            q = PhiF[:,k]
            
            qbc = self.flow_params.populate_boundary_conditions(q,*bc_coefs_pert)
            qlap_bc = numbaops.laplacian_boundary_conditions_2D(Re,x,y,q,qbc)
            qdiv_bc = numbaops.divergence_boundary_conditions_2D(Re,x,y,qbc)
            
            f = qlap_bc + self.lops.L.dot(q) - numbaops.evaluate_bilinearity_2D(x,y,q_sbf,q,qbc_sbf,qbc) \
                - numbaops.evaluate_bilinearity_2D(x,y,q,q_sbf,qbc,qbc_sbf)
            f -= self.lops.G.dot(self.lops.luP.solve(self.lops.D.dot(f) - qdiv_bc)) 

            A2[:,k] = Psi.T@f 

        for j in range (r): 
            
            vecj = PhiF[:,j] 
            qbcj = self.flow_params.populate_boundary_conditions(vecj,*bc_coefs_pert)
            
            for k in range (r): 
                veck = PhiF[:,k] 
                
                qbck = self.flow_params.populate_boundary_conditions(veck,*bc_coefs_pert)

                f = -numbaops.evaluate_bilinearity_2D(x,y,vecj,veck,qbcj,qbck)
                f -= self.lops.G.dot(self.lops.luP.solve(self.lops.D.dot(f))) 
                A3[:,j,k] = Psi.T@f 


        return (A2, A3), (Psi.T@B,)
    
    def evaluate_fom_dynamics(self,vec,bc_coefs_bflow,bc_coefs_pert): 
        
        x = self.flow_params.x
        y = self.flow_params.y
        Re = self.flow_params.Re
        q_sbf = self.flow_params.q_sbf
        
        qbc_sbf = self.flow_params.populate_boundary_conditions(q_sbf,*bc_coefs_bflow)
        
        qbc = self.flow_params.populate_boundary_conditions(vec,*bc_coefs_pert)
        qlap_bc = numbaops.laplacian_boundary_conditions_2D(Re,x,y,vec,qbc)
        qdiv_bc = numbaops.divergence_boundary_conditions_2D(Re,x,y,qbc)
        
        f = qlap_bc + self.lops.L.dot(vec) - numbaops.evaluate_bilinearity_2D(x,y,q_sbf,vec,qbc_sbf,qbc) \
            - numbaops.evaluate_bilinearity_2D(x,y,vec,q_sbf,qbc,qbc_sbf) \
            - numbaops.evaluate_bilinearity_2D(x,y,vec,vec,qbc,qbc)
            
        f -= self.lops.G.dot(self.lops.luP.solve(self.lops.D.dot(f) - qdiv_bc))
        
        
        return f
    
    def evaluate_full_fom_dynamics(self,vec,bc_coefs): 
        
        x = self.flow_params.x
        y = self.flow_params.y
        Re = self.flow_params.Re
        q = self.flow_params.q_sbf + vec
        
        qbc = self.flow_params.populate_boundary_conditions(q,*bc_coefs)
        qlap_bc = numbaops.laplacian_boundary_conditions_2D(Re,x,y,q,qbc)
        qdiv_bc = numbaops.divergence_boundary_conditions_2D(Re,x,y,qbc)
        
        f = qlap_bc + self.lops.L.dot(q) - numbaops.evaluate_bilinearity_2D(x,y,q,q,qbc,qbc)
        f -= self.lops.G.dot(self.lops.luP.solve(self.lops.D.dot(f) - qdiv_bc))
        
        
        return f


def estimate_wall_forcing(lops,flow_bflow,flow_pert,vec,bc_coefs_bflow,bc_coefs_pert):
    
    x = flow_bflow.x
    y = flow_bflow.y
    Re = flow_bflow.Re
    q_sbf = flow_bflow.q_sbf
    
    qbc_sbf = flow_bflow.populate_boundary_conditions(q_sbf,*bc_coefs_bflow)
    
    qbc = flow_pert.populate_boundary_conditions(vec,*bc_coefs_pert)
    qlap_bc = numbaops.laplacian_boundary_conditions_2D(Re,x,y,vec,qbc)
    qdiv_bc = numbaops.divergence_boundary_conditions_2D(Re,x,y,qbc)
    
    f = qlap_bc + lops.L.dot(vec) - numbaops.evaluate_bilinearity_2D(x,y,q_sbf,vec,qbc_sbf,qbc) \
        - numbaops.evaluate_bilinearity_2D(x,y,vec,q_sbf,qbc,qbc_sbf) \
        - numbaops.evaluate_bilinearity_2D(x,y,vec,vec,qbc,qbc)
        
    f -= lops.G.dot(lops.luP.solve(lops.D.dot(f) - qdiv_bc))
        
    
    return f