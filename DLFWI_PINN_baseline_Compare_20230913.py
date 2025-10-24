#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:52:55 2023

@author: jlhardi
"""

import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import scipy
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator
import tensorflow as tf
import pickle
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import os
#import imageio

#%% define geometry
# need to rescale/normalize from -1 to one for all input/output parrameters

# x from 0 to 1000m (Nx = 1001) and z from 0 to -1000 (Nz = 1001)
# t from 0 to 1 s
minx = 0
minz = 0
maxx = 1.5 # km
maxz = 1.5
maxt = 0.5
scale = 1 # scaling factor for nondimensionalization
geom = dde.geometry.geometry_2d.Rectangle([minx,minz], [maxx/scale,maxz/scale])
timedomain = dde.geometry.TimeDomain(0, maxt)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# source geometry for first shot
sht1x = maxx/2/scale
sht1z = maxz/2/scale
# shtgeom = dde.geometry.pointcloud.PointCloud(sht1)
# shtgeomtime = dde.geometry.GeometryXTime(shtgeom, timedomain)


def boundary_sht(x, on_boundary):
    return np.isclose(x[0], sht1x) and np.isclose(x[1], sht1z)

def boundaryxa(x, on_boundary):
    return np.isclose(x[0], minx)

def boundaryxb(x, on_boundary):
    return np.isclose(x[0], maxx)

def boundaryza(x, on_boundary):
    return np.isclose(x[1], minz)

def boundaryzb(x, on_boundary):
    return np.isclose(x[1], maxz)

def initial_ic2(x, on_initial):
    return np.isclose(x[2], 0)

def boundary_p1(x, on_boundary):
    return np.isclose(x[0], sht1x-0.25) and np.isclose(x[1], sht1z-0.25) and np.isclose(x[2], 0.25)

# def ICdomain(_, on_initial):
#     return on_initial

#%% define PDE, velocity model (IC)

# get vp model
# fam_name = 'Layers2'
# mod = '0005'
# infile = scipy.io.loadmat('input_vp/' + fam_name + '/vp_' + mod + '.mat')
# vp = infile['vp'] / 1000 # convert from m to km
# # need to have function that interpolates vp
# xx = np.linspace(minx, maxx/scale, 2001)
# zz = np.linspace(minz, maxz/scale, 2001)
# X, Z = np.meshgrid(xx, zz)
# xX = np.reshape(X, (-1, 1)).squeeze()
# zZ = np.reshape(Z, (-1, 1)).squeeze()
# vp1d = np.reshape(vp.transpose(), (-1, 1)).squeeze()
# vp_interp = LinearNDInterpolator(list(zip(xX, zZ)), vp1d)

# delta2d = scipy.signal.unit_impulse(X.shape,500)
# d1d = np.reshape(delta2d.transpose(), (-1, 1)).squeeze()
# delta_interp = LinearNDInterpolator(list(zip(xX, zZ)), d1d)

# def vp_pde(x):
#     x, z, t = np.split(x, 3, axis=1)
#     vp_pt = vp_interp(x, z) / scale # need to normalize vp
#     return np.array(vp_pt)

# fp = 2; tm = 0.5
# pdesrc = np.loadtxt('Force_Gaussian_DLWIcomp_PDEsrc.txt')
# pdesrc_interp = scipy.interpolate.CubicSpline(pdesrc[:,0], pdesrc[:,1])
# BCsrc = np.loadtxt('Pressure_Ricker_DLWIcomp_BCsrc.txt')
# BCsrc_interp = scipy.interpolate.CubicSpline(BCsrc[:,0], BCsrc[:,1])

# tsrc = np.linspace(0,maxt,1001)
#tsrc = np.linspace(0,4,1001)
# imp = scipy.signal.unit_impulse(1001)
# b, a = scipy.signal.butter(4, 0.01)
# response = scipy.signal.lfilter(b, a, imp)
# response[0] = 0
# src_interp = scipy.interpolate.CubicSpline(tsrc, response)
fp = 0.5; tm = 2
def ricker_Wavelet_t (t):
    src = (1- 2 * np.pi**2 * fp**2 * (t-tm)**2) * np.exp(-1 * np.pi**2 * fp**2 * (t-tm)**2)
    return src
# def srcfun_Delta (x):
#     x, z, t = np.split(x, 3, axis=1)
#     src = src_interp(t) / np.max(src_interp(t))
#     return src
def forcing(x):
    x,z,t = np.split(x, 3, axis =1)
    src = ricker_Wavelet_t(t) * \
        np.exp(-(100.0*(x-sht1x))**2.0 - (100.0*(z-sht1z))**2.0)
    return np.array(src)

# def aux_fun(x):
#     return np.hstack([vp_pde(x)])

# residual of the homogeneous 2D acoustic wave equation
# x[:,0:1] is x, x[:,1:2] is z, x[:,2:3] is t, and u is pressure

def pde(x, u, ex):
    vp_pt = 1
    src = ex
    du_xx = dde.grad.hessian(u, x, i=0, j=0)
    du_yy = dde.grad.hessian(u, x, i=1, j=1)
    du_tt = dde.grad.hessian(u, x, i=2, j=2)
    return du_tt - vp_pt**2 * (du_xx * 1/(scale**2) + du_yy * 1/(scale**2))-src


#%% define soft BCs and ICs

# Define sine function
#sin = tf.sin

# BCsrc = np.loadtxt('Pressure_Ricker_DLWIcomp_BCsrc.txt')
# src_interp = scipy.interpolate.CubicSpline(BCsrc[:,0], BCsrc[:,1])
# def srcfun_Compare (x):
#     x, z, t = np.split(x, 3, axis=1)
#     src = src_interp(t)
#     return src
    
# # # source term as BC
# tsrc = np.linspace(0,maxt,1001)
# tsrc = np.linspace(0,10,1001)
# imp = scipy.signal.unit_impulse(1001)
# b, a = scipy.signal.butter(4, 0.01)
# response = scipy.signal.lfilter(b, a, imp)
# response[0] = 0
# src_interp = scipy.interpolate.CubicSpline(tsrc, response)
# def srcfun_Delta (x):
#     x, z, t = np.split(x, 3, axis=1)
#     src = src_interp(t) / np.max(src_interp(t))
#     return src

# tsrc = np.linspace(0,maxt,1001)
# def beta_Wavelet (t):
#     src = 4804*t**(2) * (1-t)**(50)
#     return src

# def dt_beta_Wavelet (x):
#     x, z, t = np.split(x, 3, axis=1)
#     src = -2*109*t *(26*t-1) * (1-t)**49
#     return src

fp = 0.5; tm = 2.0
def ricker_Wavelet (x):
    x, z, t = np.split(x, 3, axis=1)
    src = 10 * (1- 2 * np.pi**2 * fp**2 * (t-tm)**2) * np.exp(-1 * np.pi**2 * fp**2 * (t-tm)**2)
    return src

# # differentiable source function for hard constraint
# def hardICBC (X,u):
#     x, z, t = X[:, 0:1], X[:, 1:2], X[:, 2:3]
#     T = sin(60.0 * t) * tf.math.exp(-(100.0*(x-sht1x))**2.0 - (100.0*(z-sht1z))**2.0) + \
#        tf.tanh(t)* u * (-1.0*tf.math.exp(-(30.0*(x-sht1x))**2.0 - (30.0*(z-sht1z))**2.0)+1.0)
#     return T

# differentiable source function for hard constraint
# beta wavelet as source
# fp = 1; tm = 1
# def hardICBC (X,u):
#     x, z, t = X[:, 0:1], X[:, 1:2], X[:, 2:3]
#     src = (1- 2 * 3.1415**2 * fp**2 * (t-tm)**2) * tf.math.exp(-1 * 3.1415**2 * fp**2 * (t-tm)**2)
#     T = src * tf.math.exp(-(100.0*(x-sht1x))**2.0 - (100.0*(z-sht1z))**2.0) + \
#        tf.tanh(t)* u * (-1.0*tf.math.exp(-(100.0*(x-sht1x))**2.0 - (100.0*(z-sht1z))**2.0)+1.0)
#     return T

def hardIC (X,u):
    x, z, t = X[:, 0:1], X[:, 1:2], X[:, 2:3]
    T = tf.tanh(t)* u
    return T

bc_sht1 = dde.icbc.boundary_conditions.DirichletBC(geomtime, ricker_Wavelet, boundary_sht)

ic_zero = dde.icbc.IC(geomtime, lambda x: 0, initial_ic2)

ic_dt_zero = dde.icbc.OperatorBC(
    geomtime,
    lambda x, u, _: dde.grad.jacobian(u,x,i=0,j=2),
    initial_ic2
    )

bc_abs_xa = dde.icbc.boundary_conditions.OperatorBC(
    geomtime,
    lambda x, u, _: dde.grad.hessian(u,x,i=2,j=2)-dde.grad.hessian(u,x,i=2,j=0)-1/2*dde.grad.hessian(u,x,i=1,j=1),
    boundaryxa
    )

bc_abs_xb = dde.icbc.boundary_conditions.OperatorBC(
    geomtime,
    lambda x, u, _: dde.grad.hessian(u,x,i=2,j=2)+dde.grad.hessian(u,x,i=2,j=0)-1/2*dde.grad.hessian(u,x,i=1,j=1),
    boundaryxb
    )

bc_abs_za = dde.icbc.boundary_conditions.OperatorBC(
    geomtime,
    lambda x, u, _: dde.grad.hessian(u,x,i=2,j=2)-dde.grad.hessian(u,x,i=2,j=1)-1/2*dde.grad.hessian(u,x,i=0,j=0),
    boundaryza
    )

bc_abs_zb = dde.icbc.boundary_conditions.OperatorBC(
    geomtime,
    lambda x, u, _: dde.grad.hessian(u,x,i=2,j=2)+dde.grad.hessian(u,x,i=2,j=1)-1/2*dde.grad.hessian(u,x,i=0,j=0),
    boundaryzb
    )  

# define known points that are 0 for times not zero based on Vp
xx = np.array([[0, 0.1, 0.2, 0.25, 1.5, 1.4, 1.3, 1.25]])
zz = np.array([[0, 0.1, 0.2, 0.25, 1.5, 1.4, 1.3, 1.25]])
tt = np.array([[0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]])
X, Z, T = np.meshgrid(xx, zz, tt)
xX = np.reshape(X, (-1, 1)).squeeze()
zZ = np.reshape(Z, (-1, 1)).squeeze()
tT = np.reshape(T, (-1, 1)).squeeze()
anchors_pts021 = np.array([[0.2,1.3,0.75,0.75,0.75,0.75,0.7,0.8],
                            [0.75,0.75,0.2,1.3,0.7,0.8,0.75,0.75],
                            [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]]).transpose()
anchors_pts022 = np.array([[0.2,1.3,0.65,0.85,0.75,0.75,0.75,0.75],
                            [0.75,0.75,0.75,0.75,0.2,1.3,0.65,0.85],
                            [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]]).transpose()
anchors_pts023 = np.array([[0.2,1.3,0.75,0.75,0.75,0.75,0.95,0.55],
                            [0.75,0.75,0.2,1.3,0.95,0.55,0.75,0.75],
                            [0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]]).transpose()
anchors_pts024 = np.array([[0.2,1.3,0.75,0.75,0.75,0.75,1.05,0.45],
                            [0.75,0.75,0.2,1.3,1.05,0.45,0.75,0.75],
                            [0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3]]).transpose()
anchors_pts025 = np.array([[0.2,1.3,0.75,0.75,1.15,0.35,0.75,0.75],
                            [0.75,0.75,0.2,1.3,0.75,0.75,1.15,0.35],
                            [0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4]]).transpose()
anchors_pts0 = np.array([xX,zZ,tT]).transpose()
anchors_pts_all = np.concatenate((anchors_pts021,anchors_pts022,anchors_pts023,
                                  anchors_pts024,anchors_pts025,anchors_pts0))
bc_pts0 = dde.icbc.boundary_conditions.PointSetBC(
    anchors_pts_all,
    0)
bc_pts_dt0 = dde.icbc.boundary_conditions.PointSetOperatorBC(
    anchors_pts_all,
    0,
    lambda x, u, _: dde.grad.jacobian(u,x,i=0,j=2))

# # define slice from physics model or learned early time prediction
# Ptslice05 = np.loadtxt('/gpfs/jlhardi/DLFWI/comp_slices/0.5_learned_xzp.dat')
# #Ptslice10 = np.loadtxt('/gpfs/jlhardi/DLFWI/comp_slices/1.0_learned_xzp.dat')
# fact = 1
# # xX = Ptslice05[:,0] / 1000
# # zZ = Ptslice05[:,1] / 1000
# xX = Ptslice05[:,0]
# zZ = Ptslice05[:,1]
# tT05 = 0.5 * np.ones(zZ.shape)
# #tT10 = 1.0 * np.ones(zZ.shape)
# pP05 = fact * np.array([Ptslice05[:,2]]).transpose()
# #pP10 = fact * np.array([Ptslice10[:,2]]).transpose()
# #pP = np.concatenate((pP05,pP10))
# anchors_Ptslice05 = np.array([xX,zZ,tT05]).transpose()
# #anchors_Ptslice10 = np.array([xX,zZ,tT10]).transpose()
# #anchors_Ptslices = np.concatenate((anchors_Ptslice05,anchors_Ptslice10))
# anchors_Ptslices = anchors_Ptslice05
# bc_Ptslices = dde.icbc.boundary_conditions.PointSetBC(anchors_Ptslice05,pP05)

#%% define NN parameters

# General parameters
train_domain = 55000
train_boundary = 2000
train_initial = 100
test = 100
#loss_weights = [1,1e4,1e3,1e2,1e2,1e2,1e2]
loss_weights = [1,1e3,1e2,1e2,1e2]
iterations = 10000
learning_rate = 1e-4
num_dense_layers = 5
num_dense_nodes = 40
activation = "sin"
xx = np.linspace(minx/scale, maxx/scale, 50)
zz = np.linspace(minz/scale, maxz/scale, 50)
X, Z = np.meshgrid(xx, zz)
xX = np.reshape(X, (-1, 1)).squeeze()
zZ = np.reshape(Z, (-1, 1)).squeeze()
tT = 0.001 * np.ones(zZ.shape)
#tT_2 = maxt * np.ones(zZ.shape)
tT_2 = maxt * np.ones(zZ.shape)
tT0 = np.zeros(zZ.shape)
anchors_t0 = np.array([xX,zZ,tT0]).transpose()
anchors_t2 = np.array([xX,zZ,tT_2]).transpose()
anchors_earlyt = np.array([xX,zZ,tT]).transpose()
anchors_sht = np.zeros([300, 3])
anchors_shtpt = np.zeros([1000, 3])
anchors_shtpt[:, 0] = sht1x
anchors_shtpt[:, 1] = sht1z
anchors_shtpt[:, 2] = np.linspace(0, maxt, 1000)
anchors_sht1 = np.zeros([300,3])
anchors_sht2 = np.zeros([300,3])
anchors_sht3 = np.zeros([300,3])
anchors_sht4 = np.zeros([300,3])
anchors_sht1[:, 0] = sht1x + 0.05/scale
anchors_sht1[:, 1] = sht1z + 0.05/scale
anchors_sht1[:, 2] = np.linspace(0, maxt, 300)
anchors_sht2[:, 0] = sht1x + 0.05/scale
anchors_sht2[:, 1] = sht1z - 0.05/scale
anchors_sht2[:, 2] = np.linspace(0, maxt, 300)
anchors_sht3[:, 0] = sht1x - 0.05/scale
anchors_sht3[:, 1] = sht1z - 0.05/scale
anchors_sht3[:, 2] = np.linspace(0, maxt, 300)
anchors_sht4[:, 0] = sht1x - 0.05/scale
anchors_sht4[:, 1] = sht1z + 0.05/scale
anchors_sht4[:, 2] = np.linspace(0, maxt, 300)
anchors_sht = np.concatenate((anchors_sht1,anchors_sht2,anchors_sht3,anchors_sht4))

anchors_all = np.concatenate((anchors_sht,anchors_shtpt,anchors_t0,anchors_t2))

#%% generate train and test points

data = dde.data.TimePDE(
    geomtime,
    pde,
    # [bc_sht1, ic_dt_zero, bc_Ptslices, bc_abs_xa, bc_abs_xb, bc_abs_za, bc_abs_zb],
    [bc_sht1, ic_dt_zero, bc_pts0, bc_pts_dt0],
    num_domain=train_domain,
    num_test=test,
    num_boundary=train_boundary,
    #num_initial=train_initial,
    anchors=anchors_all,
    #solution=srcfun,
    auxiliary_var_function=forcing,
    train_distribution="uniform"
)

#%% make NN

net = dde.nn.FNN(
    [3] + [num_dense_nodes] * num_dense_layers + [1], activation, "Glorot uniform",
)

# scale inputs
net.apply_feature_transform(lambda x: tf.concat(
    [(2*x[:,0:1])/(maxx/scale)-1, 
      (2*x[:,1:2])/(maxz/scale)-1, 
      (x[:,2:3])/1],axis=1)
) 

net.apply_output_transform(hardIC) # hard IC enforcement
    

# %% make model

# **Can only compile a model once in the same session/kernel
# To continue training, need to save model, restart kernel, then load 

# L4 norm
def L4_loss(y_true, y_pred):
    l4norm = tf.math.reduce_sum((y_pred-y_true)**4)**(1/4)
    return l4norm

def L2_loss(y_true, y_pred):
    l2norm = tf.math.reduce_sum((y_pred-y_true)**2)**(1/2)
    return l2norm

def L6_loss(y_true, y_pred):
    l6norm = tf.math.reduce_sum((y_pred-y_true)**6)**(1/6)
    return l6norm

model = dde.Model(data, net)

model.compile(
    "adam",
    lr=learning_rate,
    loss=L6_loss,
    loss_weights=loss_weights,
    decay=("inverse time", 4000, 0.3)
)

pde_residual_resampler = dde.callbacks.PDEPointResampler(period=100)
checkpointer = dde.callbacks.ModelCheckpoint("/gpfs/jlhardi/DLFWI/PINNmodel/comp/ricker0.5Hz_abs_1.5km_Ptslice0_4s_train8_lbfgsRAR",save_better_only=True,period=200)

## run to here

#%%

restore_path="/gpfs/jlhardi/DLFWI/PINNmodel/comp/ricker0.5Hz_abs_1.5km_NOPtslice_55kpts_0.5s_AdamMSEuni_train1.1-2234.ckpt"
model.restore(restore_path)

#%% RAR loop

X = geomtime.random_points(100000)
err = 1
count = 0

restore_name = "/gpfs/jlhardi/DLFWI/PINNmodel/comp/ricker0.5Hz_abs_1.5km_NOPtslice_55kpts_0.5s_lbfgsL4_train1.2-10016.ckpt"
save_tag = "ricker0.5Hz_abs_1.5km_NOPtslice_55kpts_0.5s_lbfgsL2RAR_train1.3"

iters = 3000

while count < 5:
    count = count + 1
    F = model.predict(X, operator=pde)
    err_eq = np.absolute(F)
    err_eq_sort = np.sort(err_eq,axis=0)[::-1]
    # choose number of points to add
    inx = 9
    err = np.mean(err_eq)
    print("Mean residual: %.3e" % (err))
    x_id = np.argwhere(err_eq>err_eq_sort[inx+1])
    print("Adding new points:", X[x_id[:,0]], "\n")
    data.add_anchors(X[x_id[:,0]])
    early_stopping = dde.callbacks.EarlyStopping(min_delta=1e-4, patience=2000)
    # model.compile("adam",
    #               lr=learning_rate,
    #               loss="MSE",
    #               loss_weights=loss_weights,
    #               #decay=("inverse time", 4000, 0.3)
    #               )
    dde.optimizers.config.set_LBFGS_options(maxcor=1000, ftol=0, gtol=1e-08, maxiter=iters, maxfun=None, maxls=100)
    model.compile("L-BFGS-B", loss_weights=loss_weights,loss=L2_loss)
    
    if count == 1:
        losshistory, train_state = model.train(
            callbacks=[early_stopping],disregard_previous_best=True,display_every=10,
            model_restore_path=restore_name)
    else:
        losshistory, train_state = model.train(
            callbacks=[early_stopping],disregard_previous_best=True,display_every=10)

dde.saveplot(model.losshistory, model.train_state, issave=True, isplot=True, 
             loss_fname='loss' + save_tag + '.dat', 
             train_fname='train' + save_tag + '.dat', 
             test_fname='test' + save_tag + '.dat', 
             output_dir="PINNmodel/comp")

model.save("/gpfs/jlhardi/DLFWI/PINNmodel/comp/" + save_tag, protocol="backend")

loss_curves = np.loadtxt('PINNmodel/comp/loss' + save_tag + '.dat')

plt.figure()
plt.semilogy(loss_curves[:,0],loss_curves[:,1],'k')
plt.semilogy(loss_curves[:,0],loss_curves[:,2],'b')
plt.semilogy(loss_curves[:,0],loss_curves[:,3],'r')
plt.semilogy(loss_curves[:,0],loss_curves[:,4],'g')
plt.semilogy(loss_curves[:,0],loss_curves[:,5],'g')
plt.semilogy(loss_curves[:,0],loss_curves[:,6],'g')
plt.semilogy(loss_curves[:,0],loss_curves[:,7],'g')

#%% start training with L-BFGS

# # L4 norm
# def L4_loss(y_true, y_pred):
#     l4norm = tf.math.reduce_sum((y_pred-y_true)**4)**(1/4)
#     return l4norm

# dde.optimizers.config.set_LBFGS_options(maxcor=500, ftol=0, gtol=1e-08, maxiter=15000, maxfun=None, maxls=100)

# model = dde.Model(data, net)

# pde_residual_resampler = dde.callbacks.PDEPointResampler(period=500)

# model.compile("L-BFGS-B", loss_weights=loss_weights,loss=L4_loss)

# losshistory, train_state = model.train(
#     callbacks=[pde_residual_resampler], display_every=100
#       )

# dde.saveplot(model.losshistory, model.train_state, issave=True, isplot=True, 
#              loss_fname='losslbfgs.dat', train_fname='trainlbfgs.dat', 
#              test_fname='testlbfgs.dat', output_dir="PINNmodel/hardICBC")

# model.save("/gpfs/jlhardi/DLFWI/PINNmodel/hardICBC/dtbeta_softBC_ICs_train1_LBFGS_L4",protocol="backend")

#%% train

save_tag = "ricker0.5Hz_abs_1.5km_5by40_55kpts_0.5s_AdamL6uni_train1.1"

losshistory2, train_state2 = model.train(callbacks=[pde_residual_resampler], 
    iterations=iterations, display_every=10)

dde.saveplot(model.losshistory, model.train_state, issave=True, isplot=True, 
             loss_fname='loss' + save_tag + '.dat', 
             train_fname='train' + save_tag + '.dat', 
             test_fname='test' + save_tag + '.dat', 
             output_dir="PINNmodel/comp")

model.save("/gpfs/jlhardi/DLFWI/PINNmodel/comp/" + save_tag, protocol="backend")

loss_curves = np.loadtxt('PINNmodel/comp/loss' + save_tag + '.dat')

plt.figure()
plt.semilogy(loss_curves[:,0],loss_curves[:,1],'k')
plt.semilogy(loss_curves[:,0],loss_curves[:,2],'b')
plt.semilogy(loss_curves[:,0],loss_curves[:,3],'r')
plt.semilogy(loss_curves[:,0],loss_curves[:,4],'g')
# plt.semilogy(loss_curves[:,0],loss_curves[:,5],'g')
# plt.semilogy(loss_curves[:,0],loss_curves[:,6],'g')
# plt.semilogy(loss_curves[:,0],loss_curves[:,7],'g')
# plt.semilogy(loss_curves[:,0],loss_curves[:,8],'orange')

#%% load previouly trained model


model.restore("/gpfs/jlhardi/DLFWI/PINNmodel/comp/ricker0.5Hz_abs_1.5km_5by20_55kpts_0.5s_AdamL6uni_train1.1-10000.ckpt", verbose=2)

loss_curves = np.loadtxt('PINNmodel/comp/ricker0.5Hz_abs_1.5km_5by20_55kpts_0.5s_LBFSGL6uni_train1.2')

plt.figure()
plt.semilogy(loss_curves[:,0],loss_curves[:,1],'k')
plt.semilogy(loss_curves[:,0],loss_curves[:,2],'b')
plt.semilogy(loss_curves[:,0],loss_curves[:,3],'r')
plt.semilogy(loss_curves[:,0],loss_curves[:,4],'g')
plt.semilogy(loss_curves[:,0],loss_curves[:,5],'g')
plt.semilogy(loss_curves[:,0],loss_curves[:,6],'g')
plt.semilogy(loss_curves[:,0],loss_curves[:,7],'g')
plt.semilogy(loss_curves[:,0],loss_curves[:,8],'orange')


#%% continue training with L-BFGS

restore_name = "/gpfs/jlhardi/DLFWI/PINNmodel/comp/ricker0.5Hz_abs_1.5km_5by20_55kpts_0.5s_AdamL6uni_train1.1-10000.ckpt"
save_tag = "ricker0.5Hz_abs_1.5km_5by20_55kpts_0.5s_LBFSGMSEsobol_train1.2"

iters = 10000

dde.optimizers.config.set_LBFGS_options(maxcor=1000, ftol=0, gtol=1e-08, maxiter=iters, maxfun=None, maxls=100)

model.compile("L-BFGS-B", loss_weights=loss_weights,loss="MSE")

losshistory, train_state = model.train(
    model_restore_path=restore_name, 
    callbacks=[pde_residual_resampler], display_every=10
      )

dde.saveplot(model.losshistory, model.train_state, issave=True, isplot=True, 
             loss_fname='loss_' + save_tag + '.dat', 
             train_fname='train_' + save_tag + '.dat', 
             test_fname='test_' + save_tag + '.dat', 
             output_dir="PINNmodel/comp")

model.save("/gpfs/jlhardi/DLFWI/PINNmodel/comp/" + save_tag, protocol="backend")

loss_curves = np.loadtxt('PINNmodel/comp/loss_' + save_tag + '.dat')

plt.figure()
plt.semilogy(loss_curves[:,0],loss_curves[:,1],'k')
plt.semilogy(loss_curves[:,0],loss_curves[:,2],'b')
plt.semilogy(loss_curves[:,0],loss_curves[:,3],'r')
plt.semilogy(loss_curves[:,0],loss_curves[:,4],'g')
plt.semilogy(loss_curves[:,0],loss_curves[:,5],'g')
plt.semilogy(loss_curves[:,0],loss_curves[:,6],'g')
plt.semilogy(loss_curves[:,0],loss_curves[:,7],'g')

#%% continue training loaded model with Adam

restore_name = "/gpfs/jlhardi/DLFWI/PINNmodel/comp/ricker0.5Hz_abs_1.5km_Ptslice0_0.5s_train2_lbfgs-4377.ckpt"
save_tag = "ricker0.5Hz_abs_1.5km_Ptslice0_1s_train3"

losshistory3, train_state3 = model.train(
    iterations=iterations, callbacks=[], display_every=10,
    model_restore_path=restore_name)

dde.saveplot(model.losshistory, model.train_state, issave=True, isplot=True, 
             loss_fname='loss_' + save_tag + '.dat', 
             train_fname='train_' + save_tag + '.dat', 
             test_fname='test_' + save_tag + '.dat', 
             output_dir="PINNmodel/comp")

model.save("/gpfs/jlhardi/DLFWI/PINNmodel/comp/" + save_tag, protocol="backend")

loss_curves = np.loadtxt('PINNmodel/comp/loss_' + save_tag + '.dat')

plt.figure()
plt.semilogy(loss_curves[:,0],loss_curves[:,1],'k')
plt.semilogy(loss_curves[:,0],loss_curves[:,2],'b')
plt.semilogy(loss_curves[:,0],loss_curves[:,3],'r')
plt.semilogy(loss_curves[:,0],loss_curves[:,4],'g')
plt.semilogy(loss_curves[:,0],loss_curves[:,5],'g')
plt.semilogy(loss_curves[:,0],loss_curves[:,6],'g')
plt.semilogy(loss_curves[:,0],loss_curves[:,7],'g')

#%%
    
    
# Gets variable values as a list of pairs with the name and the value
def get_variable_values(model):
    # Find variable operations
    var_ops = [op for op in model.sess.graph.get_operations() if op.type == 'VariableV2']
    # Get the values
    var_values = []
    for v in var_ops:
        try:
            var_values.append(model.sess.run(v.outputs[0]))
        except tf.errors.FailedPreconditionError:
            # Uninitialized variables are ignored
            pass
    # Return the pairs list
    return [(op.name, val) for op, val in zip(var_ops, var_values)]

def restore_var_values(model, var_values):
    # Find the variable initialization operations
    assign_ops = [model.sess.graph.get_operation_by_name(v + '/Assign') for v, _ in var_values]
    # Run the initialization operations with the given variable values
    model.sess.run(assign_ops, feed_dict={op.inputs[1]: val
                                    for op, (_, val) in zip(assign_ops, var_values)})
    
test_in2 = get_variable_values(model)

with open('PINNmodel/comp/ricker0.1_abs_5km_train4_lbfgs.ppkl', 'wb') as f:
    pickle.dump(test_in2,f)

    
with open('test_10k.list.pkl', 'rb') as f:
    vars_loaded = pickle.load(f)
    
restore_var_values(model,vars_loaded)

#%% predict

def predict_tslice(time,area,Np,v):
    xx = np.linspace(area[0],area[1],Np[0])
    zz = np.linspace(area[2],area[3],Np[1])
    X,Z = np.meshgrid(xx,zz)
    xX = np.reshape(X, (-1, 1)).squeeze()
    zZ = np.reshape(Z, (-1, 1)).squeeze()
    tT = time * np.ones(zZ.shape)
    X_pred = np.array([xX,zZ,tT]).transpose()
    y_pred = model.predict(X_pred)
    plt.figure()
    plt.imshow(np.reshape(y_pred,X.shape),vmin=v[0],vmax=v[1],
                extent=[area[0],area[1],area[3],area[2]],cmap='seismic')
    # plt.imshow(np.reshape(y_pred,X.shape),vmin=-1,vmax=1,
    #            cmap='seismic')        
    plt.colorbar()
    plt.title('t='+str(time))
    return y_pred

def predict_gather(area,Np,Nt,v):
    xx = np.linspace(area[0],area[1],Np)
    tt = np.linspace(0,maxt,Nt)
    X,T = np.meshgrid(xx,tt)
    xX = np.reshape(X, (-1, 1)).squeeze()
    tT = np.reshape(T, (-1, 1)).squeeze()
    zZ = np.ones(tT.shape) * area[2]
    X_pred = np.array([xX,zZ,tT]).transpose()
    y_pred = model.predict(X_pred)
    plt.figure()
    plt.imshow(np.reshape(y_pred,X.shape),vmin=v[0],vmax=v[1],
               extent=[area[0],area[1],maxt,0],cmap='seismic')
    plt.colorbar()
    return y_pred

def predict_point(x,z,tend):
    x_pred = np.zeros([1000,3])
    x_pred[:,0] = x
    x_pred[:,1] = z
    x_pred[:,2] = np.linspace(0,tend,1000)  
    y_pred = model.predict(x_pred)
    plt.figure()
    plt.plot(x_pred[:,2],y_pred)
    return y_pred

def loss_map_pde(time,area,Np,v):
    xx = np.linspace(area[0],area[1],Np[0])
    zz = np.linspace(area[2],area[3],Np[1])
    X,Z = np.meshgrid(xx,zz)
    xX = np.reshape(X, (-1, 1)).squeeze()
    zZ = np.reshape(Z, (-1, 1)).squeeze()
    tT = time * np.ones(zZ.shape)
    X_pred = np.array([xX,zZ,tT]).transpose()
    y_pred = model.predict(X_pred, operator=pde)
    plt.figure()
    plt.imshow(np.reshape(y_pred,X.shape),vmin=v[0],vmax=v[1],
                extent=[area[0],area[1],area[3],area[2]],cmap='seismic')   
    plt.colorbar()
    return y_pred

def loss_map_ic2(time,area,Np,v):
    xx = np.linspace(area[0],area[1],Np[0])
    zz = np.linspace(area[2],area[3],Np[1])
    X,Z = np.meshgrid(xx,zz)
    xX = np.reshape(X, (-1, 1)).squeeze()
    zZ = np.reshape(Z, (-1, 1)).squeeze()
    tT = time * np.ones(zZ.shape)
    X_pred = np.array([xX,zZ,tT]).transpose()
    y_pred = model.predict(X_pred, operator=lambda x, u, _: dde.grad.jacobian(u,x,i=0,j=2))
    plt.figure()
    plt.imshow(np.reshape(y_pred,X.shape),vmin=v[0],vmax=v[1],
                extent=[area[0],area[1],area[3],area[2]],cmap='seismic')   
    plt.colorbar()
    return y_pred

#%%

lossmap_pde = loss_map_pde(0.5,[0, maxx/scale, 0, maxz/scale],[501,501],[-100,100])

lossmap_ic2 = loss_map_ic2(0.0,[0, maxx/scale, 0, maxz/scale],[45,45],[-10,10])
    
testy = predict_tslice(0.5,[0, maxx/scale, 0, maxz/scale],[31,31],[-1,1])
plt.plot(sht1z/scale,sht1x/scale,'ko')

testy = predict_gather([0, 1/scale, 0.6],301,101,[-1,1])

testy = predict_point(sht1x, sht1z, maxt)
plt.plot(anchors_shtpt[:, 2], ricker_Wavelet(anchors_shtpt), 'k--')

testy = predict_point(0.5, 0.7, maxt)

# plot physics model
Ptslice = np.loadtxt('/gpfs/jlhardi/DLFWI/comp_slices/0.5_xzp.dat')
plt.figure()
plt.imshow(np.reshape(10*Ptslice[:,2], [np.sqrt(Ptslice.shape[0]).astype(int),np.sqrt(Ptslice.shape[0]).astype(int)]),
           vmin=-1, vmax=1, extent=[0, maxx/scale, 0, maxz/scale], cmap='seismic')
# plt.imshow(np.reshape(y_pred,X.shape),vmin=-1,vmax=1,
#            cmap='seismic')
plt.colorbar()
plt.title('ParAcousti time slice')

# save predicted early time slice at 0.5 seconds to use as constraint in future time staging
xx = np.linspace(0,1.5,51)
zz = np.linspace(0,1.5,51)
X,Z = np.meshgrid(xx,zz)
xX = np.reshape(X, (-1, 1)).squeeze()
zZ = np.reshape(Z, (-1, 1)).squeeze()
tT = 0.5 * np.ones(zZ.shape)
X_pred = np.array([xX,zZ,tT]).transpose()
y_pred = model.predict(X_pred)
np.savetxt('/gpfs/jlhardi/DLFWI/comp_slices/0.5_learned_xzp.dat',np.array([xX,zZ,y_pred.squeeze()]).transpose())

testy = predict_tslice(0.5,[0, maxx/scale, 0, maxz/scale],[31,31],[-1,1])
Ptslice = np.loadtxt('/gpfs/jlhardi/DLFWI/comp_slices/0.5_xzp.dat')
mse_slice = np.mean((10*Ptslice[:,2]-testy)**2)

# #%% make gif

# def predict_tslice_gif(time,area,Np,v):
#     xx = np.linspace(area[0],area[1],Np[0]
#     zz = np.linspace(area[2],area[3],Np[1])
#     X,Z = np.meshgrid(xx,zz)
#     xX = np.reshape(X, (-1, 1)).squeeze()
#     zZ = np.reshape(Z, (-1, 1)).squeeze()
#     tT = time * np.ones(zZ.shape)
#     X_pred = np.array([xX,zZ,tT]).transpose()
#     y_pred = Image.fromarray(model.predict(X_pred))
#     draw = ImageDraw.Draw(y_pred)
#     return draw._image

# def make_gif(t0,dt,filename):
#     frames = []
#     for i in range(12):
#         frames.append(predict_tslice_gif(t0+dt*i,[0, maxx/scale, 0, maxz/scale],[501,501],[-1,1]))
        
#     frame_one = frames[0]
#     frame_one.save(filename, format="GIF", append_images=frames,
#                     save_all=True, duration=100, loop=0)
#     return frames
    
# frames = make_gif(0,0.05,"PINNmodel/comp/prop_ex.gif")


#plt.subplots_adjust(top=1, bottom=0, left=0, right=1)

for i in np.arange(0.0, 4.0, 0.1):
    tslice = i
    testy = predict_tslice(np.round(tslice,1),[0, maxx/scale, 0, maxz/scale],[501,501],[-10,10])
    plt.plot(sht1z/scale,sht1x/scale,'ko')
    plt.savefig('PINNmodel/comp/figs/t' + str(np.round(tslice,1)) + '.png', format='png')
    plt.close()
    
# with imageio.get_writer('test.gif', mode='I') as writer:
# for filename in filenames:
#     image = imageio.imread(filename)Adam
#     writer.append_data(image)

def animate(i):
    #tslice = np.linspace(0,4,41)
    im = plt.imread('PINNmodel/comp/figs/t'+str(np.round(i,1))+'.png')
    ax = plt.gca()
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(im)

 
anim = FuncAnimation(plt.figure(), animate, np.arange(0.0, 4.0, 0.1),interval=10)
writer = animation.PillowWriter(fps=5, bitrate=1800)
anim.save("PINNmodel/comp/figs/prop_0.5Hz_4s_1.5km_Tstaged.gif", writer=writer)

#%% plot

Xtest = train_state.X_test
Xtrain = train_state.X_train
Ytrain = train_state.y_pred_train
Ytest = train_state.y_pred_test


# plot results from source location
pos_train = np.where((Xtrain[:,0]==0.175) & (Xtrain[:,1]==0))
train_anchors = Ytrain[pos_train,:].squeeze()

pos_test = np.where((Xtest[:,0]==0.175) & (Xtest[:,1]==0))
test_anchors = Ytest[pos_test,:].squeeze()


# plot time slice
time_slice = 0.5
pos_train = np.where((np.isclose(Xtrain[:,2],time_slice,0.001)))
train_slice = Ytrain[pos_train,:]

triang = tri.Triangulation(Xtrain[pos_train,0].squeeze(),Xtrain[pos_train,1].squeeze())
interpolator = tri.LinearTriInterpolator(triang, train_slice.squeeze())
Ytrain_slice = interpolator(X,Z)

plt.imshow(Ytrain_slice, extent=[0, 1000, -1000, 0])
plt.plot(Xtrain[pos_train,0].squeeze(),Xtrain[pos_train,1].squeeze(),'ko')
plt.colorbar()