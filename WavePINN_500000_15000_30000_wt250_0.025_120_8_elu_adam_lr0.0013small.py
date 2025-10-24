# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 15:08:04 2023

@author: sjgauva
"""
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
# import tensorflow as tf
import torch
import scipy
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator
import os
###
# cwd = os.getcwd()
# pinn_output = os.path.join(cwd,'LDRD_wavepinn_model_outputs')
pinn_output = os.path.join('C:\\Users\\hyoon\\Documents\\LDRD_wave\\')
os.chdir(pinn_output)
# if not os.path.exists(pinn_output):
#     os.makedirs(pinn_output)
# pinn_output = "/home/sjgauva/code/pinn_model_outputs/"
#%% define geometry
# need to rescale/normalize from -1 to one for all input/output parrameters

# x from 0 to 1000m (Nx = 1001) and z from 0 to -1000 (Nz = 1001)
# t from 0 to 1 s
minx = 0
minz = 0
maxx = 1 # km
maxz = 1
maxt = 0.5
scale = 1 # scaling factor for nondimensionalization
geom = dde.geometry.geometry_2d.Rectangle([minx,minz], [maxx/scale,maxz/scale])
timedomain = dde.geometry.TimeDomain(0, maxt)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# source geometry for first shot
sht1x = 0.5/scale
sht1z = 0.5/scale
# shtgeom = dde.geometry.pointcloud.PointCloud(sht1)
# shtgeomtime = dde.geometry.GeometryXTime(shtgeom, timedomain)
def boundary_sht(x, on_boundary):
    return np.isclose(x[0], sht1x) and np.isclose(x[1], sht1z)
def pde(x, u):
    vp_pt = 1
    du_xx = dde.grad.hessian(u, x, i=0, j=0)
    du_yy = dde.grad.hessian(u, x, i=1, j=1)
    du_tt = dde.grad.hessian(u, x, i=2, j=2)
    return du_tt - vp_pt**2 * (du_xx * 1/(scale**2) + du_yy * 1/(scale**2))
sin = dde.backend.pytorch.sin
# sin = tf.sin  
# source term as BC
is_domain = 'small'
nt_src = 501  #1001
tsrc = np.linspace(0,maxt,nt_src)
imp = scipy.signal.unit_impulse(nt_src)
b, a = scipy.signal.butter(4, 0.01)
response = scipy.signal.lfilter(b, a, imp)
response[0] = 0
src_interp = scipy.interpolate.CubicSpline(tsrc, response)

def srcfun_Delta (x):
    x, z, t = np.split(x, 3, axis=1)
    src = src_interp(t) / np.max(src_interp(t))
    return src
def hardIC(X,u):
    x, z, t = X[:, 0:1], X[:, 1:2], X[:, 2:3]
    T = dde.backend.pytorch.tanh(t)* u
    # T = tf.tanh(t)* u
    return T
bc_sht1 = dde.icbc.boundary_conditions.DirichletBC(geomtime, srcfun_Delta, boundary_sht) # set source function
ic_dt_zero = dde.icbc.OperatorBC(
    geomtime,
    lambda x, u, _: dde.grad.jacobian(u,x,i=0,j=2),
    lambda _, on_initial: on_initial,
    )
#%% define NN parameters

run = 41

# train_domain = 33000000
train_domain = 500000
train_boundary = 15000  #4100
#train_initial = 10000
test = 20000
iterations = 30000
# iterations = 10000
loss_weights = [1,250,0.025]
# loss_weights = [1,10000,10000]
learning_rate = 0.0013  #0.0013
step_size = iterations/10 
# gamma = 0.75  #0.68**10 * 0.005 = 0.00010569614100786062
gamma = 0.78  #0.78**10 = 0.08335775831236203
# T_max = iterations
# eta_min = 1.e-4
num_dense_layers = 8
num_dense_nodes = 120
activation = "elu"
# activation = "relu"
# activation = "tanh"
opt_str = 'adam'
# opt_str = 'lbgfs'
resample_period = 1000
# num_dense_layers = 9
# num_dense_nodes = 41
# activation = "sin"
# resample_period = 280

nxnz=51 #101
xx = np.linspace(minx/scale, maxx/scale, nxnz)
zz = np.linspace(minz/scale, maxz/scale, nxnz)
# xx = np.linspace(minx/scale, maxx/scale, 100)
# zz = np.linspace(minz/scale, maxz/scale, 100)
X, Z = np.meshgrid(xx, zz)
xX = np.reshape(X, (-1, 1)).squeeze()
zZ = np.reshape(Z, (-1, 1)).squeeze()
tT0 = np.zeros(zZ.shape)
tT = 0.046*2 * np.ones(zZ.shape)  # at peak (src_norm =1)
tT2 = 0.109*2 * np.ones(zZ.shape)  # at the min (-0.177787)
tT3 = 0.1625*2 * np.ones(zZ.shape)  # at 2nd peak (0.048381)
# tT = 0.001 * np.ones(zZ.shape)
# tT2 = 0.01 * np.ones(zZ.shape)
anchors_t0 = np.array([xX,zZ,tT0]).transpose()
anchors_earlyt = np.array([xX,zZ,tT]).transpose()
anchors_earlyt2 = np.array([xX,zZ,tT2]).transpose()
anchors_earlyt3 = np.array([xX,zZ,tT3]).transpose()
nt_step=251 #501 #300
anchors_shtpt = np.zeros([nt_step, 3])
anchors_shtpt[:, 0] = sht1x
anchors_shtpt[:, 1] = sht1z
anchors_shtpt[:, 2] = np.linspace(0, maxt, nt_step)
anchors_sht1 = np.zeros([nt_step,3])
anchors_sht2 = np.zeros([nt_step,3])
anchors_sht3 = np.zeros([nt_step,3])
anchors_sht4 = np.zeros([nt_step,3])
anchors_sht1[:, 0] = sht1x + 0.05/scale
anchors_sht1[:, 1] = sht1z + 0.05/scale
anchors_sht1[:, 2] = np.linspace(0, maxt, nt_step)
anchors_sht2[:, 0] = sht1x + 0.05/scale
anchors_sht2[:, 1] = sht1z - 0.05/scale
anchors_sht2[:, 2] = np.linspace(0, maxt, nt_step)
anchors_sht3[:, 0] = sht1x - 0.05/scale
anchors_sht3[:, 1] = sht1z - 0.05/scale
anchors_sht3[:, 2] = np.linspace(0, maxt, nt_step)
anchors_sht4[:, 0] = sht1x - 0.05/scale
anchors_sht4[:, 1] = sht1z + 0.05/scale
anchors_sht4[:, 2] = np.linspace(0, maxt, nt_step)

anchors_sht = np.concatenate((anchors_sht1,anchors_sht2,anchors_sht3,anchors_sht4))
anchors_all = np.concatenate((anchors_t0,anchors_earlyt,anchors_earlyt2,anchors_earlyt3,anchors_shtpt))
# anchors_all = np.concatenate((anchors_t0,anchors_earlyt,anchors_earlyt2,anchors_earlyt3,anchors_sht,anchors_shtpt))
# anchors_all = np.concatenate((anchors_t0,anchors_earlyt,anchors_earlyt2,anchors_sht,anchors_shtpt))
#%% generate train and test points

data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc_sht1,ic_dt_zero],
    #ic_zero,
    num_domain=train_domain,
    num_test=test,
    num_boundary=train_boundary,
    #num_initial=train_initial,
    anchors=anchors_all
    #solution=srcfun,
    #auxiliary_var_function=aux_fun,
    # train_distribution="Sobol",
)

#%% make NN

net = dde.nn.FNN(
    [3] + [num_dense_nodes] * num_dense_layers + [1], activation, "Glorot uniform",
)

# scale inputs
# net.apply_feature_transform(lambda x: tf.concat(
net.apply_feature_transform(lambda x: dde.backend.pytorch.concat(
    [(2*x[:,0:1])/(maxx/scale)-1, 
      (2*x[:,1:2])/(maxz/scale)-1, 
      (x[:,2:3])/maxt],axis=1)
) 

net.apply_output_transform(hardIC) # hard IC enforcement

#%% make model

# **Can only compile a model once in the same session/kernel
# To continue training, need to save model, restart kernel, then load 

model = dde.Model(data, net)

# dde.optimizers.config.set_LBFGS_options(maxcor=200, ftol=0, gtol=1e-08, maxiter=iterations, maxfun=None, maxls=100)

# model.compile(
#     "L-BFGS-B",
#     lr=learning_rate,
#     loss='MSE',
#     loss_weights=loss_weights
# )
model.compile(
    "adam",
    lr=learning_rate,
    loss='MSE',
    loss_weights=loss_weights,
    decay=("step", step_size, gamma)
)

    # decay=("cosine", T_max, eta_min)
# model.compile(
#     "adam",
#     lr=learning_rate,
#     loss='MSE',
#     loss_weights=loss_weights,
#     decay=("inverse time", 2000, 0.5)
# )
pde_residual_resampler = dde.callbacks.PDEPointResampler(period=resample_period)

#save_model = dde.callbacks.ModelCheckpoint("/gpfs/jlhardi/DLFWI/PINNmodel/hardICBC_train1_500k/hardICBC_train1_500k",
#                                           verbose=2,save_better_only=True,period=1000)
# os.mkdir(pinn_output + str(run) + '_' + str(iterations) + '/')
# output_dir = (pinn_output + str(run) + '_' + str(iterations) + '/')
output_dir = (pinn_output + 'WavePINN_'+str(train_domain)+'_'+str(train_boundary)+'_' \
              +str(iterations)+'_'+'wt'+str(loss_weights[1])+'_'+str(loss_weights[2]) \
              +'_'+str(num_dense_nodes)+'_'+str(num_dense_layers)+'_'+str(activation)+'_'+opt_str+'_'+'lr'+str(learning_rate)+is_domain+'/')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
os.chdir(output_dir)

decay_lr = learning_rate / (1 + 0.2 * np.linspace(0,10000,500) / 100) # 1+decay_rate / iterations/step_size

#%% train

losshistory, train_state = model.train(
    iterations=iterations, callbacks=[pde_residual_resampler], display_every=200)

dde.saveplot(model.losshistory, model.train_state, issave=True, isplot=True, 
             loss_fname='loss1.dat', train_fname='train1.dat', 
             test_fname='test1.dat', output_dir= output_dir + "train1_hardIC_trainBC")

model.save(output_dir + 'train1_hardIC_trainBC_model' ,protocol="backend")
plt.savefig('train_test_loss.png')
#%% plot loss curves
loss_curves = np.loadtxt(output_dir + "/train1_hardIC_trainBC/loss1.dat")

plt.figure()
plt.semilogy(loss_curves[:,0],loss_curves[:,1],'k')
plt.semilogy(loss_curves[:,0],loss_curves[:,2],'b')
plt.semilogy(loss_curves[:,0],loss_curves[:,3],'green')
plt.legend(['PDE', 'BC', 'IC'])
plt.title('Run ' + str(run) + ' training')
plt.savefig('Run_' + str(run) + '_training.png')
plt.ylabel('MSE Loss')
plt.xlabel('Iterations')
# #plt.semilogy(loss_curves[:,0],loss_curves[:,3],'r')

# plt.figure()
# plt.semilogy(loss_curves[:,0],loss_curves[:,1]+loss_curves[:,2])
# plt.semilogy(loss_curves[:,0],loss_curves[:,3]+loss_curves[:,4])

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
                extent=[area[0]*scale,area[1]*scale,area[3]*scale,area[2]]*scale,
                cmap='seismic')
    # plt.imshow(np.reshape(y_pred,X.shape),vmin=-1,vmax=1,
    #            cmap='seismic')        
    plt.colorbar(label = 'Normalized Pressure')
    return y_pred

def predict_gather(area,zconst,Np,Nt,v):
    xx = np.linspace(area[0],area[1],Np)
    tt = np.linspace(0,maxt,Nt)
    X,T = np.meshgrid(xx,tt)
    xX = np.reshape(X, (-1, 1)).squeeze()
    tT = np.reshape(T, (-1, 1)).squeeze()
    zZ = zconst * np.ones(tT.shape)
    X_pred = np.array([xX,zZ,tT]).transpose()
    y_pred = model.predict(X_pred)
    plt.figure()
    plt.imshow(np.reshape(y_pred,X.shape),vmin=v[0],vmax=v[1],
               extent=[area[0]*scale,area[1]*scale,maxt*scale,0*scale],cmap='seismic')
    plt.colorbar(label = 'Normalized Pressure')
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
#%% save and predict time slices

for time in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    testy = predict_tslice(time,[0, maxx/scale, 0, maxz/scale],[301,151],[-1,1])
    plt.plot(sht1z/scale,sht1x/scale,'ko')
    plt.xlabel('Horizontal Distance (Km)')
    plt.ylabel('Depth (Km)')
    plt.title('Time = ' + str(time) + ' s')
    plt.savefig(str(run) + '_' + str(time) + '_s.png')

#%% shot gather over 0-0.5 s

testy = predict_gather([0, 1/scale],0.6,301,101,[-1,1])
plt.title('Shot Gather at z = 0.6 Km')
plt.xlabel('Horizontal Distance (Km)')
plt.ylabel('Time (s)')
plt.savefig('gather' + str(run) + '.png')
#%% predicted and imposed BCs

testy = predict_point(sht1x, sht1z, maxt)
plt.plot(tsrc,src_interp(tsrc)/np.max(src_interp(tsrc)),'k--')
plt.legend(['Predicted','Imposed'])
plt.xlabel('Time (s)')
plt.ylabel('Normalized Pressure')
plt.savefig(str(run) + '_imposed.png')

os.chdir(pinn_output)
