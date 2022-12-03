# -*- coding: utf-8 -*-
# @File:demo.py
# @Author:south wind
# @Date:2022-11-09
# @IDE:PyCharm
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers
import tensorflow as tf
'''neural ODE'''
#实现简单的黑盒求解器，N(t)=N0*exp(-a*t)
a=5
def g_fn(tk,hk):
    return -a*hk
def euler_step(dt,tk,hk,fun):
    return hk+dt*fun(tk,hk)
hist=[]
num_steps=100
interval_steps=1/num_steps
Nk=10
tk=0.0
hist.append([tk,Nk])
for k in range(num_steps):
    Nk=euler_step(interval_steps,tk,Nk,g_fn)
    tk+=interval_steps
    hist.append([tk,Nk])
hist=np.array(hist)
plt.plot(hist[:,0],hist[:,1])
plt.show()
#%%
def odeint(func,y_0,t,solver):
    deta_ts=t[1:]-t[:-1]
    tk=t[0]
    yk=y_0
    hist=[[tk,y_0]]
    for deta_t in deta_ts:
        yk=solver(deta_t,tk,yk,func)
        tk+=deta_t
        hist.append([tk,yk])
    return hist

def midpoint_step_keras(deta_t,tk,hk,fun):
    k1=fun(tk,hk)
    k2=fun(tk+deta_t,hk+deta_t*k1)
    return hk+deta_t*(k1+k2)/2

class Module(keras.Model):
    def __init__(self,nf):
        super(Module,self).__init__()
        self.dense_1=layers.Dense(nf,activation='tanh')
        self.dense_2=layers.Dense(nf,activation='tanh')

    def call(self,inputs,**kwargs):
        t,x=inputs
        h=self.dense_1(x)
        return self.dense_2(h)-0.25*x
t_grid=np.linspace(0,500,2000)
h0=tf.to_float([[1.0,-1.0]])
model=Module(2)
hist=odeint(model,h0,t_grid,midpoint_step_keras)




