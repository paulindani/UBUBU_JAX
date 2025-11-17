import jax
import time
import numpy as np
import jax.numpy as jnp
import numpy as np

from jax import grad
from jax import random
from jax import device_put, jit
from jax.lax import stop_gradient
from jax.lax import dynamic_slice
from jax.lax import dynamic_update_slice
from collections import namedtuple

def hmcsampler(niter,burnin,numsteps,partial,h,lpost,grad_lpost,test_function,test_dim,x0,key):
   
    def hmc_step(x,v,numsteps,partial,key_it):


        
        (key_it,split_key1,split_key2,split_key3)=random.split(key_it,4)
        u=jnp.log(random.uniform(split_key1,[1]))
               
        vo=v
        xo=x
        vn=vo
        xn=xo

        vn=vn-(h/2)*grad_lpost(xn)

        nstep=random.geometric(split_key2,1/(numsteps))
        def inner_step(it,inp):
            (xn2,vn2)=inp
            xn2=xn2+h*vn2
            vn2 = vn2-h*grad_lpost(xn2)
            return (xn2,vn2)
        
        (xn,vn)=jax.lax.fori_loop(0,nstep, inner_step,(xn,vn))
       
        xn=xn+h*vn
        vn=vn-(h/2)*grad_lpost(xn)
        vn=-vn
        acc=jnp.asarray(u<(lpost(xo)-lpost(xn)+jnp.sum(vo**2)/2-jnp.sum(vn**2)/2))
        xn=xn*acc+xo*(1-acc)
        vn=vn*acc+vo*(1-acc)
        

        vn=-vn
        vn=partial*vn+jnp.sqrt(1-partial**2)*random.normal(split_key3,[nbeta,])
        return xn,vn,acc,key_it

    tot_acc=jnp.asarray([0])
    nbeta=x0.shape[0]
    key_hmc,split_key=random.split(key)
    v0=random.normal(split_key,[nbeta,])

    xx=x0
    vv=v0

    means=jnp.zeros(test_dim)
    squaremeans=jnp.zeros(test_dim)

    
    def inner_function_burnin(it,inp):
        (xx,vv,tot_acc,key_hmc)=inp
        [xx,vv,acc,key_hmc]=hmc_step(xx,vv,numsteps,partial,key_hmc)
        tot_acc=tot_acc+acc
        return (xx,vv,tot_acc,key_hmc)
    
    (xx,vv,tot_acc,key_hmc)=jax.lax.fori_loop(0,burnin,inner_function_burnin,(xx,vv,tot_acc,key_hmc))

    def inner_function(it,inp):
        (xx,vv,tot_acc,key_hmc,means,squaremeans)=inp
        [xx,vv,acc,key_hmc]=hmc_step(xx,vv,numsteps,partial,key_hmc)
        tot_acc=tot_acc+acc
        txx=test_function(xx)
        means+=txx
        squaremeans+=(txx**2)
        return (xx,vv,tot_acc,key_hmc,means,squaremeans)
    
    (xx,vv,tot_acc,key_hmc,means,squaremeans)=jax.lax.fori_loop(0,niter,inner_function,(xx,vv,tot_acc,key_hmc,means,squaremeans))

    mean_acc=tot_acc/(niter+burnin)
    means=means/niter
    squaremeans=squaremeans/niter
    return means, squaremeans, mean_acc, numsteps*(niter+burnin)




hmcsampler_jit=jax.jit(hmcsampler, static_argnames=["lpost","grad_lpost","test_function","test_dim"])
vhmcsampler=jax.vmap(hmcsampler_jit,in_axes=[None]*9+[0,0])

def vmap_hmcsampler(niter,burnin,numsteps,partial,h,lpost,grad_lpost,test_function,test_dim,x0,keys,chunk_size):
    rep=keys.shape[0]
    means_arr=[]
    squaremeans_arr=[]
    mean_acc_arr=[]
    for it in range(rep//chunk_size):
        means, squaremeans, mean_acc,_=vhmcsampler(niter,burnin,numsteps,partial,h,lpost,grad_lpost,test_function,test_dim,x0[(it*chunk_size):((it+1)*chunk_size),:],keys[(it*chunk_size):((it+1)*chunk_size)])
        means_arr.append(means)
        squaremeans_arr.append(squaremeans)
        mean_acc_arr.append(mean_acc)
    if(rep%chunk_size>0):
        means, squaremeans, mean_acc,_=vhmcsampler(niter,burnin,numsteps,partial,h,lpost,grad_lpost,test_function,test_dim,x0[((rep//chunk_size)*chunk_size):rep,:],keys[((rep//chunk_size)*chunk_size):rep])
        means_arr.append(means)
        squaremeans_arr.append(squaremeans)
        mean_acc_arr.append(mean_acc)

    return jnp.concatenate(means_arr,0),jnp.concatenate(squaremeans_arr,0),jnp.concatenate(mean_acc_arr,0), numsteps*(niter+burnin)
