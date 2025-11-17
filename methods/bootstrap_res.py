import jax
import time
import numpy as np
import jax.numpy as jnp
import numpy as np

from jax import vmap
from jax import random
from jax import device_put, jit
from jax.lax import stop_gradient
from jax.lax import dynamic_slice
from jax.lax import dynamic_update_slice


def bootstrap_res(res,bootstrap_samp,par_runs,twoparts,process_fun,keys):
    list_grad_per_ess_arr=[]
    for it in range(bootstrap_samp):
        bind=jnp.asarray(jnp.floor(random.uniform(keys[it],[par_runs])*par_runs),dtype=int)
        _,grad_per_ess_arr_it=process_fun(res,bind)
        list_grad_per_ess_arr.append(grad_per_ess_arr_it.reshape([1,grad_per_ess_arr_it.size]))
    grad_per_ess_arr=jnp.concat(list_grad_per_ess_arr,axis=0)
    sd=jnp.std(grad_per_ess_arr,axis=0)    
    if(twoparts==None):
        max_grad_per_ess=jnp.max(grad_per_ess_arr,axis=1)
        sdmax=jnp.std(max_grad_per_ess,axis=0)
        sdmax2=sdmax
        max_grad_per_ess2=max_grad_per_ess
    else:
        max_grad_per_ess=jnp.max(grad_per_ess_arr[:,0:twoparts],axis=1)
        sdmax=jnp.std(max_grad_per_ess,axis=0)
        max_grad_per_ess2=jnp.max(grad_per_ess_arr[:,twoparts:],axis=1)
        sdmax2=jnp.std(max_grad_per_ess2,axis=0)        
    return grad_per_ess_arr,max_grad_per_ess,max_grad_per_ess2,sd,sdmax,sdmax2
