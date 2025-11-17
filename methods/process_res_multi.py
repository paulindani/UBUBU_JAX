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

def process_res_multi(res,bind):
    #(rep_inp,niter_inp,detlevels_inp,randlevels_inp,repruns_inp,maxlevel_inp,problevels_inp,ngradtot_inp,nbeta_inp,means_inp,squaremeans_inp,blockstart_inp,blockend_inp,test_dim_inp,maxruns_inp,rho_inp)=res
    (_,_,_,_,repruns_inp,_,_,_,_,means_inp,squaremeans_inp,_,_,_,_,_)=res
    res0=[]
    for tensor in res:
        res0.append(tensor[0,...])
    (rep,niter,detlevels,randlevels,_,maxlevel,problevels,ngradtot,nbeta,_,_,blockstart,blockend,test_dim,maxruns,rho)=res0
    

    wholerep=means_inp.shape[0]
    #bind=np.arange(wholerep)

    
    test_mean=jnp.zeros([1,test_dim])
    test_mean_var=jnp.zeros([1,test_dim])
    test_squaremean=jnp.zeros([1,test_dim])
    means=means_inp[bind,:,:]#jnp.zeros([wholerep,maxruns,test_dim])
    squaremeans=squaremeans_inp[bind,:,:]
    repruns=repruns_inp[bind,:]


    print("mean repruns:",repruns.mean(axis=[0]))
    test_mean=jnp.mean(dynamic_slice(means,(0,blockstart[0],0),(wholerep,blockend[0]-blockstart[0],test_dim)),[0,1])
    test_mean_var=jnp.var(dynamic_slice(means,(0,blockstart[0],0),(wholerep,blockend[0]-blockstart[0],test_dim)),[0,1])/rep
    print("test_mean_var:",test_mean_var[0])
    print("ngradtot:", ngradtot)
    test_squaremean=jnp.mean(dynamic_slice(squaremeans,(0,blockstart[0],0),(wholerep,blockend[0]-blockstart[0],test_dim)),[0,1])
    
    print("test_mean_lev_0_comp1:",test_mean[0])
    print("max_test_mean_lev_0_var:",jnp.max(test_mean_var))
    print("test_squaremean_lev_0_comp1:",test_squaremean[0])

    print("blockstart:",blockstart)
    print("blockend:",blockend)
    print("maxlevel:",maxlevel)


    for lev in range(detlevels-1):
        print("lev:",lev+1)
        max_lev_test_mean_diff=jnp.max(jnp.abs(jnp.mean(dynamic_slice(means,(0,blockstart[lev+1],0),(wholerep,blockend[lev+1]-blockstart[lev+1],test_dim)),[0,1])))
        print("max_lev_test_mean_diff:",max_lev_test_mean_diff)

        test_mean+=jnp.mean(dynamic_slice(means,(0,blockstart[lev+1],0),(wholerep,blockend[lev+1]-blockstart[lev+1],test_dim)),[0,1])
        test_mean_var_diff=jnp.var(dynamic_slice(means,(0,blockstart[lev+1],0),(wholerep,blockend[lev+1]-blockstart[lev+1],test_dim)),[0,1])/repruns[0,lev+1]
        test_mean_var+=test_mean_var_diff
        test_squaremean+=jnp.mean(dynamic_slice(squaremeans,(0,blockstart[lev+1],0),(wholerep,blockend[lev+1]-blockstart[lev+1],test_dim)),[0,1])
        max_lev_test_mean_var=jnp.max(test_mean_var_diff)
        print("max_lev_test_mean_diff_var:",max_lev_test_mean_var,'repruns[lev+1,0]:',repruns[0,lev+1])
            

    mean_last_detlevel_arr=means[0:wholerep,blockstart[detlevels],0:test_dim].reshape([wholerep,test_dim])
    squaremean_last_detlevel_arr=squaremeans[0:wholerep,blockstart[detlevels],0:test_dim].reshape([wholerep,test_dim])
    mean_last_arr=mean_last_detlevel_arr*(1/(1-rho))
    squaremean_last_arr=squaremean_last_detlevel_arr*(1/(1-rho))

    srepruns=jnp.sum(repruns,[0,])
    maxrun=jnp.argmax((srepruns>0)*1)

    def whole_it_loop(wholeit,inp):
        mean_last_arr,squaremean_last_arr,lev=inp
        mean_lev_wit=dynamic_slice(means, (wholeit, blockstart[lev], 0), (1, 1, test_dim)).reshape(1,test_dim)
        mean_last_detlevel_wit=dynamic_slice(mean_last_detlevel_arr, (wholeit,0), (1,test_dim)).reshape(1,test_dim)

        mean_last_wit_add=(mean_lev_wit-(rho**(lev-detlevels))*mean_last_detlevel_wit)/problevels[lev]
        mean_last_arr_wit=dynamic_slice(mean_last_arr, (wholeit,0), (1,test_dim)).reshape(1,test_dim)
        mean_last_arr=dynamic_update_slice(mean_last_arr, mean_last_arr_wit+(repruns[wholeit,lev]>0)*mean_last_wit_add,(wholeit,0))
        
        squaremean_wit=dynamic_slice(squaremeans, (wholeit, blockstart[lev], 0), (1, 1,test_dim)).reshape(1,test_dim)
        squaremean_last_detlevel_wit=dynamic_slice(squaremean_last_detlevel_arr, (wholeit,0), (1,test_dim)).reshape(1,test_dim)
        squaremean_last_wit_add=(squaremean_wit-(rho**(lev-detlevels))*squaremean_last_detlevel_wit)/problevels[lev]
        squaremean_last_arr_wit=dynamic_slice(squaremean_last_arr, (wholeit,0), (1,test_dim)).reshape(1,test_dim)
        squaremean_last_arr=dynamic_update_slice(squaremean_last_arr, squaremean_last_arr_wit+(repruns[wholeit,lev]>0)*squaremean_last_wit_add,(wholeit,0))
        return mean_last_arr,squaremean_last_arr,lev
    
    def lev2_loop(lev,inp):
        mean_last_arr,squaremean_last_arr=inp
        mean_last_arr,squaremean_last_arr,_=jax.lax.fori_loop(0,wholerep,whole_it_loop,(mean_last_arr,squaremean_last_arr,lev))
        return mean_last_arr,squaremean_last_arr

    mean_last_arr,squaremean_last_arr=jax.lax.fori_loop(detlevels+1,maxrun,lev2_loop,(mean_last_arr,squaremean_last_arr))

                
    test_mean=test_mean+jnp.mean(mean_last_arr,[0])
    print("lev:",detlevels)
    print("max_mean_diff_lev:", jnp.max(jnp.abs(jnp.mean(mean_last_arr,[0]))))
    print("max_test_mean_var:", jnp.max(jnp.var(mean_last_arr,[0])))
    
    test_mean_var+=jnp.var(mean_last_arr,[0])
    test_squaremean+=jnp.mean(squaremean_last_arr,[0])

    test_post_var=test_squaremean-jnp.square(test_mean)
    ess=test_post_var/test_mean_var
    grad_per_ess=ngradtot/ess
    print("test_post_var:",test_post_var[0])
    print("median test mean var:",jnp.median(test_mean_var))
    print("median test post var:",jnp.median(test_post_var))
    print("min test post var:",jnp.min(test_post_var))
    print("min_ess:",jnp.min(ess))    
    print("median_ess:",jnp.median(ess))    
    print("maxgradperess:",jnp.max(grad_per_ess))
    print("mediangradperess:",jnp.median(grad_per_ess))
    print("test_mean:",test_mean[0])
    print("test_squaremean:",test_squaremean[0])

    return ess, grad_per_ess
