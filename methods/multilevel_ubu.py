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
from jax.lax import dynamic_update_slice,dynamic_update_index_in_dim
from mpmath import mp
import mpmath
mp.dps = 100

def to_mp(x):    
    return np.asarray(mpmath.mpmathify(x.item()),dtype="object")

def hper2constarr(h,gam):
    npgam=to_mp(stop_gradient(gam))

    gh=npgam*to_mp(stop_gradient(h))
    s=np.sqrt(4*mpmath.expm1(-gh/2)-mpmath.expm1(-gh)+gh)

    hc=stop_gradient(jnp.concatenate([jnp.asarray([h]), #h
    jnp.asarray(np.asarray([mpmath.exp(-gh/2)],dtype="float64")), #eta
    jnp.asarray(np.asarray([-mpmath.expm1(-gh/2)/npgam],dtype="float64")), #etam1g
    jnp.asarray(np.asarray([s/npgam],dtype="float64")), #c11
    jnp.asarray(np.asarray([mpmath.exp(-gh)*(mpmath.expm1(gh/2)*mpmath.expm1(gh/2))/s],dtype="float64")), #c21
    jnp.asarray(np.asarray([mpmath.sqrt(8*mpmath.expm1(-gh/2)-4*mpmath.expm1(-gh)-gh*mpmath.expm1(-gh))/s],dtype="float64"))])) #c22
    return hc

# def np64(x):
#     return np.asarray(x,dtype="float64")

# def hper2constarr(h,gam):
#     npgam=np64(stop_gradient(gam))

#     gh=npgam*np64(stop_gradient(h))
#     s=np.sqrt(4*np.expm1(-gh/2)-np.expm1(-gh)+gh)

#     hc=stop_gradient(jnp.concatenate([jnp.asarray([h]), #h
#     jnp.asarray([np.exp(-gh/2)]), #eta
#     jnp.asarray([-np.expm1(-gh/2)/npgam]), #etam1g
#     jnp.asarray([s/npgam]), #c11
#     jnp.asarray([np.exp(-gh)*(np.expm1(gh/2)*np.expm1(gh/2))/s]), #c21
#     jnp.asarray([np.sqrt(8*np.expm1(-gh/2)-4*np.expm1(-gh)-gh*np.expm1(-gh))/s])])) #c22
#     return hc


def U(x,v,hc,xi1,xi2):
    xn=x+hc[2]*v+hc[3]*xi1
    vn=v*hc[1]+hc[4]*xi1+hc[5]*xi2
    return xn,vn

def multiU(x,v,hc,xiarr):
    def body_fn(inp,xi):
        (xn,vn)=inp
        xn=xn+hc[2]*vn+hc[3]*xi[0,:]
        vn=vn*hc[1]+hc[4]*xi[0,:]+hc[5]*xi[1,:]
        return (xn,vn),0
    (xn,vn),_=jax.lax.scan(body_fn,(x,v),xiarr)
    return xn,vn

def UBU_step(x,v,hper2c,vmap_grad_lpost,xi):
    if(x.ndim==1):
        rep=1
    else:
        rep=x.shape[0]
    xn,vn=U(x,v,hper2c,xi[0:rep,:],xi[rep:(2*rep),:])
    gr=vmap_grad_lpost(xn)
    vn=vn-(hper2c[0])*gr
    xn,vn=U(xn,vn,hper2c,xi[(2*rep):(3*rep),:],xi[(3*rep):(4*rep),:])
    return xn,vn

def UBU_step2(x,v, x2,v2,hper4c,vmap_grad_lpost,xi):
    if(x.ndim==1):
        rep=1
    else:
        rep=x.shape[0]
    [x2n,v2n]=U(x2,v2,hper4c,xi[0:rep,:],xi[(rep):(2*rep),:])
    gr=vmap_grad_lpost(x2n)
    v2n=v2n-hper4c[0]*gr
    [x2n,v2n]=U(x2n,v2n,hper4c,xi[(2*rep):(3*rep),:],xi[(3*rep):(4*rep),:])
    [x2n,v2n]=U(x2n,v2n,hper4c,xi[(4*rep):(5*rep),:],xi[(5*rep):(6*rep),:])
    gr=vmap_grad_lpost(x2n)
    v2n=v2n-hper4c[0]*gr
    [x2n,v2n]=U(x2n,v2n,hper4c,xi[(6*rep):(7*rep),:],xi[(7*rep):(8*rep),:])

    [xn,vn]=U(x,v,hper4c,xi[0:rep,:],xi[(rep):(2*rep),:])
    [xn,vn]=U(xn,vn,hper4c,xi[(2*rep):(3*rep),:],xi[(3*rep):(4*rep),:])
    gr=vmap_grad_lpost(xn)
    vn=vn-hper4c[0]*2*gr
    [xn,vn]=U(xn,vn,hper4c,xi[(4*rep):(5*rep),:],xi[(5*rep):(6*rep),:])
    [xn,vn]=U(xn,vn,hper4c,xi[(6*rep):(7*rep),:],xi[(7*rep):(8*rep),:])
    return xn,vn,x2n,v2n



def burnMCMC(x,v,nbeta,burn,hper2c,vmap_grad_lpost,key):
    if(x.ndim==1):
        rep=1
    else:
        rep=x.shape[0]
    #print("x.shape:",x.shape, "v.shape:", v.shape)
    xn = x
    vn = v
    key_it=key

    def one_step_MCMC(i,inp):
        (xn,vn,key_it)=inp
        key_it, subkey_it = jax.random.split(key_it)
        xi=random.normal(subkey_it,(4*rep,nbeta))
        xn,vn=UBU_step(xn,vn,hper2c,vmap_grad_lpost,xi)
        return xn,vn,key_it
    (xn,vn,key_it)=jax.lax.fori_loop(0, burn, one_step_MCMC, (xn,vn,key_it))
    return xn,vn,key_it
        
def singleMCMC(x,v,nbeta,niter,burn,hper2c,vmap_grad_lpost,vmap_test_function,test_dim,key):
    if(x.ndim==1):
        rep=1
    else:
        rep=x.shape[0]
    key_it=key
    xn,vn,key_it=burnMCMC(x,v,nbeta,burn,hper2c,vmap_grad_lpost,key_it)
    meanx=jnp.zeros([rep,test_dim])
    meanxsquare=jnp.zeros([rep,test_dim])

    def one_step_MCMC(i,inp):
        (xn,vn,key_it,meanx,meanxsquare)=inp
        key_it, subkey_it = jax.random.split(key_it)
        xi=random.normal(subkey_it,(4*rep,nbeta))
        xn,vn=UBU_step(xn,vn,hper2c,vmap_grad_lpost,xi)
        test_vals=vmap_test_function(xn)
        meanx=meanx+test_vals
        meanxsquare=meanxsquare+jnp.square(test_vals)
        return xn,vn,key_it,meanx,meanxsquare
    
    (xn,vn,key_it,meanx,meanxsquare)=jax.lax.fori_loop(0, niter, one_step_MCMC, (xn,vn,key_it,meanx,meanxsquare))
    
    meanx=meanx/niter
    meanxsquare=meanxsquare/niter
    return meanx,meanxsquare,key_it

def burnMCMC2(x,v,x2,v2,nbeta,burn,hper4c,vmap_grad_lpost,key):
    if(x.ndim==1):
        rep=1
    else:
        rep=x.shape[0]
   
    xn=x
    vn=v
    x2n=x2
    v2n=v2
    key_it=key

    def one_step_MCMC(i,inp):
        (xn,vn,x2n,v2n,key_it)=inp
        key_it, subkey_it = jax.random.split(key_it)
        xi=random.normal(subkey_it,(8*rep,nbeta))
        xn,vn,x2n,v2n=UBU_step2(xn,vn,x2n,v2n,hper4c,vmap_grad_lpost,xi)
        return xn,vn,x2n,v2n,key_it

    (xn,vn,x2n,v2n,key_it)=jax.lax.fori_loop(0, burn, one_step_MCMC, (xn,vn,x2n,v2n,key_it))

    return xn,vn,x2n,v2n,key_it

def doubleMCMC(x,v,x2,v2,nbeta,niter,extraburnin2, jointburnin,thin,hper4c,vmap_grad_lpost,vmap_test_function,test_dim,key):
    if(x.ndim==1):
        rep=1 
    else:
        rep=x.shape[0]

    key_it=key
    x2n,v2n,key_it=burnMCMC(x2,v2,nbeta,extraburnin2*2,hper4c,vmap_grad_lpost,key_it)
    xn,vn,x2n,v2n,key_it=burnMCMC2(x,v,x2n,v2n,nbeta,jointburnin,hper4c,vmap_grad_lpost,key_it)


    meanx=jnp.zeros([rep,test_dim])
    meanxsquare=jnp.zeros([rep,test_dim])
    nsamp=niter//thin

    def one_step_MCMC(i,inp):
        (xn,vn,x2n,v2n,key_it)=inp
        key_it, subkey_it = jax.random.split(key_it)
        xi=random.normal(subkey_it,(8*rep,nbeta))
        xn,vn,x2n,v2n=UBU_step2(xn,vn,x2n,v2n,hper4c,vmap_grad_lpost,xi)
        return xn,vn,x2n,v2n,key_it

    def thin_step_MCMC(i,inp):
        (xn,vn,x2n,v2n,key_it,meanx,meanxsquare)=inp
        (xn,vn,x2n,v2n,key_it)=jax.lax.fori_loop(0, thin, one_step_MCMC, (xn,vn,x2n,v2n,key_it))
        test_vals1=vmap_test_function(xn)
        test_vals2=vmap_test_function(x2n)       
        meanx=meanx+test_vals2-test_vals1
        meanxsquare=meanxsquare+jnp.square(test_vals2)-jnp.square(test_vals1)
        return xn,vn,x2n,v2n,key_it,meanx,meanxsquare
    
    (xn,vn,x2n,v2n,key_it,meanx,meanxsquare)=jax.lax.fori_loop(0, nsamp, thin_step_MCMC, (xn,vn,x2n,v2n,key_it,meanx,meanxsquare))
    errxnx2n=jnp.mean(jnp.abs(xn-x2n))

    meanx=meanx/nsamp
    meanxsquare=meanxsquare/nsamp
    
    return meanx,meanxsquare,key_it,errxnx2n







def multiMCMC(x,v,nbeta,niter,extraburnin2, jointburnin, thin, multilevels,maxmultilevels, vmap_grad_lpost,vmap_test_function,test_dim, key, hper2c_arr):
    

    def singlestep_UBU(xn,vn,xi,hper2c):
        xiwidth=xi.shape[0]//2
        xn,vn=multiU(xn,vn,hper2c,xi[0:xiwidth,:])
        gr=vmap_grad_lpost(xn)
        vn=vn-xiwidth*(hper2c[0])*gr
        xn,vn=multiU(xn,vn,hper2c,xi[xiwidth:(2*xiwidth),:])
        return xn,vn
        
    def multistep_UBU0(xn,vn,xi,hper2c):
        xn,vn=singlestep_UBU(xn,vn,xi,hper2c)        
        return xn,vn
    
    def loop_UBU0(xn,vn,hper2c,key_it,numiter):
        def inner0(it,inp):
            xn,vn,key_it=inp
            key_it,subkey_it=random.split(key_it)
            xi=random.normal(subkey_it,(2,2,nbeta))
            xn,vn=multistep_UBU0(xn,vn,xi,hper2c)
            return xn,vn,key_it
        xn,vn,key_it=jax.lax.fori_loop(0,numiter,inner0,(xn,vn,key_it))
        return xn,vn, key_it
       
    def multistep_UBU1(xn,vn,xi,hper2c):
        xiwidth=xi.shape[0]//2
        def inner(it, inp):
            xn,vn=inp
            xn,vn=singlestep_UBU(xn,vn,dynamic_slice(xi,[(it*xiwidth),0,0],[xiwidth,2,nbeta]),hper2c)
            return xn,vn
        xn,vn=jax.lax.fori_loop(0,2,inner,(xn,vn))
        # for it in range(2):
        #     xn,vn=singlestep_UBU(xn,vn,xi[(it*xiwidth):((it+1)*xiwidth),:],hper2c)
        return xn,vn
       
    def multistep_UBU2(xn,vn,xi,hper2c):
        xiwidth=xi.shape[0]//4
        def inner(it, inp):
            xn,vn=inp
            xn,vn=singlestep_UBU(xn,vn,dynamic_slice(xi,[(it*xiwidth),0,0],[xiwidth,2,nbeta]),hper2c)
            return xn,vn
        xn,vn=jax.lax.fori_loop(0,4,inner,(xn,vn))
        return xn,vn          
    
    def multistep_UBU3(xn,vn,xi,hper2c):
        xiwidth=xi.shape[0]//8
        def inner(it, inp):
            xn,vn=inp
            xn,vn=singlestep_UBU(xn,vn,dynamic_slice(xi,[(it*xiwidth),0,0],[xiwidth,2,nbeta]),hper2c)
            return xn,vn
        xn,vn=jax.lax.fori_loop(0,8,inner,(xn,vn))
        return xn,vn          
    
    def multistep_UBU4(xn,vn,xi,hper2c):
        xiwidth=xi.shape[0]//16
        def inner(it, inp):
            xn,vn=inp
            xn,vn=singlestep_UBU(xn,vn,dynamic_slice(xi,[(it*xiwidth),0,0],[xiwidth,2,nbeta]),hper2c)
            return xn,vn
        xn,vn=jax.lax.fori_loop(0,16,inner,(xn,vn))
        return xn,vn
    
    def multistep_UBU5(xn,vn,xi,hper2c):
        xiwidth=xi.shape[0]//32
        def inner(it, inp):
            xn,vn=inp
            xn,vn=singlestep_UBU(xn,vn,dynamic_slice(xi,[(it*xiwidth),0,0],[xiwidth,2,nbeta]),hper2c)
            return xn,vn
        xn,vn=jax.lax.fori_loop(0,32,inner,(xn,vn))
        return xn,vn    

    def multistep_UBU6(xn,vn,xi,hper2c):
        xiwidth=xi.shape[0]//64
        def inner(it, inp):
            xn,vn=inp
            xn,vn=singlestep_UBU(xn,vn,dynamic_slice(xi,[(it*xiwidth),0,0],[xiwidth,2,nbeta]),hper2c)
            return xn,vn
        xn,vn=jax.lax.fori_loop(0,64,inner,(xn,vn))
        return xn,vn    

    def multistep_UBU7(xn,vn,xi,hper2c):
        xiwidth=xi.shape[0]//128
        def inner(it, inp):
            xn,vn=inp
            xn,vn=singlestep_UBU(xn,vn,dynamic_slice(xi,[(it*xiwidth),0,0],[xiwidth,2,nbeta]),hper2c)
            return xn,vn
        xn,vn=jax.lax.fori_loop(0,128,inner,(xn,vn))
        return xn,vn            

    def multistep_UBU8(xn,vn,xi,hper2c):
        xiwidth=xi.shape[0]//256
        def inner(it, inp):
            xn,vn=inp
            xn,vn=singlestep_UBU(xn,vn,dynamic_slice(xi,[(it*xiwidth),0,0],[xiwidth,2,nbeta]),hper2c)
            return xn,vn
        xn,vn=jax.lax.fori_loop(0,256,inner,(xn,vn))
        return xn,vn  
    
    def multistep_UBU9(xn,vn,xi,hper2c):
        xiwidth=xi.shape[0]//512
        def inner(it, inp):
            xn,vn=inp
            xn,vn=singlestep_UBU(xn,vn,dynamic_slice(xi,[(it*xiwidth),0,0],[xiwidth,2,nbeta]),hper2c)
            return xn,vn
        xn,vn=jax.lax.fori_loop(0,512,inner,(xn,vn))
        return xn,vn  


    def multi0_inner(it,inp1):
        xnarr,vnarr,hper2c,key_it=inp1
        key_it,subkey_it=random.split(key_it)
        xiarr=random.normal(subkey_it,(4,2,nbeta))
        xnarr_up0,vnarr_up0=multistep_UBU0(dynamic_slice(xnarr,[0,0],[1,nbeta]),dynamic_slice(vnarr,[0,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up0,[0,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up0,[0,0])
        xnarr_up1,vnarr_up1=multistep_UBU1(dynamic_slice(xnarr,[1,0],[1,nbeta]),dynamic_slice(vnarr,[1,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up1,[1,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up1,[1,0])
        return xnarr,vnarr,hper2c,key_it
    
    def multi0_inner_thin(i,inp1):
        xnarr,vnarr,meanx,meanxsquare,hper2c,key_it=inp1
        xnarr,vnarr,_,key_it=jax.lax.fori_loop(0,thin,multi0_inner,(xnarr,vnarr,hper2c,key_it))
        test_val=vmap_test_function(xnarr)
        meanx=meanx+test_val
        meanxsquare=meanxsquare+test_val**2
        return xnarr,vnarr,meanx,meanxsquare,hper2c,key_it

    def multi0(inp):
        xnarr,vnarr,niter,extraburnin2, jointburnin,thin, key_it=inp
        xn1=dynamic_slice(xnarr,[1,0],[1,nbeta])
        vn1=dynamic_slice(vnarr,[1,0],[1,nbeta])
        
        hper2c=dynamic_slice(hper2c_arr,(1,0),[1,6]).reshape((6,))
        xn1,vn1,key_it=loop_UBU0(xn1,vn1,hper2c,key_it,extraburnin2*2)
        xnarr=dynamic_update_slice(xnarr,xn1,[1,0])
        vnarr=dynamic_update_slice(vnarr,vn1,[1,0])

        xnarr,vnarr,hper2c,key_it=jax.lax.fori_loop(0,jointburnin,multi0_inner,(xnarr,vnarr,hper2c,key_it))


        meanx_single=jnp.zeros([maxmultilevels+2,test_dim])
        meanxsquare_single=jnp.zeros([maxmultilevels+2,test_dim])
        
        xnarr,vnarr,meanx_single,meanxsquare_single,hper2c,key_it=jax.lax.fori_loop(0,niter//thin,multi0_inner_thin,(xnarr,vnarr,meanx_single,meanxsquare_single,hper2c,key_it))

        meanx_single=meanx_single/(niter//thin)
        meanxsquare_single=meanxsquare_single/(niter//thin)
        meanx_diff=meanx_single[1:(maxmultilevels+2),:]-meanx_single[0:(maxmultilevels+1),:]
        meanx_diff=dynamic_update_slice(meanx_diff,jnp.zeros([1,test_dim]),[multilevels+1,0])
        meanxsquare_diff=meanxsquare_single[1:(maxmultilevels+2),:]-meanxsquare_single[0:(maxmultilevels+1),:]
        meanxsquare_diff=dynamic_update_slice(meanxsquare_diff,jnp.zeros([1,test_dim]),[multilevels+1,0])
        err=jnp.mean(jnp.abs(dynamic_slice(xnarr,[1,0],[maxmultilevels+1,nbeta])-dynamic_slice(xnarr,[0,0],[maxmultilevels+1,nbeta])),axis=1)
        return meanx_diff,meanxsquare_diff,key_it,err    

    


    def multi1_inner(it,inp1):
        xnarr,vnarr,hper2c,key_it=inp1
        key_it,subkey_it=random.split(key_it)
        xiarr=random.normal(subkey_it,(8,2,nbeta))
        xnarr_up0,vnarr_up0=multistep_UBU0(dynamic_slice(xnarr,[0,0],[1,nbeta]),dynamic_slice(vnarr,[0,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up0,[0,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up0,[0,0])
        xnarr_up1,vnarr_up1=multistep_UBU1(dynamic_slice(xnarr,[1,0],[1,nbeta]),dynamic_slice(vnarr,[1,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up1,[1,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up1,[1,0])
        xnarr_up2,vnarr_up2=multistep_UBU2(dynamic_slice(xnarr,[2,0],[1,nbeta]),dynamic_slice(vnarr,[2,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up2,[2,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up2,[2,0])
        return xnarr,vnarr,hper2c,key_it
    
    def multi1_inner_thin(i,inp1):
        xnarr,vnarr,meanx,meanxsquare,hper2c,key_it=inp1
        xnarr,vnarr,hper2c,key_it=jax.lax.fori_loop(0,thin,multi1_inner,(xnarr,vnarr,hper2c,key_it))
        test_val=vmap_test_function(xnarr)
        meanx=meanx+test_val
        meanxsquare=meanxsquare+test_val**2
        return xnarr,vnarr,meanx,meanxsquare,hper2c,key_it

    def multi1(inp):
        xnarr,vnarr,niter,extraburnin2, jointburnin,thin, key_it=inp
        hper2c=dynamic_slice(hper2c_arr,(2,0),[1,6]).reshape((6,))

        xn2=dynamic_slice(xnarr,[2,0],[1,nbeta])
        vn2=dynamic_slice(vnarr,[2,0],[1,nbeta])
        xn2,vn2,key_it=loop_UBU0(xn2,vn2,hper2c,key_it,extraburnin2*4)
        xnarr=dynamic_update_slice(xnarr,xn2,[2,0])
        vnarr=dynamic_update_slice(vnarr,vn2,[2,0])

        xn12,vn12,_,key_it=jax.lax.fori_loop(0,extraburnin2*2,multi0_inner,(dynamic_slice(xnarr,[1,0],[2,nbeta]),dynamic_slice(vnarr,[1,0],[2,nbeta]),hper2c,key_it))
        xnarr=dynamic_update_slice(xnarr,xn12,[1,0])
        vnarr=dynamic_update_slice(vnarr,vn12,[1,0])

        xnarr,vnarr,hper2c,key_it=jax.lax.fori_loop(0,jointburnin,multi1_inner,(xnarr,vnarr,hper2c,key_it))

        meanx_single=jnp.zeros([maxmultilevels+2,test_dim])
        meanxsquare_single=jnp.zeros([maxmultilevels+2,test_dim])
        
        xnarr,vnarr,meanx_single,meanxsquare_single,hper2c,key_it=jax.lax.fori_loop(0,niter//thin,multi1_inner_thin,(xnarr,vnarr,meanx_single,meanxsquare_single,hper2c,key_it))

        meanx_single=meanx_single/(niter//thin)
        meanxsquare_single=meanxsquare_single/(niter//thin)
        meanx_diff=meanx_single[1:(maxmultilevels+2),:]-meanx_single[0:(maxmultilevels+1),:]
        meanx_diff=dynamic_update_slice(meanx_diff,jnp.zeros([1,test_dim]),[multilevels+1,0])
        meanxsquare_diff=meanxsquare_single[1:(maxmultilevels+2),:]-meanxsquare_single[0:(maxmultilevels+1),:]
        meanxsquare_diff=dynamic_update_slice(meanxsquare_diff,jnp.zeros([1,test_dim]),[multilevels+1,0])
        err=jnp.mean(jnp.abs(dynamic_slice(xnarr,[1,0],[maxmultilevels+1,nbeta])-dynamic_slice(xnarr,[0,0],[maxmultilevels+1,nbeta])),axis=1)

        return meanx_diff,meanxsquare_diff,key_it,err    


    def multi2_inner(it,inp1):
        xnarr,vnarr,hper2c,key_it=inp1
        key_it,subkey_it=random.split(key_it)
        xiarr=random.normal(subkey_it,(16,2,nbeta))
        xnarr_up0,vnarr_up0=multistep_UBU0(dynamic_slice(xnarr,[0,0],[1,nbeta]),dynamic_slice(vnarr,[0,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up0,[0,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up0,[0,0])
        xnarr_up1,vnarr_up1=multistep_UBU1(dynamic_slice(xnarr,[1,0],[1,nbeta]),dynamic_slice(vnarr,[1,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up1,[1,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up1,[1,0])
        xnarr_up2,vnarr_up2=multistep_UBU2(dynamic_slice(xnarr,[2,0],[1,nbeta]),dynamic_slice(vnarr,[2,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up2,[2,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up2,[2,0])
        xnarr_up3,vnarr_up3=multistep_UBU3(dynamic_slice(xnarr,[3,0],[1,nbeta]),dynamic_slice(vnarr,[3,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up3,[3,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up3,[3,0])
        return xnarr,vnarr,hper2c,key_it
    
    def multi2_inner_thin(i,inp1):
        xnarr,vnarr,meanx,meanxsquare,hper2c,key_it=inp1
        xnarr,vnarr,hper2c,key_it=jax.lax.fori_loop(0,thin,multi2_inner,(xnarr,vnarr,hper2c,key_it))
        test_val=vmap_test_function(xnarr)
        meanx=meanx+test_val
        meanxsquare=meanxsquare+test_val**2
        return xnarr,vnarr,meanx,meanxsquare,hper2c,key_it

    def multi2(inp):
        xnarr,vnarr,niter,extraburnin2, jointburnin,thin, key_it=inp
        hper2c=dynamic_slice(hper2c_arr,(3,0),[1,6]).reshape((6,))

        xn3=dynamic_slice(xnarr,[3,0],[1,nbeta])
        vn3=dynamic_slice(vnarr,[3,0],[1,nbeta])
        xn3,vn3,key_it=loop_UBU0(xn3,vn3,hper2c,key_it,extraburnin2*8)
        xnarr=dynamic_update_slice(xnarr,xn3,[3,0])
        vnarr=dynamic_update_slice(vnarr,vn3,[3,0])

        xn23,vn23,_,key_it=jax.lax.fori_loop(0,extraburnin2*4,multi0_inner,(dynamic_slice(xnarr,[2,0],[2,nbeta]),dynamic_slice(vnarr,[2,0],[2,nbeta]),hper2c,key_it))
        xnarr=dynamic_update_slice(xnarr,xn23,[2,0])
        vnarr=dynamic_update_slice(vnarr,vn23,[2,0])

        xn123,vn123,_,key_it=jax.lax.fori_loop(0,extraburnin2*2,multi1_inner,(dynamic_slice(xnarr,[1,0],[3,nbeta]),dynamic_slice(vnarr,[1,0],[3,nbeta]),hper2c,key_it))
        xnarr=dynamic_update_slice(xnarr,xn123,[1,0])
        vnarr=dynamic_update_slice(vnarr,vn123,[1,0])

        xnarr,vnarr,hper2c,key_it=jax.lax.fori_loop(0,jointburnin,multi2_inner,(xnarr,vnarr,hper2c,key_it))

        meanx_single=jnp.zeros([maxmultilevels+2,test_dim])
        meanxsquare_single=jnp.zeros([maxmultilevels+2,test_dim])
        
        xnarr,vnarr,meanx_single,meanxsquare_single,hper2c,key_it=jax.lax.fori_loop(0,niter//thin,multi2_inner_thin,(xnarr,vnarr,meanx_single,meanxsquare_single,hper2c,key_it))

        meanx_single=meanx_single/(niter//thin)
        meanxsquare_single=meanxsquare_single/(niter//thin)
        meanx_diff=meanx_single[1:(maxmultilevels+2),:]-meanx_single[0:(maxmultilevels+1),:]
        meanx_diff=dynamic_update_slice(meanx_diff,jnp.zeros([1,test_dim]),[multilevels+1,0])
        meanxsquare_diff=meanxsquare_single[1:(maxmultilevels+2),:]-meanxsquare_single[0:(maxmultilevels+1),:]
        meanxsquare_diff=dynamic_update_slice(meanxsquare_diff,jnp.zeros([1,test_dim]),[multilevels+1,0])
        err=jnp.mean(jnp.abs(dynamic_slice(xnarr,[1,0],[maxmultilevels+1,nbeta])-dynamic_slice(xnarr,[0,0],[maxmultilevels+1,nbeta])),axis=1)

        return meanx_diff,meanxsquare_diff,key_it,err    

    
    def multi3_inner(it,inp1):
        xnarr,vnarr,hper2c,key_it=inp1
        key_it,subkey_it=random.split(key_it)
        xiarr=random.normal(subkey_it,(32,2,nbeta))
        xnarr_up0,vnarr_up0=multistep_UBU0(dynamic_slice(xnarr,[0,0],[1,nbeta]),dynamic_slice(vnarr,[0,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up0,[0,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up0,[0,0])
        xnarr_up1,vnarr_up1=multistep_UBU1(dynamic_slice(xnarr,[1,0],[1,nbeta]),dynamic_slice(vnarr,[1,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up1,[1,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up1,[1,0])
        xnarr_up2,vnarr_up2=multistep_UBU2(dynamic_slice(xnarr,[2,0],[1,nbeta]),dynamic_slice(vnarr,[2,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up2,[2,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up2,[2,0])
        xnarr_up3,vnarr_up3=multistep_UBU3(dynamic_slice(xnarr,[3,0],[1,nbeta]),dynamic_slice(vnarr,[3,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up3,[3,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up3,[3,0])
        xnarr_up4,vnarr_up4=multistep_UBU4(dynamic_slice(xnarr,[4,0],[1,nbeta]),dynamic_slice(vnarr,[4,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up4,[4,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up4,[4,0])
        return xnarr,vnarr,hper2c,key_it
    
    def multi3_inner_thin(i,inp1):
        xnarr,vnarr,meanx,meanxsquare,hper2c,key_it=inp1
        xnarr,vnarr,hper2c,key_it=jax.lax.fori_loop(0,thin,multi3_inner,(xnarr,vnarr,hper2c,key_it))
        test_val=vmap_test_function(xnarr)
        meanx=meanx+test_val
        meanxsquare=meanxsquare+test_val**2
        return xnarr,vnarr,meanx,meanxsquare,hper2c,key_it

    def multi3(inp):
        xnarr,vnarr,niter,extraburnin2, jointburnin,thin, key_it=inp
        hper2c=dynamic_slice(hper2c_arr,(4,0),[1,6]).reshape((6,))

        xn4=dynamic_slice(xnarr,[4,0],[1,nbeta])
        vn4=dynamic_slice(vnarr,[4,0],[1,nbeta])
        xn4,vn4,key_it=loop_UBU0(xn4,vn4,hper2c,key_it,extraburnin2*16)
        xnarr=dynamic_update_slice(xnarr,xn4,[4,0])
        vnarr=dynamic_update_slice(vnarr,vn4,[4,0])

        xn34,vn34,_,key_it=jax.lax.fori_loop(0,extraburnin2*8,multi0_inner,(dynamic_slice(xnarr,[3,0],[2,nbeta]),dynamic_slice(vnarr,[3,0],[2,nbeta]),hper2c,key_it))
        xnarr=dynamic_update_slice(xnarr,xn34,[3,0])
        vnarr=dynamic_update_slice(vnarr,vn34,[3,0])

        xn234,vn234,_,key_it=jax.lax.fori_loop(0,extraburnin2*4,multi1_inner,(dynamic_slice(xnarr,[2,0],[3,nbeta]),dynamic_slice(vnarr,[2,0],[3,nbeta]),hper2c,key_it))
        xnarr=dynamic_update_slice(xnarr,xn234,[2,0])
        vnarr=dynamic_update_slice(vnarr,vn234,[2,0])

        xn1234,vn1234,_,key_it=jax.lax.fori_loop(0,extraburnin2*2,multi2_inner,(dynamic_slice(xnarr,[1,0],[4,nbeta]),dynamic_slice(vnarr,[1,0],[4,nbeta]),hper2c,key_it))
        xnarr=dynamic_update_slice(xnarr,xn1234,[1,0])
        vnarr=dynamic_update_slice(vnarr,vn1234,[1,0])


        xnarr,vnarr,hper2c,key_it=jax.lax.fori_loop(0,jointburnin,multi3_inner,(xnarr,vnarr,hper2c,key_it))

        meanx_single=jnp.zeros([maxmultilevels+2,test_dim])
        meanxsquare_single=jnp.zeros([maxmultilevels+2,test_dim])
        
        xnarr,vnarr,meanx_single,meanxsquare_single,hper2c,key_it=jax.lax.fori_loop(0,niter//thin,multi3_inner_thin,(xnarr,vnarr,meanx_single,meanxsquare_single,hper2c,key_it))

        meanx_single=meanx_single/(niter//thin)
        meanxsquare_single=meanxsquare_single/(niter//thin)
        meanx_diff=meanx_single[1:(maxmultilevels+2),:]-meanx_single[0:(maxmultilevels+1),:]
        meanx_diff=dynamic_update_slice(meanx_diff,jnp.zeros([1,test_dim]),[multilevels+1,0])
        meanxsquare_diff=meanxsquare_single[1:(maxmultilevels+2),:]-meanxsquare_single[0:(maxmultilevels+1),:]
        meanxsquare_diff=dynamic_update_slice(meanxsquare_diff,jnp.zeros([1,test_dim]),[multilevels+1,0])
        err=jnp.mean(jnp.abs(dynamic_slice(xnarr,[1,0],[maxmultilevels+1,nbeta])-dynamic_slice(xnarr,[0,0],[maxmultilevels+1,nbeta])),axis=1)

        return meanx_diff,meanxsquare_diff,key_it,err    
    
    def multi4_inner(it,inp1):
        xnarr,vnarr,hper2c,key_it=inp1
        key_it,subkey_it=random.split(key_it)
        xiarr=random.normal(subkey_it,(64,2,nbeta))
        xnarr_up0,vnarr_up0=multistep_UBU0(dynamic_slice(xnarr,[0,0],[1,nbeta]),dynamic_slice(vnarr,[0,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up0,[0,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up0,[0,0])
        xnarr_up1,vnarr_up1=multistep_UBU1(dynamic_slice(xnarr,[1,0],[1,nbeta]),dynamic_slice(vnarr,[1,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up1,[1,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up1,[1,0])
        xnarr_up2,vnarr_up2=multistep_UBU2(dynamic_slice(xnarr,[2,0],[1,nbeta]),dynamic_slice(vnarr,[2,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up2,[2,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up2,[2,0])
        xnarr_up3,vnarr_up3=multistep_UBU3(dynamic_slice(xnarr,[3,0],[1,nbeta]),dynamic_slice(vnarr,[3,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up3,[3,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up3,[3,0])
        xnarr_up4,vnarr_up4=multistep_UBU4(dynamic_slice(xnarr,[4,0],[1,nbeta]),dynamic_slice(vnarr,[4,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up4,[4,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up4,[4,0])
        xnarr_up5,vnarr_up5=multistep_UBU5(dynamic_slice(xnarr,[5,0],[1,nbeta]),dynamic_slice(vnarr,[5,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up5,[5,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up5,[5,0])        
        return xnarr,vnarr,hper2c,key_it
    
    def multi4_inner_thin(i,inp1):
        xnarr,vnarr,meanx,meanxsquare,hper2c,key_it=inp1
        xnarr,vnarr,hper2c,key_it=jax.lax.fori_loop(0,thin,multi4_inner,(xnarr,vnarr,hper2c,key_it))
        test_val=vmap_test_function(xnarr)
        meanx=meanx+test_val
        meanxsquare=meanxsquare+test_val**2
        return xnarr,vnarr,meanx,meanxsquare,hper2c,key_it

    
    def multi4(inp):
        xnarr,vnarr,niter,extraburnin2, jointburnin,thin, key_it=inp
        hper2c=dynamic_slice(hper2c_arr,(5,0),[1,6]).reshape((6,))

        xn5=dynamic_slice(xnarr,[5,0],[1,nbeta])
        vn5=dynamic_slice(vnarr,[5,0],[1,nbeta])
        xn5,vn5,key_it=loop_UBU0(xn5,vn5,hper2c,key_it,extraburnin2*32)
        xnarr=dynamic_update_slice(xnarr,xn5,[5,0])
        vnarr=dynamic_update_slice(vnarr,vn5,[5,0])

        xn45,vn45,_,key_it=jax.lax.fori_loop(0,extraburnin2*16,multi0_inner,(dynamic_slice(xnarr,[4,0],[2,nbeta]),dynamic_slice(vnarr,[4,0],[2,nbeta]),hper2c,key_it))
        xnarr=dynamic_update_slice(xnarr,xn45,[4,0])
        vnarr=dynamic_update_slice(vnarr,vn45,[4,0])

        xn345,vn345,_,key_it=jax.lax.fori_loop(0,extraburnin2*8,multi1_inner,(dynamic_slice(xnarr,[3,0],[3,nbeta]),dynamic_slice(vnarr,[3,0],[3,nbeta]),hper2c,key_it))
        xnarr=dynamic_update_slice(xnarr,xn345,[3,0])
        vnarr=dynamic_update_slice(vnarr,vn345,[3,0])

        xn2345,vn2345,_,key_it=jax.lax.fori_loop(0,extraburnin2*4,multi2_inner,(dynamic_slice(xnarr,[2,0],[4,nbeta]),dynamic_slice(vnarr,[2,0],[4,nbeta]),hper2c,key_it))
        xnarr=dynamic_update_slice(xnarr,xn2345,[2,0])
        vnarr=dynamic_update_slice(vnarr,vn2345,[2,0])

        xn12345,vn12345,_,key_it=jax.lax.fori_loop(0,extraburnin2*2,multi3_inner,(dynamic_slice(xnarr,[1,0],[5,nbeta]),dynamic_slice(vnarr,[1,0],[5,nbeta]),hper2c,key_it))
        xnarr=dynamic_update_slice(xnarr,xn12345,[1,0])
        vnarr=dynamic_update_slice(vnarr,vn12345,[1,0])


        xnarr,vnarr,hper2c,key_it=jax.lax.fori_loop(0,jointburnin,multi4_inner,(xnarr,vnarr,hper2c,key_it))

        meanx_single=jnp.zeros([maxmultilevels+2,test_dim])
        meanxsquare_single=jnp.zeros([maxmultilevels+2,test_dim])
        
        xnarr,vnarr,meanx_single,meanxsquare_single,hper2c,key_it=jax.lax.fori_loop(0,niter//thin,multi4_inner_thin,(xnarr,vnarr,meanx_single,meanxsquare_single,hper2c,key_it))


        meanx_single=meanx_single/(niter//thin)
        meanxsquare_single=meanxsquare_single/(niter//thin)
        meanx_diff=meanx_single[1:(maxmultilevels+2),:]-meanx_single[0:(maxmultilevels+1),:]
        meanx_diff=dynamic_update_slice(meanx_diff,jnp.zeros([1,test_dim]),[multilevels+1,0])
        meanxsquare_diff=meanxsquare_single[1:(maxmultilevels+2),:]-meanxsquare_single[0:(maxmultilevels+1),:]
        meanxsquare_diff=dynamic_update_slice(meanxsquare_diff,jnp.zeros([1,test_dim]),[multilevels+1,0])
        err=jnp.mean(jnp.abs(dynamic_slice(xnarr,[1,0],[maxmultilevels+1,nbeta])-dynamic_slice(xnarr,[0,0],[maxmultilevels+1,nbeta])),axis=1)

        return meanx_diff,meanxsquare_diff,key_it,err

    def multi5_inner(it,inp1):
        xnarr,vnarr,hper2c,key_it=inp1
        key_it,subkey_it=random.split(key_it)
        xiarr=random.normal(subkey_it,(128,2,nbeta)) # 2^(5+2) = 128
        xnarr_up0,vnarr_up0=multistep_UBU0(dynamic_slice(xnarr,[0,0],[1,nbeta]),dynamic_slice(vnarr,[0,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up0,[0,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up0,[0,0])
        xnarr_up1,vnarr_up1=multistep_UBU1(dynamic_slice(xnarr,[1,0],[1,nbeta]),dynamic_slice(vnarr,[1,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up1,[1,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up1,[1,0])
        xnarr_up2,vnarr_up2=multistep_UBU2(dynamic_slice(xnarr,[2,0],[1,nbeta]),dynamic_slice(vnarr,[2,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up2,[2,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up2,[2,0])
        xnarr_up3,vnarr_up3=multistep_UBU3(dynamic_slice(xnarr,[3,0],[1,nbeta]),dynamic_slice(vnarr,[3,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up3,[3,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up3,[3,0])
        xnarr_up4,vnarr_up4=multistep_UBU4(dynamic_slice(xnarr,[4,0],[1,nbeta]),dynamic_slice(vnarr,[4,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up4,[4,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up4,[4,0])
        xnarr_up5,vnarr_up5=multistep_UBU5(dynamic_slice(xnarr,[5,0],[1,nbeta]),dynamic_slice(vnarr,[5,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up5,[5,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up5,[5,0])
        xnarr_up6,vnarr_up6=multistep_UBU6(dynamic_slice(xnarr,[6,0],[1,nbeta]),dynamic_slice(vnarr,[6,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up6,[6,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up6,[6,0])
        return xnarr,vnarr,hper2c,key_it
    
    def multi5_inner_thin(i,inp1):
        xnarr,vnarr,meanx,meanxsquare,hper2c,key_it=inp1
        xnarr,vnarr,hper2c,key_it=jax.lax.fori_loop(0,thin,multi5_inner,(xnarr,vnarr,hper2c,key_it))
        test_val=vmap_test_function(xnarr)
        meanx=meanx+test_val
        meanxsquare=meanxsquare+test_val**2
        return xnarr,vnarr,meanx,meanxsquare,hper2c,key_it

    
    def multi5(inp):
        xnarr,vnarr,niter,extraburnin2, jointburnin,thin, key_it=inp
        hper2c=dynamic_slice(hper2c_arr,(6,0),[1,6]).reshape((6,)) # (L+1, 0) -> (5+1, 0)

        xn6=dynamic_slice(xnarr,[6,0],[1,nbeta]) # L+1 -> 6
        vn6=dynamic_slice(vnarr,[6,0],[1,nbeta]) # L+1 -> 6
        xn6,vn6,key_it=loop_UBU0(xn6,vn6,hper2c,key_it,extraburnin2*64) # 2^(L+1) -> 2^6 = 64
        xnarr=dynamic_update_slice(xnarr,xn6,[6,0]) # L+1 -> 6
        vnarr=dynamic_update_slice(vnarr,vn6,[6,0]) # L+1 -> 6

        xn56,vn56,_,key_it=jax.lax.fori_loop(0,extraburnin2*32,multi0_inner,(dynamic_slice(xnarr,[5,0],[2,nbeta]),dynamic_slice(vnarr,[5,0],[2,nbeta]),hper2c,key_it))
        xnarr=dynamic_update_slice(xnarr,xn56,[5,0])
        vnarr=dynamic_update_slice(vnarr,vn56,[5,0])

        xn456,vn456,_,key_it=jax.lax.fori_loop(0,extraburnin2*16,multi1_inner,(dynamic_slice(xnarr,[4,0],[3,nbeta]),dynamic_slice(vnarr,[4,0],[3,nbeta]),hper2c,key_it))
        xnarr=dynamic_update_slice(xnarr,xn456,[4,0])
        vnarr=dynamic_update_slice(vnarr,vn456,[4,0])

        xn3456,vn3456,_,key_it=jax.lax.fori_loop(0,extraburnin2*8,multi2_inner,(dynamic_slice(xnarr,[3,0],[4,nbeta]),dynamic_slice(vnarr,[3,0],[4,nbeta]),hper2c,key_it))
        xnarr=dynamic_update_slice(xnarr,xn3456,[3,0])
        vnarr=dynamic_update_slice(vnarr,vn3456,[3,0])

        xn23456,vn23456,_,key_it=jax.lax.fori_loop(0,extraburnin2*4,multi3_inner,(dynamic_slice(xnarr,[2,0],[5,nbeta]),dynamic_slice(vnarr,[2,0],[5,nbeta]),hper2c,key_it))
        xnarr=dynamic_update_slice(xnarr,xn23456,[2,0])
        vnarr=dynamic_update_slice(vnarr,vn23456,[2,0])

        xn123456,vn123456,_,key_it=jax.lax.fori_loop(0,extraburnin2*2,multi4_inner,(dynamic_slice(xnarr,[1,0],[6,nbeta]),dynamic_slice(vnarr,[1,0],[6,nbeta]),hper2c,key_it))
        xnarr=dynamic_update_slice(xnarr,xn123456,[1,0])
        vnarr=dynamic_update_slice(vnarr,vn123456,[1,0])


        xnarr,vnarr,hper2c,key_it=jax.lax.fori_loop(0,jointburnin,multi5_inner,(xnarr,vnarr,hper2c,key_it))

        meanx_single=jnp.zeros([maxmultilevels+2,test_dim])
        meanxsquare_single=jnp.zeros([maxmultilevels+2,test_dim])
        
        xnarr,vnarr,meanx_single,meanxsquare_single,hper2c,key_it=jax.lax.fori_loop(0,niter//thin,multi5_inner_thin,(xnarr,vnarr,meanx_single,meanxsquare_single,hper2c,key_it))


        meanx_single=meanx_single/(niter//thin)
        meanxsquare_single=meanxsquare_single/(niter//thin)
        meanx_diff=meanx_single[1:(maxmultilevels+2),:]-meanx_single[0:(maxmultilevels+1),:]
        meanx_diff=dynamic_update_slice(meanx_diff,jnp.zeros([1,test_dim]),[multilevels+1,0])
        meanxsquare_diff=meanxsquare_single[1:(maxmultilevels+2),:]-meanxsquare_single[0:(maxmultilevels+1),:]
        meanxsquare_diff=dynamic_update_slice(meanxsquare_diff,jnp.zeros([1,test_dim]),[multilevels+1,0])
        err=jnp.mean(jnp.abs(dynamic_slice(xnarr,[1,0],[maxmultilevels+1,nbeta])-dynamic_slice(xnarr,[0,0],[maxmultilevels+1,nbeta])),axis=1)

        return meanx_diff,meanxsquare_diff,key_it,err

    # -----------------------------------------------------------------

    def multi6_inner(it,inp1):
        xnarr,vnarr,hper2c,key_it=inp1
        key_it,subkey_it=random.split(key_it)
        xiarr=random.normal(subkey_it,(256,2,nbeta)) # 2^(6+2) = 256
        xnarr_up0,vnarr_up0=multistep_UBU0(dynamic_slice(xnarr,[0,0],[1,nbeta]),dynamic_slice(vnarr,[0,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up0,[0,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up0,[0,0])
        xnarr_up1,vnarr_up1=multistep_UBU1(dynamic_slice(xnarr,[1,0],[1,nbeta]),dynamic_slice(vnarr,[1,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up1,[1,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up1,[1,0])
        xnarr_up2,vnarr_up2=multistep_UBU2(dynamic_slice(xnarr,[2,0],[1,nbeta]),dynamic_slice(vnarr,[2,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up2,[2,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up2,[2,0])
        xnarr_up3,vnarr_up3=multistep_UBU3(dynamic_slice(xnarr,[3,0],[1,nbeta]),dynamic_slice(vnarr,[3,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up3,[3,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up3,[3,0])
        xnarr_up4,vnarr_up4=multistep_UBU4(dynamic_slice(xnarr,[4,0],[1,nbeta]),dynamic_slice(vnarr,[4,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up4,[4,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up4,[4,0])
        xnarr_up5,vnarr_up5=multistep_UBU5(dynamic_slice(xnarr,[5,0],[1,nbeta]),dynamic_slice(vnarr,[5,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up5,[5,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up5,[5,0])
        xnarr_up6,vnarr_up6=multistep_UBU6(dynamic_slice(xnarr,[6,0],[1,nbeta]),dynamic_slice(vnarr,[6,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up6,[6,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up6,[6,0])
        xnarr_up7,vnarr_up7=multistep_UBU7(dynamic_slice(xnarr,[7,0],[1,nbeta]),dynamic_slice(vnarr,[7,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up7,[7,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up7,[7,0])
        return xnarr,vnarr,hper2c,key_it
    
    def multi6_inner_thin(i,inp1):
        xnarr,vnarr,meanx,meanxsquare,hper2c,key_it=inp1
        xnarr,vnarr,hper2c,key_it=jax.lax.fori_loop(0,thin,multi6_inner,(xnarr,vnarr,hper2c,key_it))
        test_val=vmap_test_function(xnarr)
        meanx=meanx+test_val
        meanxsquare=meanxsquare+test_val**2
        return xnarr,vnarr,meanx,meanxsquare,hper2c,key_it

    
    def multi6(inp):
        xnarr,vnarr,niter,extraburnin2, jointburnin,thin, key_it=inp
        hper2c=dynamic_slice(hper2c_arr,(7,0),[1,6]).reshape((6,)) # (L+1, 0) -> (6+1, 0)

        xn7=dynamic_slice(xnarr,[7,0],[1,nbeta]) # L+1 -> 7
        vn7=dynamic_slice(vnarr,[7,0],[1,nbeta]) # L+1 -> 7
        xn7,vn7,key_it=loop_UBU0(xn7,vn7,hper2c,key_it,extraburnin2*128) # 2^(L+1) -> 2^7 = 128
        xnarr=dynamic_update_slice(xnarr,xn7,[7,0]) # L+1 -> 7
        vnarr=dynamic_update_slice(vnarr,vn7,[7,0]) # L+1 -> 7

        xn67,vn67,_,key_it=jax.lax.fori_loop(0,extraburnin2*64,multi0_inner,(dynamic_slice(xnarr,[6,0],[2,nbeta]),dynamic_slice(vnarr,[6,0],[2,nbeta]),hper2c,key_it))
        xnarr=dynamic_update_slice(xnarr,xn67,[6,0])
        vnarr=dynamic_update_slice(vnarr,vn67,[6,0])

        xn567,vn567,_,key_it=jax.lax.fori_loop(0,extraburnin2*32,multi1_inner,(dynamic_slice(xnarr,[5,0],[3,nbeta]),dynamic_slice(vnarr,[5,0],[3,nbeta]),hper2c,key_it))
        xnarr=dynamic_update_slice(xnarr,xn567,[5,0])
        vnarr=dynamic_update_slice(vnarr,vn567,[5,0])

        xn4567,vn4567,_,key_it=jax.lax.fori_loop(0,extraburnin2*16,multi2_inner,(dynamic_slice(xnarr,[4,0],[4,nbeta]),dynamic_slice(vnarr,[4,0],[4,nbeta]),hper2c,key_it))
        xnarr=dynamic_update_slice(xnarr,xn4567,[4,0])
        vnarr=dynamic_update_slice(vnarr,vn4567,[4,0])

        xn34567,vn34567,_,key_it=jax.lax.fori_loop(0,extraburnin2*8,multi3_inner,(dynamic_slice(xnarr,[3,0],[5,nbeta]),dynamic_slice(vnarr,[3,0],[5,nbeta]),hper2c,key_it))
        xnarr=dynamic_update_slice(xnarr,xn34567,[3,0])
        vnarr=dynamic_update_slice(vnarr,vn34567,[3,0])

        xn234567,vn234567,_,key_it=jax.lax.fori_loop(0,extraburnin2*4,multi4_inner,(dynamic_slice(xnarr,[2,0],[6,nbeta]),dynamic_slice(vnarr,[2,0],[6,nbeta]),hper2c,key_it))
        xnarr=dynamic_update_slice(xnarr,xn234567,[2,0])
        vnarr=dynamic_update_slice(vnarr,vn234567,[2,0])

        xn1234567,vn1234567,_,key_it=jax.lax.fori_loop(0,extraburnin2*2,multi5_inner,(dynamic_slice(xnarr,[1,0],[7,nbeta]),dynamic_slice(vnarr,[1,0],[7,nbeta]),hper2c,key_it))
        xnarr=dynamic_update_slice(xnarr,xn1234567,[1,0])
        vnarr=dynamic_update_slice(vnarr,vn1234567,[1,0])


        xnarr,vnarr,hper2c,key_it=jax.lax.fori_loop(0,jointburnin,multi6_inner,(xnarr,vnarr,hper2c,key_it))

        meanx_single=jnp.zeros([maxmultilevels+2,test_dim])
        meanxsquare_single=jnp.zeros([maxmultilevels+2,test_dim])
        
        xnarr,vnarr,meanx_single,meanxsquare_single,hper2c,key_it=jax.lax.fori_loop(0,niter//thin,multi6_inner_thin,(xnarr,vnarr,meanx_single,meanxsquare_single,hper2c,key_it))


        meanx_single=meanx_single/(niter//thin)
        meanxsquare_single=meanxsquare_single/(niter//thin)
        meanx_diff=meanx_single[1:(maxmultilevels+2),:]-meanx_single[0:(maxmultilevels+1),:]
        meanx_diff=dynamic_update_slice(meanx_diff,jnp.zeros([1,test_dim]),[multilevels+1,0])
        meanxsquare_diff=meanxsquare_single[1:(maxmultilevels+2),:]-meanxsquare_single[0:(maxmultilevels+1),:]
        meanxsquare_diff=dynamic_update_slice(meanxsquare_diff,jnp.zeros([1,test_dim]),[multilevels+1,0])
        err=jnp.mean(jnp.abs(dynamic_slice(xnarr,[1,0],[maxmultilevels+1,nbeta])-dynamic_slice(xnarr,[0,0],[maxmultilevels+1,nbeta])),axis=1)

        return meanx_diff,meanxsquare_diff,key_it,err

    # -----------------------------------------------------------------

    def multi7_inner(it,inp1):
        xnarr,vnarr,hper2c,key_it=inp1
        key_it,subkey_it=random.split(key_it)
        xiarr=random.normal(subkey_it,(512,2,nbeta)) # 2^(7+2) = 512
        xnarr_up0,vnarr_up0=multistep_UBU0(dynamic_slice(xnarr,[0,0],[1,nbeta]),dynamic_slice(vnarr,[0,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up0,[0,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up0,[0,0])
        xnarr_up1,vnarr_up1=multistep_UBU1(dynamic_slice(xnarr,[1,0],[1,nbeta]),dynamic_slice(vnarr,[1,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up1,[1,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up1,[1,0])
        xnarr_up2,vnarr_up2=multistep_UBU2(dynamic_slice(xnarr,[2,0],[1,nbeta]),dynamic_slice(vnarr,[2,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up2,[2,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up2,[2,0])
        xnarr_up3,vnarr_up3=multistep_UBU3(dynamic_slice(xnarr,[3,0],[1,nbeta]),dynamic_slice(vnarr,[3,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up3,[3,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up3,[3,0])
        xnarr_up4,vnarr_up4=multistep_UBU4(dynamic_slice(xnarr,[4,0],[1,nbeta]),dynamic_slice(vnarr,[4,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up4,[4,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up4,[4,0])
        xnarr_up5,vnarr_up5=multistep_UBU5(dynamic_slice(xnarr,[5,0],[1,nbeta]),dynamic_slice(vnarr,[5,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up5,[5,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up5,[5,0])
        xnarr_up6,vnarr_up6=multistep_UBU6(dynamic_slice(xnarr,[6,0],[1,nbeta]),dynamic_slice(vnarr,[6,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up6,[6,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up6,[6,0])
        xnarr_up7,vnarr_up7=multistep_UBU7(dynamic_slice(xnarr,[7,0],[1,nbeta]),dynamic_slice(vnarr,[7,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up7,[7,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up7,[7,0])
        xnarr_up8,vnarr_up8=multistep_UBU8(dynamic_slice(xnarr,[8,0],[1,nbeta]),dynamic_slice(vnarr,[8,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up8,[8,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up8,[8,0])
        return xnarr,vnarr,hper2c,key_it
    
    def multi7_inner_thin(i,inp1):
        xnarr,vnarr,meanx,meanxsquare,hper2c,key_it=inp1
        xnarr,vnarr,hper2c,key_it=jax.lax.fori_loop(0,thin,multi7_inner,(xnarr,vnarr,hper2c,key_it))
        test_val=vmap_test_function(xnarr)
        meanx=meanx+test_val
        meanxsquare=meanxsquare+test_val**2
        return xnarr,vnarr,meanx,meanxsquare,hper2c,key_it

    
    def multi7(inp):
        xnarr,vnarr,niter,extraburnin2, jointburnin,thin, key_it=inp
        hper2c=dynamic_slice(hper2c_arr,(8,0),[1,6]).reshape((6,)) # (L+1, 0) -> (7+1, 0)

        xn8=dynamic_slice(xnarr,[8,0],[1,nbeta]) # L+1 -> 8
        vn8=dynamic_slice(vnarr,[8,0],[1,nbeta]) # L+1 -> 8
        xn8,vn8,key_it=loop_UBU0(xn8,vn8,hper2c,key_it,extraburnin2*256) # 2^(L+1) -> 2^8 = 256
        xnarr=dynamic_update_slice(xnarr,xn8,[8,0]) # L+1 -> 8
        vnarr=dynamic_update_slice(vnarr,vn8,[8,0]) # L+1 -> 8

        xn78,vn78,_,key_it=jax.lax.fori_loop(0,extraburnin2*128,multi0_inner,(dynamic_slice(xnarr,[7,0],[2,nbeta]),dynamic_slice(vnarr,[7,0],[2,nbeta]),hper2c,key_it))
        xnarr=dynamic_update_slice(xnarr,xn78,[7,0])
        vnarr=dynamic_update_slice(vnarr,vn78,[7,0])

        xn678,vn678,_,key_it=jax.lax.fori_loop(0,extraburnin2*64,multi1_inner,(dynamic_slice(xnarr,[6,0],[3,nbeta]),dynamic_slice(vnarr,[6,0],[3,nbeta]),hper2c,key_it))
        xnarr=dynamic_update_slice(xnarr,xn678,[6,0])
        vnarr=dynamic_update_slice(vnarr,vn678,[6,0])

        xn5678,vn5678,_,key_it=jax.lax.fori_loop(0,extraburnin2*32,multi2_inner,(dynamic_slice(xnarr,[5,0],[4,nbeta]),dynamic_slice(vnarr,[5,0],[4,nbeta]),hper2c,key_it))
        xnarr=dynamic_update_slice(xnarr,xn5678,[5,0])
        vnarr=dynamic_update_slice(vnarr,vn5678,[5,0])

        xn45678,vn45678,_,key_it=jax.lax.fori_loop(0,extraburnin2*16,multi3_inner,(dynamic_slice(xnarr,[4,0],[5,nbeta]),dynamic_slice(vnarr,[4,0],[5,nbeta]),hper2c,key_it))
        xnarr=dynamic_update_slice(xnarr,xn45678,[4,0])
        vnarr=dynamic_update_slice(vnarr,vn45678,[4,0])

        xn345678,vn345678,_,key_it=jax.lax.fori_loop(0,extraburnin2*8,multi4_inner,(dynamic_slice(xnarr,[3,0],[6,nbeta]),dynamic_slice(vnarr,[3,0],[6,nbeta]),hper2c,key_it))
        xnarr=dynamic_update_slice(xnarr,xn345678,[3,0])
        vnarr=dynamic_update_slice(vnarr,vn345678,[3,0])

        xn2345678,vn2345678,_,key_it=jax.lax.fori_loop(0,extraburnin2*4,multi5_inner,(dynamic_slice(xnarr,[2,0],[7,nbeta]),dynamic_slice(vnarr,[2,0],[7,nbeta]),hper2c,key_it))
        xnarr=dynamic_update_slice(xnarr,xn2345678,[2,0])
        vnarr=dynamic_update_slice(vnarr,vn2345678,[2,0])

        xn12345678,vn12345678,_,key_it=jax.lax.fori_loop(0,extraburnin2*2,multi6_inner,(dynamic_slice(xnarr,[1,0],[8,nbeta]),dynamic_slice(vnarr,[1,0],[8,nbeta]),hper2c,key_it))
        xnarr=dynamic_update_slice(xnarr,xn12345678,[1,0])
        vnarr=dynamic_update_slice(vnarr,vn12345678,[1,0])


        xnarr,vnarr,hper2c,key_it=jax.lax.fori_loop(0,jointburnin,multi7_inner,(xnarr,vnarr,hper2c,key_it))

        meanx_single=jnp.zeros([maxmultilevels+2,test_dim])
        meanxsquare_single=jnp.zeros([maxmultilevels+2,test_dim])
        
        xnarr,vnarr,meanx_single,meanxsquare_single,hper2c,key_it=jax.lax.fori_loop(0,niter//thin,multi7_inner_thin,(xnarr,vnarr,meanx_single,meanxsquare_single,hper2c,key_it))


        meanx_single=meanx_single/(niter//thin)
        meanxsquare_single=meanxsquare_single/(niter//thin)
        meanx_diff=meanx_single[1:(maxmultilevels+2),:]-meanx_single[0:(maxmultilevels+1),:]
        meanx_diff=dynamic_update_slice(meanx_diff,jnp.zeros([1,test_dim]),[multilevels+1,0])
        meanxsquare_diff=meanxsquare_single[1:(maxmultilevels+2),:]-meanxsquare_single[0:(maxmultilevels+1),:]
        meanxsquare_diff=dynamic_update_slice(meanxsquare_diff,jnp.zeros([1,test_dim]),[multilevels+1,0])
        err=jnp.mean(jnp.abs(dynamic_slice(xnarr,[1,0],[maxmultilevels+1,nbeta])-dynamic_slice(xnarr,[0,0],[maxmultilevels+1,nbeta])),axis=1)

        return meanx_diff,meanxsquare_diff,key_it,err

    # -----------------------------------------------------------------

    def multi8_inner(it,inp1):
        xnarr,vnarr,hper2c,key_it=inp1
        key_it,subkey_it=random.split(key_it)
        xiarr=random.normal(subkey_it,(1024,2,nbeta)) # 2^(8+2) = 1024
        xnarr_up0,vnarr_up0=multistep_UBU0(dynamic_slice(xnarr,[0,0],[1,nbeta]),dynamic_slice(vnarr,[0,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up0,[0,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up0,[0,0])
        xnarr_up1,vnarr_up1=multistep_UBU1(dynamic_slice(xnarr,[1,0],[1,nbeta]),dynamic_slice(vnarr,[1,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up1,[1,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up1,[1,0])
        xnarr_up2,vnarr_up2=multistep_UBU2(dynamic_slice(xnarr,[2,0],[1,nbeta]),dynamic_slice(vnarr,[2,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up2,[2,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up2,[2,0])
        xnarr_up3,vnarr_up3=multistep_UBU3(dynamic_slice(xnarr,[3,0],[1,nbeta]),dynamic_slice(vnarr,[3,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up3,[3,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up3,[3,0])
        xnarr_up4,vnarr_up4=multistep_UBU4(dynamic_slice(xnarr,[4,0],[1,nbeta]),dynamic_slice(vnarr,[4,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up4,[4,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up4,[4,0])
        xnarr_up5,vnarr_up5=multistep_UBU5(dynamic_slice(xnarr,[5,0],[1,nbeta]),dynamic_slice(vnarr,[5,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up5,[5,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up5,[5,0])
        xnarr_up6,vnarr_up6=multistep_UBU6(dynamic_slice(xnarr,[6,0],[1,nbeta]),dynamic_slice(vnarr,[6,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up6,[6,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up6,[6,0])
        xnarr_up7,vnarr_up7=multistep_UBU7(dynamic_slice(xnarr,[7,0],[1,nbeta]),dynamic_slice(vnarr,[7,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up7,[7,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up7,[7,0])
        xnarr_up8,vnarr_up8=multistep_UBU8(dynamic_slice(xnarr,[8,0],[1,nbeta]),dynamic_slice(vnarr,[8,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up8,[8,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up8,[8,0])
        xnarr_up9,vnarr_up9=multistep_UBU9(dynamic_slice(xnarr,[9,0],[1,nbeta]),dynamic_slice(vnarr,[9,0],[1,nbeta]),xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up9,[9,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up9,[9,0])
        return xnarr,vnarr,hper2c,key_it
    
    def multi8_inner_thin(i,inp1):
        xnarr,vnarr,meanx,meanxsquare,hper2c,key_it=inp1
        xnarr,vnarr,hper2c,key_it=jax.lax.fori_loop(0,thin,multi8_inner,(xnarr,vnarr,hper2c,key_it))
        test_val=vmap_test_function(xnarr)
        meanx=meanx+test_val
        meanxsquare=meanxsquare+test_val**2
        return xnarr,vnarr,meanx,meanxsquare,hper2c,key_it

    
    def multi8(inp):
        xnarr,vnarr,niter,extraburnin2, jointburnin,thin, key_it=inp
        hper2c=dynamic_slice(hper2c_arr,(9,0),[1,6]).reshape((6,)) # (L+1, 0) -> (8+1, 0)

        xn9=dynamic_slice(xnarr,[9,0],[1,nbeta]) # L+1 -> 9
        vn9=dynamic_slice(vnarr,[9,0],[1,nbeta]) # L+1 -> 9
        xn9,vn9,key_it=loop_UBU0(xn9,vn9,hper2c,key_it,extraburnin2*512) # 2^(L+1) -> 2^9 = 512
        xnarr=dynamic_update_slice(xnarr,xn9,[9,0]) # L+1 -> 9
        vnarr=dynamic_update_slice(vnarr,vn9,[9,0]) # L+1 -> 9

        xn89,vn89,_,key_it=jax.lax.fori_loop(0,extraburnin2*256,multi0_inner,(dynamic_slice(xnarr,[8,0],[2,nbeta]),dynamic_slice(vnarr,[8,0],[2,nbeta]),hper2c,key_it))
        xnarr=dynamic_update_slice(xnarr,xn89,[8,0])
        vnarr=dynamic_update_slice(vnarr,vn89,[8,0])

        xn789,vn789,_,key_it=jax.lax.fori_loop(0,extraburnin2*128,multi1_inner,(dynamic_slice(xnarr,[7,0],[3,nbeta]),dynamic_slice(vnarr,[7,0],[3,nbeta]),hper2c,key_it))
        xnarr=dynamic_update_slice(xnarr,xn789,[7,0])
        vnarr=dynamic_update_slice(vnarr,vn789,[7,0])

        xn6789,vn6789,_,key_it=jax.lax.fori_loop(0,extraburnin2*64,multi2_inner,(dynamic_slice(xnarr,[6,0],[4,nbeta]),dynamic_slice(vnarr,[6,0],[4,nbeta]),hper2c,key_it))
        xnarr=dynamic_update_slice(xnarr,xn6789,[6,0])
        vnarr=dynamic_update_slice(vnarr,vn6789,[6,0])

        xn56789,vn56789,_,key_it=jax.lax.fori_loop(0,extraburnin2*32,multi3_inner,(dynamic_slice(xnarr,[5,0],[5,nbeta]),dynamic_slice(vnarr,[5,0],[5,nbeta]),hper2c,key_it))
        xnarr=dynamic_update_slice(xnarr,xn56789,[5,0])
        vnarr=dynamic_update_slice(vnarr,vn56789,[5,0])

        xn456789,vn456789,_,key_it=jax.lax.fori_loop(0,extraburnin2*16,multi4_inner,(dynamic_slice(xnarr,[4,0],[6,nbeta]),dynamic_slice(vnarr,[4,0],[6,nbeta]),hper2c,key_it))
        xnarr=dynamic_update_slice(xnarr,xn456789,[4,0])
        vnarr=dynamic_update_slice(vnarr,vn456789,[4,0])

        xn3456789,vn3456789,_,key_it=jax.lax.fori_loop(0,extraburnin2*8,multi5_inner,(dynamic_slice(xnarr,[3,0],[7,nbeta]),dynamic_slice(vnarr,[3,0],[7,nbeta]),hper2c,key_it))
        xnarr=dynamic_update_slice(xnarr,xn3456789,[3,0])
        vnarr=dynamic_update_slice(vnarr,vn3456789,[3,0])

        xn23456789,vn23456789,_,key_it=jax.lax.fori_loop(0,extraburnin2*4,multi6_inner,(dynamic_slice(xnarr,[2,0],[8,nbeta]),dynamic_slice(vnarr,[2,0],[8,nbeta]),hper2c,key_it))
        xnarr=dynamic_update_slice(xnarr,xn23456789,[2,0])
        vnarr=dynamic_update_slice(vnarr,vn23456789,[2,0])

        xn123456789,vn123456789,_,key_it=jax.lax.fori_loop(0,extraburnin2*2,multi7_inner,(dynamic_slice(xnarr,[1,0],[9,nbeta]),dynamic_slice(vnarr,[1,0],[9,nbeta]),hper2c,key_it))
        xnarr=dynamic_update_slice(xnarr,xn123456789,[1,0])
        vnarr=dynamic_update_slice(vnarr,vn123456789,[1,0])


        xnarr,vnarr,hper2c,key_it=jax.lax.fori_loop(0,jointburnin,multi8_inner,(xnarr,vnarr,hper2c,key_it))

        meanx_single=jnp.zeros([maxmultilevels+2,test_dim])
        meanxsquare_single=jnp.zeros([maxmultilevels+2,test_dim])
        
        xnarr,vnarr,meanx_single,meanxsquare_single,hper2c,key_it=jax.lax.fori_loop(0,niter//thin,multi8_inner_thin,(xnarr,vnarr,meanx_single,meanxsquare_single,hper2c,key_it))


        meanx_single=meanx_single/(niter//thin)
        meanxsquare_single=meanxsquare_single/(niter//thin)
        meanx_diff=meanx_single[1:(maxmultilevels+2),:]-meanx_single[0:(maxmultilevels+1),:]
        meanx_diff=dynamic_update_slice(meanx_diff,jnp.zeros([1,test_dim]),[multilevels+1,0])
        meanxsquare_diff=meanxsquare_single[1:(maxmultilevels+2),:]-meanxsquare_single[0:(maxmultilevels+1),:]
        meanxsquare_diff=dynamic_update_slice(meanxsquare_diff,jnp.zeros([1,test_dim]),[multilevels+1,0])
        err=jnp.mean(jnp.abs(dynamic_slice(xnarr,[1,0],[maxmultilevels+1,nbeta])-dynamic_slice(xnarr,[0,0],[maxmultilevels+1,nbeta])),axis=1)
        return meanx_diff,meanxsquare_diff,key_it,err    

    xnarr=jnp.ones([maxmultilevels+2,1])@stop_gradient(x)
    vnarr=jnp.ones([maxmultilevels+2,1])@stop_gradient(v)
    key_it=key
    match maxmultilevels:
        case 0:
            meanx,meanxsquare,key_it,err=multi0((xnarr,vnarr,niter,extraburnin2, jointburnin,thin, key_it))
        case 1:
            meanx,meanxsquare,key_it,err=jax.lax.switch(multilevels,(multi0,multi1),(xnarr,vnarr,niter,extraburnin2, jointburnin,thin, key_it))
        case 2:
            meanx,meanxsquare,key_it,err=jax.lax.switch(multilevels,(multi0,multi1,multi2),(xnarr,vnarr,niter,extraburnin2, jointburnin,thin, key_it))
        case 3:
            meanx,meanxsquare,key_it,err=jax.lax.switch(multilevels,(multi0,multi1,multi2,multi3),(xnarr,vnarr,niter,extraburnin2, jointburnin,thin, key_it))
        case 4:
            meanx,meanxsquare,key_it,err=jax.lax.switch(multilevels,(multi0,multi1,multi2,multi3,multi4),(xnarr,vnarr,niter,extraburnin2, jointburnin,thin, key_it))
        case 5:
            meanx,meanxsquare,key_it,err=jax.lax.switch(multilevels,(multi0,multi1,multi2,multi3,multi4,multi5),(xnarr,vnarr,niter,extraburnin2, jointburnin,thin, key_it))
        case 6:
            meanx,meanxsquare,key_it,err=jax.lax.switch(multilevels,(multi0,multi1,multi2,multi3,multi4,multi5,multi6),(xnarr,vnarr,niter,extraburnin2, jointburnin,thin, key_it))
        case 7:
            meanx,meanxsquare,key_it,err=jax.lax.switch(multilevels,(multi0,multi1,multi2,multi3,multi4,multi5,multi6,multi7),(xnarr,vnarr,niter,extraburnin2, jointburnin,thin, key_it))
        case _:
            meanx,meanxsquare,key_it,err=jax.lax.switch(multilevels,(multi0,multi1,multi2,multi3,multi4,multi5,multi6,multi7,multi8),(xnarr,vnarr,niter,extraburnin2, jointburnin,thin, key_it))
        

    return meanx, meanxsquare,key_it,err


def generate_multilevels(rep,r,c,maxlevel,key):    
    _,sub_key0=random.split(key)
    repruns=np.zeros(maxlevel+1,dtype=int)
    problevels=np.zeros(maxlevel+1)    
    repruns[0]=rep
    problevels[0]=rep
    for lev in range(maxlevel):
        problevels[lev+1]=rep*(c/(r**lev))
        repruns[lev+1]=np.ceil(problevels[lev+1])
    detlevels=min((sum(problevels[1:(maxlevel+1)]>=1.0))+1,maxlevel)   
    repruns=jnp.asarray(repruns)
    randlevels=maxlevel-detlevels
    if(detlevels<maxlevel):
        repruns_up=jnp.asarray(random.uniform(sub_key0,[1,randlevels])<problevels[(detlevels+1):(detlevels+1+randlevels)],dtype=int).reshape([randlevels,])
        repruns=dynamic_update_slice(repruns,repruns_up,[detlevels+1,])
    multilevelsfind=jnp.nonzero(repruns[(detlevels+1):(maxlevel+1)],size=randlevels,fill_value=-1)
    multilevels=(multilevelsfind[0][0]>(-1))*(jnp.max(multilevelsfind[0])+1)
    return multilevels
    
vmap_generate_multilevels=jax.vmap(generate_multilevels,[None]*4+[0])

def multilevel_ubu(niter,burnin,rep,h,gam, vmap_grad_lpost, vmap_test_function, r,c,nbeta,test_dim,beta_min,maxlevel,max_parallel_chain, multilevels,key):

    key_it,sub_key0=random.split(key)


    singleMCMC_jit=jax.jit(singleMCMC,static_argnames=['vmap_grad_lpost','vmap_test_function','test_dim','nbeta'])
    doubleMCMC_jit=jax.jit(doubleMCMC,static_argnames=['vmap_grad_lpost','vmap_test_function','test_dim','nbeta'])


    repruns=np.zeros(maxlevel+1,dtype=int)
    problevels=np.zeros(maxlevel+1)    
    repruns[0]=rep
    problevels[0]=rep
    for lev in range(maxlevel):
        problevels[lev+1]=rep*(c/(r**lev))
        repruns[lev+1]=np.ceil(problevels[lev+1])
    maxruns=sum(repruns)
    detlevels=min((sum(problevels[1:(maxlevel+1)]>=1.0))+1,maxlevel)
    randlevels=maxlevel-detlevels

    blockstart=np.concatenate((np.asarray([0]),np.cumsum(repruns[0:(maxlevel)])))
    blockend=np.cumsum(repruns[0:(maxlevel+1)])
    
    ngradtot=(burnin+niter)*rep

    for lev in range(detlevels):
        extraburnin2=burnin*(2**(lev+1))
        jointburnin=burnin*(2**lev)*(lev+1)*3
        ngradtot+=repruns[lev+1]*(extraburnin2+jointburnin+niter*(2**lev)*3)
    for lev in range(detlevels,maxlevel):
        extraburnin2=burnin*(2**(lev+1))
        jointburnin=burnin*(2**lev)*(lev+1)*3
        ngradtot+=problevels[lev+1]*(extraburnin2+jointburnin+niter*(2**lev)*3)

    repruns[(detlevels+1):(maxlevel+1)]=0    

      
   
    rho=0.25
    means=jnp.zeros([maxruns,test_dim])
    squaremeans=jnp.zeros([maxruns,test_dim])
           
    
    no_batches=np.asarray(np.ceil(rep/max_parallel_chain), dtype=int)
    hper2c=jnp.asarray(hper2constarr(h,gam))
    for repit in range(no_batches):
        if(repit<(no_batches-1)):
            par_chain=max_parallel_chain
        else:
            par_chain=rep-max_parallel_chain*(no_batches-1)
        x=jnp.ones([par_chain,1])@beta_min.reshape([1,nbeta])
        key_it,subkey_it=random.split(key_it)
        v=random.normal(subkey_it,[par_chain,nbeta])
        key_it,subkey_it=random.split(key_it)      
        means_up,squaremeans_up,key_it=singleMCMC_jit(x,v,nbeta,niter,burnin,hper2c,vmap_grad_lpost,vmap_test_function,test_dim, subkey_it)      
        means=dynamic_update_slice(means,means_up,(repit*max_parallel_chain,0))
        squaremeans=dynamic_update_slice(squaremeans,squaremeans_up,(repit*max_parallel_chain,0))
    

    for lev in range(0,detlevels-1):
        no_batches=np.asarray(np.ceil(repruns[lev+1]/max_parallel_chain), dtype=int)
        hper4c=jnp.asarray(hper2constarr(h/(2**(lev+1)),gam))
        for repit in range(no_batches):
            if(repit<(no_batches-1)):
                par_chain=max_parallel_chain
            else:
                par_chain=repruns[lev+1]-max_parallel_chain*(no_batches-1)
            x=jnp.ones([par_chain,1])@beta_min.reshape([1,nbeta])
            key_it,subkey_it=random.split(key_it)
            v=random.normal(subkey_it,x.shape)
            x2=x
            v2=v
            extraburnin2=burnin*(2**(lev+1))
            jointburnin=burnin*(2**(lev))*(lev+1)
            thin=2**lev
            key_it,subkey_it=random.split(key_it)
            means_up,squaremeans_up,key_it,err=doubleMCMC_jit(x,v,x2,v2,nbeta,niter*(2**(lev)),extraburnin2, jointburnin,thin,hper4c,vmap_grad_lpost,vmap_test_function,test_dim, subkey_it)
            means=dynamic_update_slice(means,means_up,(blockstart[lev+1]+repit*max_parallel_chain,0))
            squaremeans=dynamic_update_slice(squaremeans,squaremeans_up,(blockstart[lev+1]+repit*max_parallel_chain,0))
            print("Lev:",lev,"/",lev+1,"xn/x2n error:", err)
            print("h:",h/(2**(lev)),"h/2:",h/(2**(lev+1)))

    repruns=jnp.asarray(repruns)
    if(detlevels<maxlevel):
        repruns_up=jnp.asarray(random.uniform(sub_key0,[1,randlevels])<problevels[(detlevels+1):(detlevels+1+randlevels)],dtype=int).reshape([randlevels,])
        repruns=dynamic_update_slice(repruns,repruns_up[0:multilevels],[detlevels+1,])
    
    x=beta_min.reshape([1,nbeta])
    key_it,subkey_it=random.split(key_it)
    v=random.normal(subkey_it,x.shape)
    key_it,subkey_it=random.split(key_it)

    extraburnin2=burnin*(2**(detlevels-1))
    jointburnin=burnin*(2**(detlevels-1))*(detlevels-1)

    maxmultilevels=randlevels
    print("maxmultilevels:",maxmultilevels)
    print("multilevels:",multilevels)

    hper2c_list=[None]*(maxmultilevels+2)
    print("Lev:",detlevels-1,"/",detlevels,"...")
    hm=h/(2**(detlevels-1))
    print("hm:",hm)
    for it in range(maxmultilevels+2):    
        hper2c_list[it]=hper2constarr(hm/(2**(it)),gam)
    hper2c_arr=jnp.stack(hper2c_list)

    multiMCMC_jit=jax.jit(multiMCMC,static_argnames=['nbeta','vmap_grad_lpost','vmap_test_function','test_dim','maxmultilevels'])
    means_up,squaremeans_up,key_it,err=multiMCMC_jit(x,v,nbeta,niter*(2**(detlevels-1)),extraburnin2, jointburnin,thin,multilevels,maxmultilevels,vmap_grad_lpost,vmap_test_function,test_dim, subkey_it,hper2c_arr)
    print("Multi error:",err)
    means=dynamic_update_slice(means,means_up,(blockstart[detlevels],0))
    squaremeans=dynamic_update_slice(squaremeans,squaremeans_up,(blockstart[detlevels],0))
    
       
    return (rep,niter,detlevels,randlevels,repruns,maxlevel,problevels,ngradtot,nbeta,means,squaremeans,blockstart,blockend,test_dim,maxruns,rho)


vmultilevel_ubu=jax.vmap(multilevel_ubu,[None]*15+[0])

def vmap_multilevel_ubu(niter,burnin,rep,h,gam, vmap_grad_lpost, vmap_test_function, r,c,nbeta,test_dim,beta_min,maxlevel,max_parallel_chain, keys, chunk_size):
    par_runs=keys.shape[0]
    res=[None]*(par_runs//chunk_size+(par_runs%chunk_size>0))
    multilevels_arr=vmap_generate_multilevels(rep,r,c,maxlevel,keys)
    multilevels_arr=(multilevels_arr<=8)*multilevels_arr+(multilevels_arr>8)*8
    sort_indices = jnp.argsort(multilevels_arr)
    keys=keys[sort_indices]
    multilevels_arr=multilevels_arr[sort_indices]
    print(multilevels_arr)
    cpus = jax.devices("cpu")

    for wit in range(par_runs//chunk_size):
        batch_key=keys[wit*chunk_size:(wit+1)*chunk_size]
        res[wit]=jax.device_put(vmultilevel_ubu(niter,burnin,rep,h,gam, vmap_grad_lpost, vmap_test_function, r,c,nbeta,test_dim,beta_min,maxlevel,max_parallel_chain, max(multilevels_arr[wit*chunk_size:(wit+1)*chunk_size]),batch_key),cpus[0])
    if(par_runs%chunk_size>0):
        batch_key=keys[(par_runs//chunk_size)*chunk_size:]
        res[par_runs//chunk_size]=jax.device_put(vmultilevel_ubu(niter,burnin,rep,h,gam, vmap_grad_lpost, vmap_test_function, r,c,nbeta,test_dim,beta_min,maxlevel,max_parallel_chain, max(multilevels_arr[(par_runs//chunk_size)*chunk_size:]),batch_key),cpus[0])

    res1=[]
    num_res=len(res)
    for it in range(len(res[0])):
        res0it=[]
        for it2 in range(num_res):
            res0it.append(res[it2][it])
        stacked=jnp.concatenate(res0it,axis=0)
        res1.append(stacked)
    return res1