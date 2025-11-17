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

def multiO(v,hc,xiarr):
    def body_fn(inp,xi):
        vn=inp
        vn=vn*hc[1]+hc[4]*xi[0,:]+hc[5]*xi[1,:]
        return vn,0
    vn,_=jax.lax.scan(body_fn,v,xiarr)
    return vn


def vmap_grad_lpost_approx(x,beta_star,grad_beta_star,Hprodv):
    return grad_beta_star+Hprodv(x-beta_star)

def OHO_UBU_approx_step2(x,v,x2,v2,nbeta,beta_star2,grad_beta_star2,rep,hper4c,beta_min,Hprodv,exp_hM,xi):   
    if(x.ndim==1):
        rep=1
    else:
        rep=x.shape[0]

    xn=x
    vn=v

    vn=vn*hper4c[1]+hper4c[4]*xi[0:rep,:]+hper4c[5]*xi[(rep):(2*rep),:]
    vn=vn*hper4c[1]+hper4c[4]*xi[(2*rep):(3*rep),:]+hper4c[5]*xi[(3*rep):(4*rep),:]
    beta_min_mx=jnp.ones([rep,1])@beta_min.reshape([1,nbeta])
    
    xnvn=exp_hM(jnp.concatenate([xn-beta_min_mx,vn],axis=1),hper4c[0]*2)
    xn=xnvn[0:rep,0:nbeta]+beta_min_mx
    vn=xnvn[0:rep,(nbeta):(2*nbeta)]

    vn=vn*hper4c[1]+hper4c[4]*xi[(4*rep):(5*rep),:]+hper4c[5]*xi[(5*rep):(6*rep),:]
    vn=vn*hper4c[1]+hper4c[4]*xi[(6*rep):(7*rep),:]+hper4c[5]*xi[(7*rep):(8*rep),:]

    [x2n,v2n]=U(x2,v2,hper4c,xi[0:rep,:],xi[(rep):(2*rep),:])
    gr=vmap_grad_lpost_approx(x2n,beta_star2,grad_beta_star2,Hprodv)
    v2n=v2n-hper4c[0]*gr
    [x2n,v2n]=U(x2n,v2n,hper4c,xi[(2*rep):(3*rep),:],xi[(3*rep):(4*rep),:])
    [x2n,v2n]=U(x2n,v2n,hper4c,xi[(4*rep):(5*rep),:],xi[(5*rep):(6*rep),:])
    gr=vmap_grad_lpost_approx(x2n,beta_star2,grad_beta_star2,Hprodv)        
    v2n=v2n-hper4c[0]*gr
    [x2n,v2n]=U(x2n,v2n,hper4c,xi[(6*rep):(7*rep),:],xi[(7*rep):(8*rep),:])


    return xn,vn,x2n,v2n

def UBU_approx_step2(x,v, x2,v2,beta_star,grad_beta_star,beta_star2,grad_beta_star2, rep,hper4c,Hprodv,xi):   
    if(x.ndim==1):
        rep=1
    else:
        rep=x.shape[0]
  
    [x2n,v2n]=U(x2,v2,hper4c,xi[0:rep,:],xi[(rep):(2*rep),:])
    gr=vmap_grad_lpost_approx(x2n,beta_star2,grad_beta_star2,Hprodv)
    v2n=v2n-hper4c[0]*gr
    [x2n,v2n]=U(x2n,v2n,hper4c,xi[(2*rep):(3*rep),:],xi[(3*rep):(4*rep),:])
    [x2n,v2n]=U(x2n,v2n,hper4c,xi[(4*rep):(5*rep),:],xi[(5*rep):(6*rep),:])
    gr=vmap_grad_lpost_approx(x2n,beta_star2,grad_beta_star2,Hprodv)        
    v2n=v2n-hper4c[0]*gr
    [x2n,v2n]=U(x2n,v2n,hper4c,xi[(6*rep):(7*rep),:],xi[(7*rep):(8*rep),:])

    [xn,vn]=U(x,v,hper4c,xi[0:rep,:],xi[(rep):(2*rep),:])
    [xn,vn]=U(xn,vn,hper4c,xi[(2*rep):(3*rep),:],xi[(3*rep):(4*rep),:])
    gr=vmap_grad_lpost_approx(xn,beta_star,grad_beta_star,Hprodv)
    vn=vn-2*hper4c[0]*gr
    [xn,vn]=U(xn,vn,hper4c,xi[(4*rep):(5*rep),:],xi[(5*rep):(6*rep),:])
    [xn,vn]=U(xn,vn,hper4c,xi[(6*rep):(7*rep),:],xi[(7*rep):(8*rep),:])
    return xn,vn,x2n,v2n




def burnMCMC2(nbeta,rep,burn_oho_ubu,burn_ubu,x0,hper4c,beta_min,vmap_grad_lpost,Hprodv,exp_hM,repfullgrad,key):
    def beta_star_no_update(inp):
        beta_star,grad_beta_star,_=inp
        return beta_star,grad_beta_star
    def beta_star_update(inp):
        _, _,xn=inp
        return xn,vmap_grad_lpost(xn)
    
    def one_step_oho_ubu(it,inp):
        (xn,vn,x2n,v2n,beta_star2,grad_beta_star2,key_it)=inp
        key_it, subkey_it = random.split(key_it)
        xi=random.normal(subkey_it,(8*rep,nbeta))
        beta_star2,grad_beta_star2=jax.lax.cond(((2*it) % repfullgrad==0), beta_star_update,beta_star_no_update, (beta_star2, grad_beta_star2,x2n))        
        xn,vn,x2n,v2n=OHO_UBU_approx_step2(xn,vn,x2n,v2n,nbeta,beta_star2,grad_beta_star2,rep,hper4c,beta_min,Hprodv,exp_hM,xi)

        return xn,vn,x2n,v2n,beta_star2,grad_beta_star2,key_it

    def one_step_ubu_approx2(it,inp):
        (xn,vn,x2n,v2n,beta_star,grad_beta_star,beta_star2,grad_beta_star2,key_it)=inp
        key_it, subkey_it = jax.random.split(key_it)
        xi=random.normal(subkey_it,(8*rep,nbeta))
    
        beta_star,grad_beta_star=jax.lax.cond(((it-burn_oho_ubu) % repfullgrad==0), beta_star_update,beta_star_no_update, (beta_star, grad_beta_star,xn))        
        beta_star2,grad_beta_star2=jax.lax.cond(((2*it) % repfullgrad==0), beta_star_update,beta_star_no_update, (beta_star2, grad_beta_star2,x2n))

        xn,vn,x2n,v2n=UBU_approx_step2(xn,vn,x2n,v2n,beta_star,grad_beta_star,beta_star2,grad_beta_star2, rep,hper4c,Hprodv,xi)        
                
        return xn,vn,x2n,v2n,beta_star,grad_beta_star,beta_star2,grad_beta_star2,key_it
    

    key_it=key
    key_it, subkey_it,subkey_it2 = jax.random.split(key_it,3)
   
    xn=x0
    vn=random.normal(subkey_it2,(rep,nbeta))
    x2n=xn
    v2n=vn


    beta_star=jnp.zeros([rep,nbeta])
    grad_beta_star=jnp.zeros([rep,nbeta])   
    beta_star2=jnp.zeros([rep,nbeta])
    grad_beta_star2=jnp.zeros([rep,nbeta])

    (xn,vn,x2n,v2n,beta_star2,grad_beta_star2,key_it)=jax.lax.fori_loop(0, burn_oho_ubu, one_step_oho_ubu, (xn,vn,x2n,v2n,beta_star2,grad_beta_star2,key_it))
    (xn,vn,x2n,v2n,beta_star,grad_beta_star,beta_star2,grad_beta_star2,key_it)=jax.lax.fori_loop(burn_oho_ubu,burn_oho_ubu+burn_ubu,one_step_ubu_approx2, (xn,vn,x2n,v2n,beta_star,grad_beta_star,beta_star2,grad_beta_star2,key_it))
    
    return xn,vn,x2n,v2n,beta_star,grad_beta_star,beta_star2,grad_beta_star2,key_it


def Gaussian_samples(nbeta,rep,niter,test_dim,vmap_test_function,beta_min,invcholHprodv,max_Gaussian_samp,key_it):
    msamp=max((max_Gaussian_samp//rep)*rep,rep)
    meanx=jnp.zeros([rep,test_dim])
    meanxsquare=jnp.zeros([rep,test_dim])
    runs=(rep*niter)//msamp
    subkey=random.split(key_it,runs+2)
    key_it=subkey[runs+1]
    for it in range(runs):
        xn=jnp.ones([msamp,1])@beta_min.reshape([1,nbeta])+invcholHprodv(random.normal(subkey[it],(msamp,nbeta)))
        test_vals=vmap_test_function(xn).reshape(rep,msamp//rep,test_dim)
        meanx+=test_vals.sum([1])
        meanxsquare+=(test_vals**2).sum([1])
    if((rep*niter)>(runs*msamp)):
        xn=jnp.ones([rep*niter-runs*msamp,1])@beta_min.reshape([1,nbeta])+invcholHprodv(random.normal(subkey[runs],(rep*niter-runs*msamp,nbeta)))
        test_vals=vmap_test_function(xn).reshape(rep,(rep*niter-runs*msamp)//rep,test_dim)
        meanx+=test_vals.sum([1])
        meanxsquare+=(test_vals**2).sum([1])
    meanx=meanx/niter
    meanxsquare=meanxsquare/niter   
    return meanx,meanxsquare,key_it





def doubleMCMC_levels01(nbeta,rep,niter,extraburnin2, jointburnin,thin, x0, hper4c, test_dim, beta_min,vmap_grad_lpost,vmap_test_function,Hprodv,exp_hM,repfullgrad,key):
    def beta_star_no_update(inp):
        beta_star,grad_beta_star,_=inp
        return beta_star,grad_beta_star
    def beta_star_update(inp):
        _, _,xn=inp
        return xn,vmap_grad_lpost(xn)
          
    def one_step_MCMC01(thinit,inp):
        (xn,vn,x2n,v2n,beta_star2,grad_beta_star2,it,key_it)=inp
        key_it, subkey_it = jax.random.split(key_it)
        xi=random.normal(subkey_it,(8*rep,nbeta))
        beta_star2,grad_beta_star2=jax.lax.cond((2*it*thin+2*thinit) % repfullgrad==0, beta_star_update,beta_star_no_update, (beta_star2, grad_beta_star2,x2n))        
        xn,vn,x2n,v2n=OHO_UBU_approx_step2(xn,vn,x2n,v2n,nbeta,beta_star2,grad_beta_star2,rep,hper4c,beta_min,Hprodv,exp_hM,xi)
        return xn,vn,x2n,v2n,beta_star2,grad_beta_star2,it,key_it

    def thin_step_MCMC01(it,inp):
        (xn,vn,x2n,v2n,beta_star2,grad_beta_star2,meanx,meanxsquare,key_it)=inp
        (xn,vn,x2n,v2n,beta_star2,grad_beta_star2,it,key_it)=jax.lax.fori_loop(0, thin, one_step_MCMC01, (xn,vn,x2n,v2n,beta_star2,grad_beta_star2,it,key_it))
        test_vals1=vmap_test_function(xn)
        test_vals2=vmap_test_function(x2n)       
        meanx=meanx+test_vals2-test_vals1
        meanxsquare=meanxsquare+jnp.square(test_vals2)-jnp.square(test_vals1)
        return xn,vn,x2n,v2n,beta_star2,grad_beta_star2,meanx,meanxsquare,key_it

    xn,vn,x2n,v2n,_,_,beta_star2,grad_beta_star2,key_it=burnMCMC2(nbeta,rep,extraburnin2+jointburnin,0,x0,hper4c,beta_min,vmap_grad_lpost,Hprodv,exp_hM,repfullgrad,key)
   
    meanx=jnp.zeros([rep,test_dim])
    meanxsquare=jnp.zeros([rep,test_dim])
    nsamp=niter//thin

    (xn,vn,x2n,v2n,beta_star2,grad_beta_star2,meanx,meanxsquare,key_it)=jax.lax.fori_loop(0, nsamp, thin_step_MCMC01, (xn,vn,x2n,v2n,beta_star2,grad_beta_star2,meanx,meanxsquare,key_it))
    errxnx2n=jnp.mean(jnp.abs(xn-x2n))

    meanx=meanx/nsamp
    meanxsquare=meanxsquare/nsamp
    return meanx,meanxsquare,key_it,errxnx2n

def doubleMCMC(nbeta,rep,niter,extraburnin2, jointburnin,thin, x0, hper4c,test_dim, beta_min,vmap_grad_lpost,vmap_test_function, Hprodv,exp_hM,repfullgrad,key):
    def beta_star_no_update(inp):
        beta_star,grad_beta_star,_=inp
        return beta_star,grad_beta_star
    def beta_star_update(inp):
        _, _,xn=inp
        return xn,vmap_grad_lpost(xn)
        
    xn,vn,x2n,v2n,beta_star,grad_beta_star,beta_star2,grad_beta_star2,key_it=burnMCMC2(nbeta,rep,extraburnin2,jointburnin,x0,hper4c,beta_min,vmap_grad_lpost,Hprodv,exp_hM,repfullgrad,key)

    meanx=jnp.zeros([rep,test_dim])
    meanxsquare=jnp.zeros([rep,test_dim])
    nsamp=niter//thin

    
    def one_step_MCMC(thinit,inp):
        (xn,vn,x2n,v2n,beta_star,grad_beta_star,beta_star2,grad_beta_star2,it,key_it)=inp
        key_it, subkey_it = jax.random.split(key_it)
        xi=random.normal(subkey_it,(8*rep,nbeta))
        beta_star,grad_beta_star=jax.lax.cond((jointburnin+it*thin+thinit) % repfullgrad==0, beta_star_update,beta_star_no_update, (beta_star, grad_beta_star,xn))
        beta_star2,grad_beta_star2=jax.lax.cond(2*(extraburnin2+jointburnin+it*thin+thinit) % repfullgrad==0, beta_star_update,beta_star_no_update, (beta_star2, grad_beta_star2,x2n))
        xn,vn,x2n,v2n=UBU_approx_step2(xn,vn,x2n,v2n,beta_star,grad_beta_star,beta_star2,grad_beta_star2, rep,hper4c,Hprodv,xi)
        return xn,vn,x2n,v2n,beta_star,grad_beta_star,beta_star2,grad_beta_star2,it,key_it

    def thin_step_MCMC(it,inp):
        (xn,vn,x2n,v2n,beta_star,grad_beta_star,beta_star2,grad_beta_star2,meanx,meanxsquare,key_it)=inp
        (xn,vn,x2n,v2n,beta_star,grad_beta_star,beta_star2,grad_beta_star2,it,key_it)=jax.lax.fori_loop(0, thin, one_step_MCMC, (xn,vn,x2n,v2n,beta_star,grad_beta_star,beta_star2,grad_beta_star2,it,key_it))
        test_vals1=vmap_test_function(xn)
        test_vals2=vmap_test_function(x2n)       
        meanx=meanx+test_vals2-test_vals1
        meanxsquare=meanxsquare+jnp.square(test_vals2)-jnp.square(test_vals1)
        return xn,vn,x2n,v2n,beta_star,grad_beta_star,beta_star2,grad_beta_star2,meanx,meanxsquare,key_it

    (xn,vn,x2n,v2n,beta_star,grad_beta_star,beta_star2,grad_beta_star2,meanx,meanxsquare,key_it)=jax.lax.fori_loop(0, nsamp, thin_step_MCMC, (xn,vn,x2n,v2n,beta_star,grad_beta_star,beta_star2,grad_beta_star2,meanx,meanxsquare,key_it))
    errxnx2n=jnp.mean(jnp.abs(xn-x2n))
    meanx=meanx/nsamp
    meanxsquare=meanxsquare/nsamp
    return meanx,meanxsquare,key_it,errxnx2n














def multiMCMC(nbeta,niter,extraburnin2, jointburnin, thin, multilevels,maxmultilevels, x0,vmap_grad_lpost,Hprodv, exp_hM, beta_min,vmap_test_function,test_dim, repfullgrad,key, hper2c_arr):
    
    def beta_star_no_update(inp):
        beta_star,grad_beta_star,_=inp
        return beta_star,grad_beta_star
    def beta_star_update(inp):
        _, _,xn=inp
        return xn,vmap_grad_lpost(xn)

    def singlestep_UBU(xn,vn,beta_star,grad_beta_star,xi,hper2c):
        xiwidth=xi.shape[0]//2
        xn,vn=multiU(xn,vn,hper2c,xi[0:xiwidth,:])
        gr=vmap_grad_lpost_approx(xn,beta_star,grad_beta_star,Hprodv)
        vn=vn-xiwidth*(hper2c[0])*gr
        xn,vn=multiU(xn,vn,hper2c,xi[xiwidth:(2*xiwidth),:])
        return xn,vn
    
    def singlestep_OHO(xn,vn,xi,hper2c):
        xiwidth=xi.shape[0]//2
        vn=multiO(vn,hper2c,xi[0:xiwidth,:])
        xnvn=exp_hM(jnp.concatenate([xn-beta_min.reshape([1,nbeta]),vn],axis=1),hper2c[0]*2)
        xn=xnvn[0:1,0:nbeta]+beta_min.reshape([1,nbeta])
        vn=xnvn[0:1,(nbeta):(2*nbeta)]
        vn=multiO(vn,hper2c,xi[xiwidth:(2*xiwidth),:])
        return xn,vn
        
    def multistep_UBU0(xn,vn,beta_star,grad_beta_star,iter,xi,hper2c):
        beta_star,grad_beta_star=jax.lax.cond((iter%repfullgrad==0), beta_star_update,beta_star_no_update, (beta_star, grad_beta_star,xn))
        iter+=1
        xn,vn=singlestep_UBU(xn,vn,beta_star,grad_beta_star,xi,hper2c)
        return xn,vn,beta_star,grad_beta_star,iter
    
       
    def multistep_UBU1(xn,vn,beta_star,grad_beta_star,iter,xi,hper2c):
        xiwidth=xi.shape[0]//2
        def inner(it, inp):
            xn,vn,beta_star,grad_beta_star,iter=inp
            beta_star,grad_beta_star=jax.lax.cond((iter%repfullgrad==0), beta_star_update,beta_star_no_update, (beta_star, grad_beta_star,xn))
            iter+=1            
            xn,vn=singlestep_UBU(xn,vn,beta_star,grad_beta_star,dynamic_slice(xi,[(it*xiwidth),0,0],[xiwidth,2,nbeta]),hper2c)
            return xn,vn,beta_star,grad_beta_star,iter
        xn,vn,beta_star,grad_beta_star,iter=jax.lax.fori_loop(0,2,inner,(xn,vn,beta_star,grad_beta_star,iter))
        return xn,vn,beta_star,grad_beta_star,iter
    
    def multistep_UBU2(xn,vn,beta_star,grad_beta_star,iter,xi,hper2c):
        xiwidth=xi.shape[0]//4
        def inner(it, inp):
            xn,vn,beta_star,grad_beta_star,iter=inp
            beta_star,grad_beta_star=jax.lax.cond((iter%repfullgrad==0), beta_star_update,beta_star_no_update, (beta_star, grad_beta_star,xn))
            iter+=1            
            xn,vn=singlestep_UBU(xn,vn,beta_star,grad_beta_star,dynamic_slice(xi,[(it*xiwidth),0,0],[xiwidth,2,nbeta]),hper2c)
            return xn,vn,beta_star,grad_beta_star,iter
        xn,vn,beta_star,grad_beta_star,iter=jax.lax.fori_loop(0,4,inner,(xn,vn,beta_star,grad_beta_star,iter))
        return xn,vn,beta_star,grad_beta_star,iter

    def multistep_UBU3(xn,vn,beta_star,grad_beta_star,iter,xi,hper2c):
        xiwidth=xi.shape[0]//8
        def inner(it, inp):
            xn,vn,beta_star,grad_beta_star,iter=inp
            beta_star,grad_beta_star=jax.lax.cond((iter%repfullgrad==0), beta_star_update,beta_star_no_update, (beta_star, grad_beta_star,xn))
            iter+=1            
            xn,vn=singlestep_UBU(xn,vn,beta_star,grad_beta_star,dynamic_slice(xi,[(it*xiwidth),0,0],[xiwidth,2,nbeta]),hper2c)
            return xn,vn,beta_star,grad_beta_star,iter
        xn,vn,beta_star,grad_beta_star,iter=jax.lax.fori_loop(0,8,inner,(xn,vn,beta_star,grad_beta_star,iter))
        return xn,vn,beta_star,grad_beta_star,iter
    
    def multistep_UBU4(xn,vn,beta_star,grad_beta_star,iter,xi,hper2c):
        xiwidth=xi.shape[0]//16
        def inner(it, inp):
            xn,vn,beta_star,grad_beta_star,iter=inp
            beta_star,grad_beta_star=jax.lax.cond((iter%repfullgrad==0), beta_star_update,beta_star_no_update, (beta_star, grad_beta_star,xn))
            iter+=1            
            xn,vn=singlestep_UBU(xn,vn,beta_star,grad_beta_star,dynamic_slice(xi,[(it*xiwidth),0,0],[xiwidth,2,nbeta]),hper2c)
            return xn,vn,beta_star,grad_beta_star,iter
        xn,vn,beta_star,grad_beta_star,iter=jax.lax.fori_loop(0,16,inner,(xn,vn,beta_star,grad_beta_star,iter))
        return xn,vn,beta_star,grad_beta_star,iter    
    
    def multistep_UBU5(xn,vn,beta_star,grad_beta_star,iter,xi,hper2c):
        xiwidth=xi.shape[0]//32
        def inner(it, inp):
            xn,vn,beta_star,grad_beta_star,iter=inp
            beta_star,grad_beta_star=jax.lax.cond((iter%repfullgrad==0), beta_star_update,beta_star_no_update, (beta_star, grad_beta_star,xn))
            iter+=1            
            xn,vn=singlestep_UBU(xn,vn,beta_star,grad_beta_star,dynamic_slice(xi,[(it*xiwidth),0,0],[xiwidth,2,nbeta]),hper2c)
            return xn,vn,beta_star,grad_beta_star,iter
        xn,vn,beta_star,grad_beta_star,iter=jax.lax.fori_loop(0,32,inner,(xn,vn,beta_star,grad_beta_star,iter))
        return xn,vn,beta_star,grad_beta_star,iter       

    def multistep_UBU6(xn,vn,beta_star,grad_beta_star,iter,xi,hper2c):
        xiwidth=xi.shape[0]//64
        def inner(it, inp):
            xn,vn,beta_star,grad_beta_star,iter=inp
            beta_star,grad_beta_star=jax.lax.cond((iter%repfullgrad==0), beta_star_update,beta_star_no_update, (beta_star, grad_beta_star,xn))
            iter+=1            
            xn,vn=singlestep_UBU(xn,vn,beta_star,grad_beta_star,dynamic_slice(xi,[(it*xiwidth),0,0],[xiwidth,2,nbeta]),hper2c)
            return xn,vn,beta_star,grad_beta_star,iter
        xn,vn,beta_star,grad_beta_star,iter=jax.lax.fori_loop(0,64,inner,(xn,vn,beta_star,grad_beta_star,iter))
        return xn,vn,beta_star,grad_beta_star,iter

    def multistep_UBU7(xn,vn,beta_star,grad_beta_star,iter,xi,hper2c):
        xiwidth=xi.shape[0]//128
        def inner(it, inp):
            xn,vn,beta_star,grad_beta_star,iter=inp
            beta_star,grad_beta_star=jax.lax.cond((iter%repfullgrad==0), beta_star_update,beta_star_no_update, (beta_star, grad_beta_star,xn))
            iter+=1            
            xn,vn=singlestep_UBU(xn,vn,beta_star,grad_beta_star,dynamic_slice(xi,[(it*xiwidth),0,0],[xiwidth,2,nbeta]),hper2c)
            return xn,vn,beta_star,grad_beta_star,iter
        xn,vn,beta_star,grad_beta_star,iter=jax.lax.fori_loop(0,128,inner,(xn,vn,beta_star,grad_beta_star,iter))
        return xn,vn,beta_star,grad_beta_star,iter

    def multistep_UBU8(xn,vn,beta_star,grad_beta_star,iter,xi,hper2c):
        xiwidth=xi.shape[0]//256
        def inner(it, inp):
            xn,vn,beta_star,grad_beta_star,iter=inp
            beta_star,grad_beta_star=jax.lax.cond((iter%repfullgrad==0), beta_star_update,beta_star_no_update, (beta_star, grad_beta_star,xn))
            iter+=1            
            xn,vn=singlestep_UBU(xn,vn,beta_star,grad_beta_star,dynamic_slice(xi,[(it*xiwidth),0,0],[xiwidth,2,nbeta]),hper2c)
            return xn,vn,beta_star,grad_beta_star,iter
        xn,vn,beta_star,grad_beta_star,iter=jax.lax.fori_loop(0,256,inner,(xn,vn,beta_star,grad_beta_star,iter))
        return xn,vn,beta_star,grad_beta_star,iter
    
    def multistep_UBU9(xn,vn,beta_star,grad_beta_star,iter,xi,hper2c):
        xiwidth=xi.shape[0]//512
        def inner(it, inp):
            xn,vn,beta_star,grad_beta_star,iter=inp
            beta_star,grad_beta_star=jax.lax.cond((iter%repfullgrad==0), beta_star_update,beta_star_no_update, (beta_star, grad_beta_star,xn))
            iter+=1            
            xn,vn=singlestep_UBU(xn,vn,beta_star,grad_beta_star,dynamic_slice(xi,[(it*xiwidth),0,0],[xiwidth,2,nbeta]),hper2c)
            return xn,vn,beta_star,grad_beta_star,iter
        xn,vn,beta_star,grad_beta_star,iter=jax.lax.fori_loop(0,512,inner,(xn,vn,beta_star,grad_beta_star,iter))
        return xn,vn,beta_star,grad_beta_star,iter

    def loop_OHO_UBU(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it,hper2c,numiter):
        def inner(it,inp):
            xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it=inp
            key_it,subkey_it=random.split(key_it)
            xi=random.normal(subkey_it,(4,2,nbeta))
            xn0,vn0=singlestep_OHO(xnarr[0:1,0:nbeta],vnarr[0:1,0:nbeta],xi,hper2c)
            xn1,vn1,beta_star1,grad_beta_star1,iter1=multistep_UBU1(xnarr[1:2,0:nbeta],vnarr[1:2,0:nbeta],beta_star_arr[1:2,0:nbeta],grad_beta_star_arr[1:2,0:nbeta],iter_arr[1],xi,hper2c)
            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star1,[1,0])            
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star1,[1,0])
            iter_arr=dynamic_update_slice(iter_arr,iter1.reshape([1,]),[1,])
            xnarr=dynamic_update_slice(xnarr,xn0,[0,0])
            vnarr=dynamic_update_slice(vnarr,vn0,[0,0])
            xnarr=dynamic_update_slice(xnarr,xn1,[1,0])
            vnarr=dynamic_update_slice(vnarr,vn1,[1,0])
            return xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it=jax.lax.fori_loop(0,numiter,inner,(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it))
        return xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it       
    
    def loop_OHO_UBU2(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it,hper2c,numiter):
        def inner(it,inp):
            xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it=inp
            key_it,subkey_it=random.split(key_it)
            xi=random.normal(subkey_it,(8,2,nbeta))
            xn0,vn0=singlestep_OHO(xnarr[0:1,0:nbeta],vnarr[0:1,0:nbeta],xi,hper2c)
            xn1,vn1,beta_star1,grad_beta_star1,iter1=multistep_UBU1(xnarr[1:2,0:nbeta],vnarr[1:2,0:nbeta],beta_star_arr[1:2,0:nbeta],grad_beta_star_arr[1:2,0:nbeta],iter_arr[1],xi,hper2c)
            xn2,vn2,beta_star2,grad_beta_star2,iter2=multistep_UBU2(xnarr[2:3,0:nbeta],vnarr[2:3,0:nbeta],beta_star_arr[2:3,0:nbeta],grad_beta_star_arr[2:3,0:nbeta],iter_arr[2],xi,hper2c)            
            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star1,[1,0])            
            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star2,[2,0])            
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star1,[1,0])
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star2,[2,0])
            iter_arr=dynamic_update_slice(iter_arr,iter1.reshape([1,]),[1,])
            iter_arr=dynamic_update_slice(iter_arr,iter2.reshape([1,]),[2,])
            xnarr=dynamic_update_slice(xnarr,xn0,[0,0])
            vnarr=dynamic_update_slice(vnarr,vn0,[0,0])
            xnarr=dynamic_update_slice(xnarr,xn1,[1,0])
            vnarr=dynamic_update_slice(vnarr,vn1,[1,0])
            xnarr=dynamic_update_slice(xnarr,xn2,[2,0])
            vnarr=dynamic_update_slice(vnarr,vn2,[2,0])
            return xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it=jax.lax.fori_loop(0,numiter,inner,(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it))
        return xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it
    
    def loop_OHO_UBU3(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it,hper2c,numiter):
        def inner(it,inp):
            xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it=inp
            key_it,subkey_it=random.split(key_it)
            xi=random.normal(subkey_it,(16,2,nbeta)) 
            
            xn0,vn0=singlestep_OHO(xnarr[0:1,0:nbeta],vnarr[0:1,0:nbeta],xi,hper2c)
            
            xn1,vn1,beta_star1,grad_beta_star1,iter1=multistep_UBU1(xnarr[1:2,0:nbeta],vnarr[1:2,0:nbeta],beta_star_arr[1:2,0:nbeta],grad_beta_star_arr[1:2,0:nbeta],iter_arr[1],xi,hper2c)
            xn2,vn2,beta_star2,grad_beta_star2,iter2=multistep_UBU2(xnarr[2:3,0:nbeta],vnarr[2:3,0:nbeta],beta_star_arr[2:3,0:nbeta],grad_beta_star_arr[2:3,0:nbeta],iter_arr[2],xi,hper2c)            
            xn3,vn3,beta_star3,grad_beta_star3,iter3=multistep_UBU3(xnarr[3:4,0:nbeta],vnarr[3:4,0:nbeta],beta_star_arr[3:4,0:nbeta],grad_beta_star_arr[3:4,0:nbeta],iter_arr[3],xi,hper2c)            

            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star1,[1,0])            
            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star2,[2,0])            
            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star3,[3,0])            
            
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star1,[1,0])
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star2,[2,0])
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star3,[3,0])
            
            iter_arr=dynamic_update_slice(iter_arr,iter1.reshape([1,]),[1,])
            iter_arr=dynamic_update_slice(iter_arr,iter2.reshape([1,]),[2,])
            iter_arr=dynamic_update_slice(iter_arr,iter3.reshape([1,]),[3,])
            
            xnarr=dynamic_update_slice(xnarr,xn0,[0,0])
            vnarr=dynamic_update_slice(vnarr,vn0,[0,0])
            
            xnarr=dynamic_update_slice(xnarr,xn1,[1,0])
            vnarr=dynamic_update_slice(vnarr,vn1,[1,0])
            xnarr=dynamic_update_slice(xnarr,xn2,[2,0])
            vnarr=dynamic_update_slice(vnarr,vn2,[2,0])
            xnarr=dynamic_update_slice(xnarr,xn3,[3,0])
            vnarr=dynamic_update_slice(vnarr,vn3,[3,0])
            
            return xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it
    
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it=jax.lax.fori_loop(0,numiter,inner,(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it))
        return xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it

    def loop_OHO_UBU4(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it,hper2c,numiter):
        def inner(it,inp):
            xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it=inp
            key_it,subkey_it=random.split(key_it)
            xi=random.normal(subkey_it,(32,2,nbeta)) 
            
            xn0,vn0=singlestep_OHO(xnarr[0:1,0:nbeta],vnarr[0:1,0:nbeta],xi,hper2c)
            
            xn1,vn1,beta_star1,grad_beta_star1,iter1=multistep_UBU1(xnarr[1:2,0:nbeta],vnarr[1:2,0:nbeta],beta_star_arr[1:2,0:nbeta],grad_beta_star_arr[1:2,0:nbeta],iter_arr[1],xi,hper2c)
            xn2,vn2,beta_star2,grad_beta_star2,iter2=multistep_UBU2(xnarr[2:3,0:nbeta],vnarr[2:3,0:nbeta],beta_star_arr[2:3,0:nbeta],grad_beta_star_arr[2:3,0:nbeta],iter_arr[2],xi,hper2c)            
            xn3,vn3,beta_star3,grad_beta_star3,iter3=multistep_UBU3(xnarr[3:4,0:nbeta],vnarr[3:4,0:nbeta],beta_star_arr[3:4,0:nbeta],grad_beta_star_arr[3:4,0:nbeta],iter_arr[3],xi,hper2c)            
            xn4,vn4,beta_star4,grad_beta_star4,iter4=multistep_UBU4(xnarr[4:5,0:nbeta],vnarr[4:5,0:nbeta],beta_star_arr[4:5,0:nbeta],grad_beta_star_arr[4:5,0:nbeta],iter_arr[4],xi,hper2c)            

            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star1,[1,0])            
            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star2,[2,0])            
            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star3,[3,0])            
            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star4,[4,0])            
            
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star1,[1,0])
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star2,[2,0])
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star3,[3,0])
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star4,[4,0])
            
            iter_arr=dynamic_update_slice(iter_arr,iter1.reshape([1,]),[1,])
            iter_arr=dynamic_update_slice(iter_arr,iter2.reshape([1,]),[2,])
            iter_arr=dynamic_update_slice(iter_arr,iter3.reshape([1,]),[3,])
            iter_arr=dynamic_update_slice(iter_arr,iter4.reshape([1,]),[4,])
            
            xnarr=dynamic_update_slice(xnarr,xn0,[0,0])
            vnarr=dynamic_update_slice(vnarr,vn0,[0,0])
            
            xnarr=dynamic_update_slice(xnarr,xn1,[1,0])
            vnarr=dynamic_update_slice(vnarr,vn1,[1,0])
            xnarr=dynamic_update_slice(xnarr,xn2,[2,0])
            vnarr=dynamic_update_slice(vnarr,vn2,[2,0])
            xnarr=dynamic_update_slice(xnarr,xn3,[3,0])
            vnarr=dynamic_update_slice(vnarr,vn3,[3,0])
            xnarr=dynamic_update_slice(xnarr,xn4,[4,0])
            vnarr=dynamic_update_slice(vnarr,vn4,[4,0])
            
            return xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it
        
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it=jax.lax.fori_loop(0,numiter,inner,(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it))
        return xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it    

    def loop_OHO_UBU5(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it,hper2c,numiter):
        def inner(it,inp):
            xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it=inp
            key_it,subkey_it=random.split(key_it)
            xi=random.normal(subkey_it,(64,2,nbeta)) 
            
            xn0,vn0=singlestep_OHO(xnarr[0:1,0:nbeta],vnarr[0:1,0:nbeta],xi,hper2c)
            
            xn1,vn1,beta_star1,grad_beta_star1,iter1=multistep_UBU1(xnarr[1:2,0:nbeta],vnarr[1:2,0:nbeta],beta_star_arr[1:2,0:nbeta],grad_beta_star_arr[1:2,0:nbeta],iter_arr[1],xi,hper2c)
            xn2,vn2,beta_star2,grad_beta_star2,iter2=multistep_UBU2(xnarr[2:3,0:nbeta],vnarr[2:3,0:nbeta],beta_star_arr[2:3,0:nbeta],grad_beta_star_arr[2:3,0:nbeta],iter_arr[2],xi,hper2c)            
            xn3,vn3,beta_star3,grad_beta_star3,iter3=multistep_UBU3(xnarr[3:4,0:nbeta],vnarr[3:4,0:nbeta],beta_star_arr[3:4,0:nbeta],grad_beta_star_arr[3:4,0:nbeta],iter_arr[3],xi,hper2c)            
            xn4,vn4,beta_star4,grad_beta_star4,iter4=multistep_UBU4(xnarr[4:5,0:nbeta],vnarr[4:5,0:nbeta],beta_star_arr[4:5,0:nbeta],grad_beta_star_arr[4:5,0:nbeta],iter_arr[4],xi,hper2c)            
            xn5,vn5,beta_star5,grad_beta_star5,iter5=multistep_UBU5(xnarr[5:6,0:nbeta],vnarr[5:6,0:nbeta],beta_star_arr[5:6,0:nbeta],grad_beta_star_arr[5:6,0:nbeta],iter_arr[5],xi,hper2c)            

            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star1,[1,0])            
            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star2,[2,0])            
            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star3,[3,0])            
            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star4,[4,0])            
            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star5,[5,0])            
            
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star1,[1,0])
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star2,[2,0])
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star3,[3,0])
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star4,[4,0])
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star5,[5,0])
            
            iter_arr=dynamic_update_slice(iter_arr,iter1.reshape([1,]),[1,])
            iter_arr=dynamic_update_slice(iter_arr,iter2.reshape([1,]),[2,])
            iter_arr=dynamic_update_slice(iter_arr,iter3.reshape([1,]),[3,])
            iter_arr=dynamic_update_slice(iter_arr,iter4.reshape([1,]),[4,])
            iter_arr=dynamic_update_slice(iter_arr,iter5.reshape([1,]),[5,])
            
            xnarr=dynamic_update_slice(xnarr,xn0,[0,0])
            vnarr=dynamic_update_slice(vnarr,vn0,[0,0])
            
            xnarr=dynamic_update_slice(xnarr,xn1,[1,0])
            vnarr=dynamic_update_slice(vnarr,vn1,[1,0])
            xnarr=dynamic_update_slice(xnarr,xn2,[2,0])
            vnarr=dynamic_update_slice(vnarr,vn2,[2,0])
            xnarr=dynamic_update_slice(xnarr,xn3,[3,0])
            vnarr=dynamic_update_slice(vnarr,vn3,[3,0])
            xnarr=dynamic_update_slice(xnarr,xn4,[4,0])
            vnarr=dynamic_update_slice(vnarr,vn4,[4,0])
            xnarr=dynamic_update_slice(xnarr,xn5,[5,0])
            vnarr=dynamic_update_slice(vnarr,vn5,[5,0])
            
            return xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it
    
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it=jax.lax.fori_loop(0,numiter,inner,(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it))
        return xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it

    def loop_OHO_UBU6(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it,hper2c,numiter):
        def inner(it,inp):
            xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it=inp
            key_it,subkey_it=random.split(key_it)
            # xi shape is (2**(6+1), 2, nbeta) -> (128, 2, nbeta)
            xi=random.normal(subkey_it,(128,2,nbeta)) 
            
            xn0,vn0=singlestep_OHO(xnarr[0:1,0:nbeta],vnarr[0:1,0:nbeta],xi,hper2c)
            
            xn1,vn1,beta_star1,grad_beta_star1,iter1=multistep_UBU1(xnarr[1:2,0:nbeta],vnarr[1:2,0:nbeta],beta_star_arr[1:2,0:nbeta],grad_beta_star_arr[1:2,0:nbeta],iter_arr[1],xi,hper2c)
            xn2,vn2,beta_star2,grad_beta_star2,iter2=multistep_UBU2(xnarr[2:3,0:nbeta],vnarr[2:3,0:nbeta],beta_star_arr[2:3,0:nbeta],grad_beta_star_arr[2:3,0:nbeta],iter_arr[2],xi,hper2c)            
            xn3,vn3,beta_star3,grad_beta_star3,iter3=multistep_UBU3(xnarr[3:4,0:nbeta],vnarr[3:4,0:nbeta],beta_star_arr[3:4,0:nbeta],grad_beta_star_arr[3:4,0:nbeta],iter_arr[3],xi,hper2c)            
            xn4,vn4,beta_star4,grad_beta_star4,iter4=multistep_UBU4(xnarr[4:5,0:nbeta],vnarr[4:5,0:nbeta],beta_star_arr[4:5,0:nbeta],grad_beta_star_arr[4:5,0:nbeta],iter_arr[4],xi,hper2c)            
            xn5,vn5,beta_star5,grad_beta_star5,iter5=multistep_UBU5(xnarr[5:6,0:nbeta],vnarr[5:6,0:nbeta],beta_star_arr[5:6,0:nbeta],grad_beta_star_arr[5:6,0:nbeta],iter_arr[5],xi,hper2c)            
            xn6,vn6,beta_star6,grad_beta_star6,iter6=multistep_UBU6(xnarr[6:7,0:nbeta],vnarr[6:7,0:nbeta],beta_star_arr[6:7,0:nbeta],grad_beta_star_arr[6:7,0:nbeta],iter_arr[6],xi,hper2c)            

            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star1,[1,0])            
            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star2,[2,0])            
            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star3,[3,0])            
            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star4,[4,0])            
            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star5,[5,0])            
            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star6,[6,0])            
            
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star1,[1,0])
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star2,[2,0])
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star3,[3,0])
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star4,[4,0])
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star5,[5,0])
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star6,[6,0])
            
            iter_arr=dynamic_update_slice(iter_arr,iter1.reshape([1,]),[1,])
            iter_arr=dynamic_update_slice(iter_arr,iter2.reshape([1,]),[2,])
            iter_arr=dynamic_update_slice(iter_arr,iter3.reshape([1,]),[3,])
            iter_arr=dynamic_update_slice(iter_arr,iter4.reshape([1,]),[4,])
            iter_arr=dynamic_update_slice(iter_arr,iter5.reshape([1,]),[5,])
            iter_arr=dynamic_update_slice(iter_arr,iter6.reshape([1,]),[6,])
            
            xnarr=dynamic_update_slice(xnarr,xn0,[0,0])
            vnarr=dynamic_update_slice(vnarr,vn0,[0,0])
            
            xnarr=dynamic_update_slice(xnarr,xn1,[1,0])
            vnarr=dynamic_update_slice(vnarr,vn1,[1,0])
            xnarr=dynamic_update_slice(xnarr,xn2,[2,0])
            vnarr=dynamic_update_slice(vnarr,vn2,[2,0])
            xnarr=dynamic_update_slice(xnarr,xn3,[3,0])
            vnarr=dynamic_update_slice(vnarr,vn3,[3,0])
            xnarr=dynamic_update_slice(xnarr,xn4,[4,0])
            vnarr=dynamic_update_slice(vnarr,vn4,[4,0])
            xnarr=dynamic_update_slice(xnarr,xn5,[5,0])
            vnarr=dynamic_update_slice(vnarr,vn5,[5,0])
            xnarr=dynamic_update_slice(xnarr,xn6,[6,0])
            vnarr=dynamic_update_slice(vnarr,vn6,[6,0])
            
            return xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it
    
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it=jax.lax.fori_loop(0,numiter,inner,(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it))
        return xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it

    def loop_OHO_UBU7(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it,hper2c,numiter):
        def inner(it,inp):
            xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it=inp
            key_it,subkey_it=random.split(key_it)
            # xi shape is (2**(7+1), 2, nbeta) -> (256, 2, nbeta)
            xi=random.normal(subkey_it,(256,2,nbeta)) 
            
            xn0,vn0=singlestep_OHO(xnarr[0:1,0:nbeta],vnarr[0:1,0:nbeta],xi,hper2c)
            
            xn1,vn1,beta_star1,grad_beta_star1,iter1=multistep_UBU1(xnarr[1:2,0:nbeta],vnarr[1:2,0:nbeta],beta_star_arr[1:2,0:nbeta],grad_beta_star_arr[1:2,0:nbeta],iter_arr[1],xi,hper2c)
            xn2,vn2,beta_star2,grad_beta_star2,iter2=multistep_UBU2(xnarr[2:3,0:nbeta],vnarr[2:3,0:nbeta],beta_star_arr[2:3,0:nbeta],grad_beta_star_arr[2:3,0:nbeta],iter_arr[2],xi,hper2c)            
            xn3,vn3,beta_star3,grad_beta_star3,iter3=multistep_UBU3(xnarr[3:4,0:nbeta],vnarr[3:4,0:nbeta],beta_star_arr[3:4,0:nbeta],grad_beta_star_arr[3:4,0:nbeta],iter_arr[3],xi,hper2c)            
            xn4,vn4,beta_star4,grad_beta_star4,iter4=multistep_UBU4(xnarr[4:5,0:nbeta],vnarr[4:5,0:nbeta],beta_star_arr[4:5,0:nbeta],grad_beta_star_arr[4:5,0:nbeta],iter_arr[4],xi,hper2c)            
            xn5,vn5,beta_star5,grad_beta_star5,iter5=multistep_UBU5(xnarr[5:6,0:nbeta],vnarr[5:6,0:nbeta],beta_star_arr[5:6,0:nbeta],grad_beta_star_arr[5:6,0:nbeta],iter_arr[5],xi,hper2c)            
            xn6,vn6,beta_star6,grad_beta_star6,iter6=multistep_UBU6(xnarr[6:7,0:nbeta],vnarr[6:7,0:nbeta],beta_star_arr[6:7,0:nbeta],grad_beta_star_arr[6:7,0:nbeta],iter_arr[6],xi,hper2c)            
            xn7,vn7,beta_star7,grad_beta_star7,iter7=multistep_UBU7(xnarr[7:8,0:nbeta],vnarr[7:8,0:nbeta],beta_star_arr[7:8,0:nbeta],grad_beta_star_arr[7:8,0:nbeta],iter_arr[7],xi,hper2c)            

            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star1,[1,0])            
            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star2,[2,0])            
            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star3,[3,0])            
            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star4,[4,0])            
            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star5,[5,0])            
            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star6,[6,0])            
            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star7,[7,0])            
            
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star1,[1,0])
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star2,[2,0])
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star3,[3,0])
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star4,[4,0])
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star5,[5,0])
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star6,[6,0])
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star7,[7,0])
            
            iter_arr=dynamic_update_slice(iter_arr,iter1.reshape([1,]),[1,])
            iter_arr=dynamic_update_slice(iter_arr,iter2.reshape([1,]),[2,])
            iter_arr=dynamic_update_slice(iter_arr,iter3.reshape([1,]),[3,])
            iter_arr=dynamic_update_slice(iter_arr,iter4.reshape([1,]),[4,])
            iter_arr=dynamic_update_slice(iter_arr,iter5.reshape([1,]),[5,])
            iter_arr=dynamic_update_slice(iter_arr,iter6.reshape([1,]),[6,])
            iter_arr=dynamic_update_slice(iter_arr,iter7.reshape([1,]),[7,])
            
            xnarr=dynamic_update_slice(xnarr,xn0,[0,0])
            vnarr=dynamic_update_slice(vnarr,vn0,[0,0])
            
            xnarr=dynamic_update_slice(xnarr,xn1,[1,0])
            vnarr=dynamic_update_slice(vnarr,vn1,[1,0])
            xnarr=dynamic_update_slice(xnarr,xn2,[2,0])
            vnarr=dynamic_update_slice(vnarr,vn2,[2,0])
            xnarr=dynamic_update_slice(xnarr,xn3,[3,0])
            vnarr=dynamic_update_slice(vnarr,vn3,[3,0])
            xnarr=dynamic_update_slice(xnarr,xn4,[4,0])
            vnarr=dynamic_update_slice(vnarr,vn4,[4,0])
            xnarr=dynamic_update_slice(xnarr,xn5,[5,0])
            vnarr=dynamic_update_slice(vnarr,vn5,[5,0])
            xnarr=dynamic_update_slice(xnarr,xn6,[6,0])
            vnarr=dynamic_update_slice(vnarr,vn6,[6,0])
            xnarr=dynamic_update_slice(xnarr,xn7,[7,0])
            vnarr=dynamic_update_slice(vnarr,vn7,[7,0])
            
            return xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it
    
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it=jax.lax.fori_loop(0,numiter,inner,(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it))
        return xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it

    def loop_OHO_UBU8(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it,hper2c,numiter):
        def inner(it,inp):
            xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it=inp
            key_it,subkey_it=random.split(key_it)
            # xi shape is (2**(8+1), 2, nbeta) -> (512, 2, nbeta)
            xi=random.normal(subkey_it,(512,2,nbeta)) 
            
            xn0,vn0=singlestep_OHO(xnarr[0:1,0:nbeta],vnarr[0:1,0:nbeta],xi,hper2c)
            
            xn1,vn1,beta_star1,grad_beta_star1,iter1=multistep_UBU1(xnarr[1:2,0:nbeta],vnarr[1:2,0:nbeta],beta_star_arr[1:2,0:nbeta],grad_beta_star_arr[1:2,0:nbeta],iter_arr[1],xi,hper2c)
            xn2,vn2,beta_star2,grad_beta_star2,iter2=multistep_UBU2(xnarr[2:3,0:nbeta],vnarr[2:3,0:nbeta],beta_star_arr[2:3,0:nbeta],grad_beta_star_arr[2:3,0:nbeta],iter_arr[2],xi,hper2c)            
            xn3,vn3,beta_star3,grad_beta_star3,iter3=multistep_UBU3(xnarr[3:4,0:nbeta],vnarr[3:4,0:nbeta],beta_star_arr[3:4,0:nbeta],grad_beta_star_arr[3:4,0:nbeta],iter_arr[3],xi,hper2c)            
            xn4,vn4,beta_star4,grad_beta_star4,iter4=multistep_UBU4(xnarr[4:5,0:nbeta],vnarr[4:5,0:nbeta],beta_star_arr[4:5,0:nbeta],grad_beta_star_arr[4:5,0:nbeta],iter_arr[4],xi,hper2c)            
            xn5,vn5,beta_star5,grad_beta_star5,iter5=multistep_UBU5(xnarr[5:6,0:nbeta],vnarr[5:6,0:nbeta],beta_star_arr[5:6,0:nbeta],grad_beta_star_arr[5:6,0:nbeta],iter_arr[5],xi,hper2c)            
            xn6,vn6,beta_star6,grad_beta_star6,iter6=multistep_UBU6(xnarr[6:7,0:nbeta],vnarr[6:7,0:nbeta],beta_star_arr[6:7,0:nbeta],grad_beta_star_arr[6:7,0:nbeta],iter_arr[6],xi,hper2c)            
            xn7,vn7,beta_star7,grad_beta_star7,iter7=multistep_UBU7(xnarr[7:8,0:nbeta],vnarr[7:8,0:nbeta],beta_star_arr[7:8,0:nbeta],grad_beta_star_arr[7:8,0:nbeta],iter_arr[7],xi,hper2c)            
            xn8,vn8,beta_star8,grad_beta_star8,iter8=multistep_UBU8(xnarr[8:9,0:nbeta],vnarr[8:9,0:nbeta],beta_star_arr[8:9,0:nbeta],grad_beta_star_arr[8:9,0:nbeta],iter_arr[8],xi,hper2c)            

            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star1,[1,0])            
            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star2,[2,0])            
            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star3,[3,0])            
            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star4,[4,0])            
            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star5,[5,0])            
            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star6,[6,0])            
            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star7,[7,0])            
            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star8,[8,0])            
            
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star1,[1,0])
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star2,[2,0])
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star3,[3,0])
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star4,[4,0])
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star5,[5,0])
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star6,[6,0])
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star7,[7,0])
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star8,[8,0])
            
            iter_arr=dynamic_update_slice(iter_arr,iter1.reshape([1,]),[1,])
            iter_arr=dynamic_update_slice(iter_arr,iter2.reshape([1,]),[2,])
            iter_arr=dynamic_update_slice(iter_arr,iter3.reshape([1,]),[3,])
            iter_arr=dynamic_update_slice(iter_arr,iter4.reshape([1,]),[4,])
            iter_arr=dynamic_update_slice(iter_arr,iter5.reshape([1,]),[5,])
            iter_arr=dynamic_update_slice(iter_arr,iter6.reshape([1,]),[6,])
            iter_arr=dynamic_update_slice(iter_arr,iter7.reshape([1,]),[7,])
            iter_arr=dynamic_update_slice(iter_arr,iter8.reshape([1,]),[8,])
            
            xnarr=dynamic_update_slice(xnarr,xn0,[0,0])
            vnarr=dynamic_update_slice(vnarr,vn0,[0,0])
            
            xnarr=dynamic_update_slice(xnarr,xn1,[1,0])
            vnarr=dynamic_update_slice(vnarr,vn1,[1,0])
            xnarr=dynamic_update_slice(xnarr,xn2,[2,0])
            vnarr=dynamic_update_slice(vnarr,vn2,[2,0])
            xnarr=dynamic_update_slice(xnarr,xn3,[3,0])
            vnarr=dynamic_update_slice(vnarr,vn3,[3,0])
            xnarr=dynamic_update_slice(xnarr,xn4,[4,0])
            vnarr=dynamic_update_slice(vnarr,vn4,[4,0])
            xnarr=dynamic_update_slice(xnarr,xn5,[5,0])
            vnarr=dynamic_update_slice(vnarr,vn5,[5,0])
            xnarr=dynamic_update_slice(xnarr,xn6,[6,0])
            vnarr=dynamic_update_slice(vnarr,vn6,[6,0])
            xnarr=dynamic_update_slice(xnarr,xn7,[7,0])
            vnarr=dynamic_update_slice(vnarr,vn7,[7,0])
            xnarr=dynamic_update_slice(xnarr,xn8,[8,0])
            vnarr=dynamic_update_slice(vnarr,vn8,[8,0])
            
            return xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it
    
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it=jax.lax.fori_loop(0,numiter,inner,(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it))
        return xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it

    def loop_OHO_UBU9(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it,hper2c,numiter):
        def inner(it,inp):
            xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it=inp
            key_it,subkey_it=random.split(key_it)
            # xi shape is (2**(9+1), 2, nbeta) -> (1024, 2, nbeta)
            xi=random.normal(subkey_it,(1024,2,nbeta)) 
            
            xn0,vn0=singlestep_OHO(xnarr[0:1,0:nbeta],vnarr[0:1,0:nbeta],xi,hper2c)
            
            xn1,vn1,beta_star1,grad_beta_star1,iter1=multistep_UBU1(xnarr[1:2,0:nbeta],vnarr[1:2,0:nbeta],beta_star_arr[1:2,0:nbeta],grad_beta_star_arr[1:2,0:nbeta],iter_arr[1],xi,hper2c)
            xn2,vn2,beta_star2,grad_beta_star2,iter2=multistep_UBU2(xnarr[2:3,0:nbeta],vnarr[2:3,0:nbeta],beta_star_arr[2:3,0:nbeta],grad_beta_star_arr[2:3,0:nbeta],iter_arr[2],xi,hper2c)            
            xn3,vn3,beta_star3,grad_beta_star3,iter3=multistep_UBU3(xnarr[3:4,0:nbeta],vnarr[3:4,0:nbeta],beta_star_arr[3:4,0:nbeta],grad_beta_star_arr[3:4,0:nbeta],iter_arr[3],xi,hper2c)            
            xn4,vn4,beta_star4,grad_beta_star4,iter4=multistep_UBU4(xnarr[4:5,0:nbeta],vnarr[4:5,0:nbeta],beta_star_arr[4:5,0:nbeta],grad_beta_star_arr[4:5,0:nbeta],iter_arr[4],xi,hper2c)            
            xn5,vn5,beta_star5,grad_beta_star5,iter5=multistep_UBU5(xnarr[5:6,0:nbeta],vnarr[5:6,0:nbeta],beta_star_arr[5:6,0:nbeta],grad_beta_star_arr[5:6,0:nbeta],iter_arr[5],xi,hper2c)            
            xn6,vn6,beta_star6,grad_beta_star6,iter6=multistep_UBU6(xnarr[6:7,0:nbeta],vnarr[6:7,0:nbeta],beta_star_arr[6:7,0:nbeta],grad_beta_star_arr[6:7,0:nbeta],iter_arr[6],xi,hper2c)            
            xn7,vn7,beta_star7,grad_beta_star7,iter7=multistep_UBU7(xnarr[7:8,0:nbeta],vnarr[7:8,0:nbeta],beta_star_arr[7:8,0:nbeta],grad_beta_star_arr[7:8,0:nbeta],iter_arr[7],xi,hper2c)            
            xn8,vn8,beta_star8,grad_beta_star8,iter8=multistep_UBU8(xnarr[8:9,0:nbeta],vnarr[8:9,0:nbeta],beta_star_arr[8:9,0:nbeta],grad_beta_star_arr[8:9,0:nbeta],iter_arr[8],xi,hper2c)            
            xn9,vn9,beta_star9,grad_beta_star9,iter9=multistep_UBU9(xnarr[9:10,0:nbeta],vnarr[9:10,0:nbeta],beta_star_arr[9:10,0:nbeta],grad_beta_star_arr[9:10,0:nbeta],iter_arr[9],xi,hper2c)            

            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star1,[1,0])            
            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star2,[2,0])            
            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star3,[3,0])            
            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star4,[4,0])            
            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star5,[5,0])            
            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star6,[6,0])            
            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star7,[7,0])            
            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star8,[8,0])            
            beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star9,[9,0])            
            
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star1,[1,0])
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star2,[2,0])
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star3,[3,0])
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star4,[4,0])
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star5,[5,0])
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star6,[6,0])
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star7,[7,0])
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star8,[8,0])
            grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star9,[9,0])
            
            iter_arr=dynamic_update_slice(iter_arr,iter1.reshape([1,]),[1,])
            iter_arr=dynamic_update_slice(iter_arr,iter2.reshape([1,]),[2,])
            iter_arr=dynamic_update_slice(iter_arr,iter3.reshape([1,]),[3,])
            iter_arr=dynamic_update_slice(iter_arr,iter4.reshape([1,]),[4,])
            iter_arr=dynamic_update_slice(iter_arr,iter5.reshape([1,]),[5,])
            iter_arr=dynamic_update_slice(iter_arr,iter6.reshape([1,]),[6,])
            iter_arr=dynamic_update_slice(iter_arr,iter7.reshape([1,]),[7,])
            iter_arr=dynamic_update_slice(iter_arr,iter8.reshape([1,]),[8,])
            iter_arr=dynamic_update_slice(iter_arr,iter9.reshape([1,]),[9,])
            
            xnarr=dynamic_update_slice(xnarr,xn0,[0,0])
            vnarr=dynamic_update_slice(vnarr,vn0,[0,0])
            
            xnarr=dynamic_update_slice(xnarr,xn1,[1,0])
            vnarr=dynamic_update_slice(vnarr,vn1,[1,0])
            xnarr=dynamic_update_slice(xnarr,xn2,[2,0])
            vnarr=dynamic_update_slice(vnarr,vn2,[2,0])
            xnarr=dynamic_update_slice(xnarr,xn3,[3,0])
            vnarr=dynamic_update_slice(vnarr,vn3,[3,0])
            xnarr=dynamic_update_slice(xnarr,xn4,[4,0])
            vnarr=dynamic_update_slice(vnarr,vn4,[4,0])
            xnarr=dynamic_update_slice(xnarr,xn5,[5,0])
            vnarr=dynamic_update_slice(vnarr,vn5,[5,0])
            xnarr=dynamic_update_slice(xnarr,xn6,[6,0])
            vnarr=dynamic_update_slice(vnarr,vn6,[6,0])
            xnarr=dynamic_update_slice(xnarr,xn7,[7,0])
            vnarr=dynamic_update_slice(vnarr,vn7,[7,0])
            xnarr=dynamic_update_slice(xnarr,xn8,[8,0])
            vnarr=dynamic_update_slice(vnarr,vn8,[8,0])
            xnarr=dynamic_update_slice(xnarr,xn9,[9,0])
            vnarr=dynamic_update_slice(vnarr,vn9,[9,0])
            
            return xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it
    
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it=jax.lax.fori_loop(0,numiter,inner,(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it))
        return xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,key_it

    def multi0_inner(it,inp1):
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it=inp1
        key_it,subkey_it=random.split(key_it)
        xiarr=random.normal(subkey_it,(4,2,nbeta))
        xnarr_up0,vnarr_up0,beta_star0,grad_beta_star0,iter0=multistep_UBU0(xnarr[0:1,0:nbeta],vnarr[0:1,0:nbeta],beta_star_arr[0:1,0:nbeta],grad_beta_star_arr[0:1,0:nbeta],iter_arr[0],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up0,[0,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up0,[0,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star0,[0,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star0,[0,0])
        iter_arr=dynamic_update_slice(iter_arr,iter0.reshape([1,]),[0,])
        xnarr_up1,vnarr_up1,beta_star1,grad_beta_star1,iter1=multistep_UBU1(xnarr[1:2,0:nbeta],vnarr[1:2,0:nbeta],beta_star_arr[1:2,0:nbeta],grad_beta_star_arr[1:2,0:nbeta],iter_arr[1],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up1,[1,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up1,[1,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star1,[1,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star1,[1,0])
        iter_arr=dynamic_update_slice(iter_arr,iter1.reshape([1,]),[1,])
        return xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it
    
    def multi0_inner_thin(i,inp1):
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,meanx,meanxsquare,hper2c,key_it=inp1
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it=jax.lax.fori_loop(0,thin,multi0_inner,(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it))
        test_val=vmap_test_function(xnarr)
        meanx=meanx+test_val
        meanxsquare=meanxsquare+test_val**2
        return xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,meanx,meanxsquare,hper2c,key_it

    def multi0(inp):
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,niter,extraburnin2, jointburnin,thin, key_it=inp
        hper2c=dynamic_slice(hper2c_arr,(1,0),[1,6]).reshape((6,))

        xnarr01=xnarr[0:2,0:nbeta]
        vnarr01=vnarr[0:2,0:nbeta]
        beta_star_arr01=beta_star_arr[0:2,0:nbeta]
        grad_beta_star_arr01=grad_beta_star_arr[0:2,0:nbeta]
        iter_arr01=iter_arr[0:2]
        xnarr01,vnarr01,beta_star_arr01,grad_beta_star_arr01,iter_arr01,key_it=loop_OHO_UBU(xnarr01,vnarr01,beta_star_arr01,grad_beta_star_arr01,iter_arr01,key_it,hper2c,extraburnin2)
        xnarr=dynamic_update_slice(xnarr,xnarr01,[0,0])
        vnarr=dynamic_update_slice(vnarr,vnarr01,[0,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr01,[0,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr01,[0,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr01,[0,])

        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it=jax.lax.fori_loop(0,jointburnin,multi0_inner,(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it))

        meanx_single=jnp.zeros([maxmultilevels+2,test_dim])
        meanxsquare_single=jnp.zeros([maxmultilevels+2,test_dim])
        
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,meanx_single,meanxsquare_single,hper2c,key_it=jax.lax.fori_loop(0,niter//thin,multi0_inner_thin,(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,meanx_single,meanxsquare_single,hper2c,key_it))

        meanx_single=meanx_single/(niter//thin)
        meanxsquare_single=meanxsquare_single/(niter//thin)
        meanx_diff=meanx_single[1:(maxmultilevels+2),:]-meanx_single[0:(maxmultilevels+1),:]
        meanx_diff=dynamic_update_slice(meanx_diff,jnp.zeros([1,test_dim]),[multilevels+1,0])
        meanxsquare_diff=meanxsquare_single[1:(maxmultilevels+2),:]-meanxsquare_single[0:(maxmultilevels+1),:]
        meanxsquare_diff=dynamic_update_slice(meanxsquare_diff,jnp.zeros([1,test_dim]),[multilevels+1,0])
        err=jnp.mean(jnp.abs(dynamic_slice(xnarr,[1,0],[maxmultilevels,nbeta])-dynamic_slice(xnarr,[0,0],[maxmultilevels,nbeta])),axis=1)
        return meanx_diff,meanxsquare_diff,key_it,err

    
    def multi1_inner(it,inp1):
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it=inp1
        key_it,subkey_it=random.split(key_it)
        xiarr=random.normal(subkey_it,(8,2,nbeta))
        xnarr_up0,vnarr_up0,beta_star0,grad_beta_star0,iter0=multistep_UBU0(xnarr[0:1,0:nbeta],vnarr[0:1,0:nbeta],beta_star_arr[0:1,0:nbeta],grad_beta_star_arr[0:1,0:nbeta],iter_arr[0],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up0,[0,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up0,[0,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star0,[0,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star0,[0,0])
        iter_arr=dynamic_update_slice(iter_arr,iter0.reshape([1,]),[0,])

        xnarr_up1,vnarr_up1,beta_star1,grad_beta_star1,iter1=multistep_UBU1(xnarr[1:2,0:nbeta],vnarr[1:2,0:nbeta],beta_star_arr[1:2,0:nbeta],grad_beta_star_arr[1:2,0:nbeta],iter_arr[1],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up1,[1,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up1,[1,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star1,[1,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star1,[1,0])
        iter_arr=dynamic_update_slice(iter_arr,iter1.reshape([1,]),[1,])

        xnarr_up2,vnarr_up2,beta_star2,grad_beta_star2,iter2=multistep_UBU2(xnarr[2:3,0:nbeta],vnarr[2:3,0:nbeta],beta_star_arr[2:3,0:nbeta],grad_beta_star_arr[2:3,0:nbeta],iter_arr[2],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up2,[2,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up2,[2,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star2,[2,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star2,[2,0])
        iter_arr=dynamic_update_slice(iter_arr,iter2.reshape([1,]),[2,])

        return xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it
    
    def multi1_inner_thin(i,inp1):
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,meanx,meanxsquare,hper2c,key_it=inp1
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it=jax.lax.fori_loop(0,thin,multi1_inner,(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it))
        test_val=vmap_test_function(xnarr)
        meanx=meanx+test_val
        meanxsquare=meanxsquare+test_val**2
        return xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,meanx,meanxsquare,hper2c,key_it

    def multi1(inp):
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,niter,extraburnin2, jointburnin,thin, key_it=inp
        hper2c=dynamic_slice(hper2c_arr,(2,0),[1,6]).reshape((6,))

        xnarr12=xnarr[1:3,0:nbeta]
        vnarr12=vnarr[1:3,0:nbeta]
        beta_star_arr12=beta_star_arr[1:3,0:nbeta]
        grad_beta_star_arr12=grad_beta_star_arr[1:3,0:nbeta]
        iter_arr12=iter_arr[1:3]
        xnarr12,vnarr12,beta_star_arr12,grad_beta_star_arr12,iter_arr12,key_it=loop_OHO_UBU(xnarr12,vnarr12,beta_star_arr12,grad_beta_star_arr12,iter_arr12,key_it,hper2c,extraburnin2*2)
        xnarr=dynamic_update_slice(xnarr,xnarr12,[1,0])
        vnarr=dynamic_update_slice(vnarr,vnarr12,[1,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr12,[1,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr12,[1,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr12,[1,])
        
        xnarr=dynamic_update_slice(xnarr,xnarr[1:2,0:nbeta],[0,0])
        vnarr=dynamic_update_slice(vnarr,vnarr[1:2,0:nbeta],[0,0])

        xnarr012=xnarr[0:3,0:nbeta]
        vnarr012=vnarr[0:3,0:nbeta]
        beta_star_arr012=beta_star_arr[0:3,0:nbeta]
        grad_beta_star_arr012=grad_beta_star_arr[0:3,0:nbeta]
        iter_arr012=iter_arr[0:3]
        xnarr012,vnarr012,beta_star_arr012,grad_beta_star_arr012,iter_arr012,key_it=loop_OHO_UBU2(xnarr012,vnarr012,beta_star_arr012,grad_beta_star_arr012,iter_arr012,key_it,hper2c,extraburnin2)
        xnarr=dynamic_update_slice(xnarr,xnarr012,[0,0])
        vnarr=dynamic_update_slice(vnarr,vnarr012,[0,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr012,[0,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr012,[0,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr012,[0,])

        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it=jax.lax.fori_loop(0,jointburnin,multi1_inner,(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it))

        meanx_single=jnp.zeros([maxmultilevels+2,test_dim])
        meanxsquare_single=jnp.zeros([maxmultilevels+2,test_dim])
        
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,meanx_single,meanxsquare_single,hper2c,key_it=jax.lax.fori_loop(0,niter//thin,multi1_inner_thin,(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,meanx_single,meanxsquare_single,hper2c,key_it))

        meanx_single=meanx_single/(niter//thin)
        meanxsquare_single=meanxsquare_single/(niter//thin)
        meanx_diff=meanx_single[1:(maxmultilevels+2),:]-meanx_single[0:(maxmultilevels+1),:]
        meanx_diff=dynamic_update_slice(meanx_diff,jnp.zeros([1,test_dim]),[multilevels+1,0])
        meanxsquare_diff=meanxsquare_single[1:(maxmultilevels+2),:]-meanxsquare_single[0:(maxmultilevels+1),:]
        meanxsquare_diff=dynamic_update_slice(meanxsquare_diff,jnp.zeros([1,test_dim]),[multilevels+1,0])
        err=jnp.mean(jnp.abs(dynamic_slice(xnarr,[1,0],[maxmultilevels,nbeta])-dynamic_slice(xnarr,[0,0],[maxmultilevels,nbeta])),axis=1)
        return meanx_diff,meanxsquare_diff,key_it,err

    def multi2_inner(it,inp1):
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it=inp1
        key_it,subkey_it=random.split(key_it)
        xiarr=random.normal(subkey_it,(16,2,nbeta)) 
        
        xnarr_up0,vnarr_up0,beta_star0,grad_beta_star0,iter0=multistep_UBU0(xnarr[0:1,0:nbeta],vnarr[0:1,0:nbeta],beta_star_arr[0:1,0:nbeta],grad_beta_star_arr[0:1,0:nbeta],iter_arr[0],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up0,[0,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up0,[0,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star0,[0,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star0,[0,0])
        iter_arr=dynamic_update_slice(iter_arr,iter0.reshape([1,]),[0,])

        xnarr_up1,vnarr_up1,beta_star1,grad_beta_star1,iter1=multistep_UBU1(xnarr[1:2,0:nbeta],vnarr[1:2,0:nbeta],beta_star_arr[1:2,0:nbeta],grad_beta_star_arr[1:2,0:nbeta],iter_arr[1],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up1,[1,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up1,[1,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star1,[1,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star1,[1,0])
        iter_arr=dynamic_update_slice(iter_arr,iter1.reshape([1,]),[1,])

        xnarr_up2,vnarr_up2,beta_star2,grad_beta_star2,iter2=multistep_UBU2(xnarr[2:3,0:nbeta],vnarr[2:3,0:nbeta],beta_star_arr[2:3,0:nbeta],grad_beta_star_arr[2:3,0:nbeta],iter_arr[2],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up2,[2,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up2,[2,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star2,[2,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star2,[2,0])
        iter_arr=dynamic_update_slice(iter_arr,iter2.reshape([1,]),[2,])

        xnarr_up3,vnarr_up3,beta_star3,grad_beta_star3,iter3=multistep_UBU3(xnarr[3:4,0:nbeta],vnarr[3:4,0:nbeta],beta_star_arr[3:4,0:nbeta],grad_beta_star_arr[3:4,0:nbeta],iter_arr[3],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up3,[3,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up3,[3,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star3,[3,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star3,[3,0])
        iter_arr=dynamic_update_slice(iter_arr,iter3.reshape([1,]),[3,])

        return xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it

    def multi2_inner_thin(i,inp1):
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,meanx,meanxsquare,hper2c,key_it=inp1
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it=jax.lax.fori_loop(0,thin,multi2_inner,(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it))
        test_val=vmap_test_function(xnarr)
        meanx=meanx+test_val
        meanxsquare=meanxsquare+test_val**2
        return xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,meanx,meanxsquare,hper2c,key_it

    def multi2(inp):
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,niter,extraburnin2, jointburnin,thin, key_it=inp
        hper2c=dynamic_slice(hper2c_arr,(3,0),[1,6]).reshape((6,))

        xnarr23=xnarr[2:4,0:nbeta]
        vnarr23=vnarr[2:4,0:nbeta]
        beta_star_arr23=beta_star_arr[2:4,0:nbeta]
        grad_beta_star_arr23=grad_beta_star_arr[2:4,0:nbeta]
        iter_arr23=iter_arr[2:4]
        xnarr23,vnarr23,beta_star_arr23,grad_beta_star_arr23,iter_arr23,key_it=loop_OHO_UBU(xnarr23,vnarr23,beta_star_arr23,grad_beta_star_arr23,iter_arr23,key_it,hper2c,extraburnin2*4)
        xnarr=dynamic_update_slice(xnarr,xnarr23,[2,0])
        vnarr=dynamic_update_slice(vnarr,vnarr23,[2,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr23,[2,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr23,[2,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr23,[2,])
        
        xnarr=dynamic_update_slice(xnarr,xnarr[2:3,0:nbeta],[1,0])
        vnarr=dynamic_update_slice(vnarr,vnarr[2:3,0:nbeta],[1,0])

        xnarr123=xnarr[1:4,0:nbeta]
        vnarr123=vnarr[1:4,0:nbeta]
        beta_star_arr123=beta_star_arr[1:4,0:nbeta]
        grad_beta_star_arr123=grad_beta_star_arr[1:4,0:nbeta]
        iter_arr123=iter_arr[1:4]
        xnarr123,vnarr123,beta_star_arr123,grad_beta_star_arr123,iter_arr123,key_it=loop_OHO_UBU2(xnarr123,vnarr123,beta_star_arr123,grad_beta_star_arr123,iter_arr123,key_it,hper2c,extraburnin2*2)
        xnarr=dynamic_update_slice(xnarr,xnarr123,[1,0])
        vnarr=dynamic_update_slice(vnarr,vnarr123,[1,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr123,[1,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr123,[1,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr123,[1,])

        xnarr=dynamic_update_slice(xnarr,xnarr[1:2,0:nbeta],[0,0])
        vnarr=dynamic_update_slice(vnarr,vnarr[1:2,0:nbeta],[0,0])

        xnarr0123=xnarr[0:4,0:nbeta]
        vnarr0123=vnarr[0:4,0:nbeta]
        beta_star_arr0123=beta_star_arr[0:4,0:nbeta]
        grad_beta_star_arr0123=grad_beta_star_arr[0:4,0:nbeta]
        iter_arr0123=iter_arr[0:4]
        xnarr0123,vnarr0123,beta_star_arr0123,grad_beta_star_arr0123,iter_arr0123,key_it=loop_OHO_UBU3(xnarr0123,vnarr0123,beta_star_arr0123,grad_beta_star_arr0123,iter_arr0123,key_it,hper2c,extraburnin2)
        xnarr=dynamic_update_slice(xnarr,xnarr0123,[0,0])
        vnarr=dynamic_update_slice(vnarr,vnarr0123,[0,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr0123,[0,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr0123,[0,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr0123,[0,])

        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it=jax.lax.fori_loop(0,jointburnin,multi2_inner,(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it))

        meanx_single=jnp.zeros([maxmultilevels+2,test_dim])
        meanxsquare_single=jnp.zeros([maxmultilevels+2,test_dim])
        
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,meanx_single,meanxsquare_single,hper2c,key_it=jax.lax.fori_loop(0,niter//thin,multi2_inner_thin,(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,meanx_single,meanxsquare_single,hper2c,key_it))

        meanx_single=meanx_single/(niter//thin)
        meanxsquare_single=meanxsquare_single/(niter//thin)
        meanx_diff=meanx_single[1:(maxmultilevels+2),:]-meanx_single[0:(maxmultilevels+1),:]
        meanx_diff=dynamic_update_slice(meanx_diff,jnp.zeros([1,test_dim]),[multilevels+1,0])
        meanxsquare_diff=meanxsquare_single[1:(maxmultilevels+2),:]-meanxsquare_single[0:(maxmultilevels+1),:]
        meanxsquare_diff=dynamic_update_slice(meanxsquare_diff,jnp.zeros([1,test_dim]),[multilevels+1,0])
        err=jnp.mean(jnp.abs(dynamic_slice(xnarr,[1,0],[maxmultilevels,nbeta])-dynamic_slice(xnarr,[0,0],[maxmultilevels,nbeta])),axis=1)
        return meanx_diff,meanxsquare_diff,key_it,err

    def multi3_inner(it,inp1):
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it=inp1
        key_it,subkey_it=random.split(key_it)
        xiarr=random.normal(subkey_it,(32,2,nbeta)) 
        
        xnarr_up0,vnarr_up0,beta_star0,grad_beta_star0,iter0=multistep_UBU0(xnarr[0:1,0:nbeta],vnarr[0:1,0:nbeta],beta_star_arr[0:1,0:nbeta],grad_beta_star_arr[0:1,0:nbeta],iter_arr[0],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up0,[0,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up0,[0,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star0,[0,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star0,[0,0])
        iter_arr=dynamic_update_slice(iter_arr,iter0.reshape([1,]),[0,])

        xnarr_up1,vnarr_up1,beta_star1,grad_beta_star1,iter1=multistep_UBU1(xnarr[1:2,0:nbeta],vnarr[1:2,0:nbeta],beta_star_arr[1:2,0:nbeta],grad_beta_star_arr[1:2,0:nbeta],iter_arr[1],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up1,[1,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up1,[1,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star1,[1,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star1,[1,0])
        iter_arr=dynamic_update_slice(iter_arr,iter1.reshape([1,]),[1,])

        xnarr_up2,vnarr_up2,beta_star2,grad_beta_star2,iter2=multistep_UBU2(xnarr[2:3,0:nbeta],vnarr[2:3,0:nbeta],beta_star_arr[2:3,0:nbeta],grad_beta_star_arr[2:3,0:nbeta],iter_arr[2],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up2,[2,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up2,[2,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star2,[2,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star2,[2,0])
        iter_arr=dynamic_update_slice(iter_arr,iter2.reshape([1,]),[2,])

        xnarr_up3,vnarr_up3,beta_star3,grad_beta_star3,iter3=multistep_UBU3(xnarr[3:4,0:nbeta],vnarr[3:4,0:nbeta],beta_star_arr[3:4,0:nbeta],grad_beta_star_arr[3:4,0:nbeta],iter_arr[3],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up3,[3,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up3,[3,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star3,[3,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star3,[3,0])
        iter_arr=dynamic_update_slice(iter_arr,iter3.reshape([1,]),[3,])

        xnarr_up4,vnarr_up4,beta_star4,grad_beta_star4,iter4=multistep_UBU4(xnarr[4:5,0:nbeta],vnarr[4:5,0:nbeta],beta_star_arr[4:5,0:nbeta],grad_beta_star_arr[4:5,0:nbeta],iter_arr[4],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up4,[4,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up4,[4,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star4,[4,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star4,[4,0])
        iter_arr=dynamic_update_slice(iter_arr,iter4.reshape([1,]),[4,])

        return xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it



    def multi3_inner_thin(i,inp1):
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,meanx,meanxsquare,hper2c,key_it=inp1
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it=jax.lax.fori_loop(0,thin,multi3_inner,(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it))
        test_val=vmap_test_function(xnarr)
        meanx=meanx+test_val
        meanxsquare=meanxsquare+test_val**2
        return xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,meanx,meanxsquare,hper2c,key_it
    
    def multi3(inp):
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,niter,extraburnin2, jointburnin,thin, key_it=inp
        hper2c=dynamic_slice(hper2c_arr,(4,0),[1,6]).reshape((6,))

        xnarr34=xnarr[3:5,0:nbeta]
        vnarr34=vnarr[3:5,0:nbeta]
        beta_star_arr34=beta_star_arr[3:5,0:nbeta]
        grad_beta_star_arr34=grad_beta_star_arr[3:5,0:nbeta]
        iter_arr34=iter_arr[3:5]
        xnarr34,vnarr34,beta_star_arr34,grad_beta_star_arr34,iter_arr34,key_it=loop_OHO_UBU(xnarr34,vnarr34,beta_star_arr34,grad_beta_star_arr34,iter_arr34,key_it,hper2c,extraburnin2*8)
        xnarr=dynamic_update_slice(xnarr,xnarr34,[3,0])
        vnarr=dynamic_update_slice(vnarr,vnarr34,[3,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr34,[3,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr34,[3,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr34,[3,])

        xnarr=dynamic_update_slice(xnarr,xnarr[3:4,0:nbeta],[2,0])
        vnarr=dynamic_update_slice(vnarr,vnarr[3:4,0:nbeta],[2,0])

        xnarr234=xnarr[2:5,0:nbeta]
        vnarr234=vnarr[2:5,0:nbeta]
        beta_star_arr234=beta_star_arr[2:5,0:nbeta]
        grad_beta_star_arr234=grad_beta_star_arr[2:5,0:nbeta]
        iter_arr234=iter_arr[2:5]
        xnarr234,vnarr234,beta_star_arr234,grad_beta_star_arr234,iter_arr234,key_it=loop_OHO_UBU2(xnarr234,vnarr234,beta_star_arr234,grad_beta_star_arr234,iter_arr234,key_it,hper2c,extraburnin2*4)
        xnarr=dynamic_update_slice(xnarr,xnarr234,[2,0])
        vnarr=dynamic_update_slice(vnarr,vnarr234,[2,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr234,[2,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr234,[2,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr234,[2,])
        
        xnarr=dynamic_update_slice(xnarr,xnarr[2:3,0:nbeta],[1,0])
        vnarr=dynamic_update_slice(vnarr,vnarr[2:3,0:nbeta],[1,0])

        xnarr1234=xnarr[1:5,0:nbeta]
        vnarr1234=vnarr[1:5,0:nbeta]
        beta_star_arr1234=beta_star_arr[1:5,0:nbeta]
        grad_beta_star_arr1234=grad_beta_star_arr[1:5,0:nbeta]
        iter_arr1234=iter_arr[1:5]
        xnarr1234,vnarr1234,beta_star_arr1234,grad_beta_star_arr1234,iter_arr1234,key_it=loop_OHO_UBU3(xnarr1234,vnarr1234,beta_star_arr1234,grad_beta_star_arr1234,iter_arr1234,key_it,hper2c,extraburnin2*2)
        xnarr=dynamic_update_slice(xnarr,xnarr1234,[1,0])
        vnarr=dynamic_update_slice(vnarr,vnarr1234,[1,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr1234,[1,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr1234,[1,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr1234,[1,])

        xnarr=dynamic_update_slice(xnarr,xnarr[1:2,0:nbeta],[0,0])
        vnarr=dynamic_update_slice(vnarr,vnarr[1:2,0:nbeta],[0,0])

        xnarr01234=xnarr[0:5,0:nbeta]
        vnarr01234=vnarr[0:5,0:nbeta]
        beta_star_arr01234=beta_star_arr[0:5,0:nbeta]
        grad_beta_star_arr01234=grad_beta_star_arr[0:5,0:nbeta]
        iter_arr01234=iter_arr[0:5]
        xnarr01234,vnarr01234,beta_star_arr01234,grad_beta_star_arr01234,iter_arr01234,key_it=loop_OHO_UBU4(xnarr01234,vnarr01234,beta_star_arr01234,grad_beta_star_arr01234,iter_arr01234,key_it,hper2c,extraburnin2)
        xnarr=dynamic_update_slice(xnarr,xnarr01234,[0,0])
        vnarr=dynamic_update_slice(vnarr,vnarr01234,[0,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr01234,[0,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr01234,[0,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr01234,[0,])

        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it=jax.lax.fori_loop(0,jointburnin,multi3_inner,(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it))

        meanx_single=jnp.zeros([maxmultilevels+2,test_dim])
        meanxsquare_single=jnp.zeros([maxmultilevels+2,test_dim])
        
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,meanx_single,meanxsquare_single,hper2c,key_it=jax.lax.fori_loop(0,niter//thin,multi3_inner_thin,(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,meanx_single,meanxsquare_single,hper2c,key_it))

        meanx_single=meanx_single/(niter//thin)
        meanxsquare_single=meanxsquare_single/(niter//thin)
        meanx_diff=meanx_single[1:(maxmultilevels+2),:]-meanx_single[0:(maxmultilevels+1),:]
        meanx_diff=dynamic_update_slice(meanx_diff,jnp.zeros([1,test_dim]),[multilevels+1,0])
        meanxsquare_diff=meanxsquare_single[1:(maxmultilevels+2),:]-meanxsquare_single[0:(maxmultilevels+1),:]
        meanxsquare_diff=dynamic_update_slice(meanxsquare_diff,jnp.zeros([1,test_dim]),[multilevels+1,0])
        err=jnp.mean(jnp.abs(dynamic_slice(xnarr,[1,0],[maxmultilevels,nbeta])-dynamic_slice(xnarr,[0,0],[maxmultilevels,nbeta])),axis=1)
        return meanx_diff,meanxsquare_diff,key_it,err

    def multi4_inner(it,inp1):
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it=inp1
        key_it,subkey_it=random.split(key_it)
        xiarr=random.normal(subkey_it,(64,2,nbeta)) 
        
        xnarr_up0,vnarr_up0,beta_star0,grad_beta_star0,iter0=multistep_UBU0(xnarr[0:1,0:nbeta],vnarr[0:1,0:nbeta],beta_star_arr[0:1,0:nbeta],grad_beta_star_arr[0:1,0:nbeta],iter_arr[0],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up0,[0,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up0,[0,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star0,[0,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star0,[0,0])
        iter_arr=dynamic_update_slice(iter_arr,iter0.reshape([1,]),[0,])

        xnarr_up1,vnarr_up1,beta_star1,grad_beta_star1,iter1=multistep_UBU1(xnarr[1:2,0:nbeta],vnarr[1:2,0:nbeta],beta_star_arr[1:2,0:nbeta],grad_beta_star_arr[1:2,0:nbeta],iter_arr[1],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up1,[1,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up1,[1,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star1,[1,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star1,[1,0])
        iter_arr=dynamic_update_slice(iter_arr,iter1.reshape([1,]),[1,])

        xnarr_up2,vnarr_up2,beta_star2,grad_beta_star2,iter2=multistep_UBU2(xnarr[2:3,0:nbeta],vnarr[2:3,0:nbeta],beta_star_arr[2:3,0:nbeta],grad_beta_star_arr[2:3,0:nbeta],iter_arr[2],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up2,[2,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up2,[2,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star2,[2,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star2,[2,0])
        iter_arr=dynamic_update_slice(iter_arr,iter2.reshape([1,]),[2,])

        xnarr_up3,vnarr_up3,beta_star3,grad_beta_star3,iter3=multistep_UBU3(xnarr[3:4,0:nbeta],vnarr[3:4,0:nbeta],beta_star_arr[3:4,0:nbeta],grad_beta_star_arr[3:4,0:nbeta],iter_arr[3],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up3,[3,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up3,[3,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star3,[3,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star3,[3,0])
        iter_arr=dynamic_update_slice(iter_arr,iter3.reshape([1,]),[3,])

        xnarr_up4,vnarr_up4,beta_star4,grad_beta_star4,iter4=multistep_UBU4(xnarr[4:5,0:nbeta],vnarr[4:5,0:nbeta],beta_star_arr[4:5,0:nbeta],grad_beta_star_arr[4:5,0:nbeta],iter_arr[4],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up4,[4,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up4,[4,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star4,[4,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star4,[4,0])
        iter_arr=dynamic_update_slice(iter_arr,iter4.reshape([1,]),[4,])

        xnarr_up5,vnarr_up5,beta_star5,grad_beta_star5,iter5=multistep_UBU5(xnarr[5:6,0:nbeta],vnarr[5:6,0:nbeta],beta_star_arr[5:6,0:nbeta],grad_beta_star_arr[5:6,0:nbeta],iter_arr[5],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up5,[5,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up5,[5,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star5,[5,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star5,[5,0])
        iter_arr=dynamic_update_slice(iter_arr,iter5.reshape([1,]),[5,])

        return xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it

    def multi4_inner_thin(i,inp1):
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,meanx,meanxsquare,hper2c,key_it=inp1
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it=jax.lax.fori_loop(0,thin,multi4_inner,(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it))
        test_val=vmap_test_function(xnarr)
        meanx=meanx+test_val
        meanxsquare=meanxsquare+test_val**2
        return xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,meanx,meanxsquare,hper2c,key_it
    
    def multi4(inp):
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,niter,extraburnin2, jointburnin,thin, key_it=inp
        hper2c=dynamic_slice(hper2c_arr,(5,0),[1,6]).reshape((6,))

        xnarr45=xnarr[4:6,0:nbeta]
        vnarr45=vnarr[4:6,0:nbeta]
        beta_star_arr45=beta_star_arr[4:6,0:nbeta]
        grad_beta_star_arr45=grad_beta_star_arr[4:6,0:nbeta]
        iter_arr45=iter_arr[4:6]
        xnarr45,vnarr45,beta_star_arr45,grad_beta_star_arr45,iter_arr45,key_it=loop_OHO_UBU(xnarr45,vnarr45,beta_star_arr45,grad_beta_star_arr45,iter_arr45,key_it,hper2c,extraburnin2*16)
        xnarr=dynamic_update_slice(xnarr,xnarr45,[4,0])
        vnarr=dynamic_update_slice(vnarr,vnarr45,[4,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr45,[4,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr45,[4,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr45,[4,])

        xnarr=dynamic_update_slice(xnarr,xnarr[4:5,0:nbeta],[3,0])
        vnarr=dynamic_update_slice(vnarr,vnarr[4:5,0:nbeta],[3,0])

        xnarr345=xnarr[3:6,0:nbeta]
        vnarr345=vnarr[3:6,0:nbeta]
        beta_star_arr345=beta_star_arr[3:6,0:nbeta]
        grad_beta_star_arr345=grad_beta_star_arr[3:6,0:nbeta]
        iter_arr345=iter_arr[3:6]
        xnarr345,vnarr345,beta_star_arr345,grad_beta_star_arr345,iter_arr345,key_it=loop_OHO_UBU2(xnarr345,vnarr345,beta_star_arr345,grad_beta_star_arr345,iter_arr345,key_it,hper2c,extraburnin2*8)
        xnarr=dynamic_update_slice(xnarr,xnarr345,[3,0])
        vnarr=dynamic_update_slice(vnarr,vnarr345,[3,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr345,[3,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr345,[3,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr345,[3,])
        
        xnarr=dynamic_update_slice(xnarr,xnarr[3:4,0:nbeta],[2,0])
        vnarr=dynamic_update_slice(vnarr,vnarr[3:4,0:nbeta],[2,0])

        xnarr2345=xnarr[2:6,0:nbeta]
        vnarr2345=vnarr[2:6,0:nbeta]
        beta_star_arr2345=beta_star_arr[2:6,0:nbeta]
        grad_beta_star_arr2345=grad_beta_star_arr[2:6,0:nbeta]
        iter_arr2345=iter_arr[2:6]
        xnarr2345,vnarr2345,beta_star_arr2345,grad_beta_star_arr2345,iter_arr2345,key_it=loop_OHO_UBU3(xnarr2345,vnarr2345,beta_star_arr2345,grad_beta_star_arr2345,iter_arr2345,key_it,hper2c,extraburnin2*4)
        xnarr=dynamic_update_slice(xnarr,xnarr2345,[2,0])
        vnarr=dynamic_update_slice(vnarr,vnarr2345,[2,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr2345,[2,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr2345,[2,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr2345,[2,])

        xnarr=dynamic_update_slice(xnarr,xnarr[2:3,0:nbeta],[1,0])
        vnarr=dynamic_update_slice(vnarr,vnarr[2:3,0:nbeta],[1,0])

        xnarr12345=xnarr[1:6,0:nbeta]
        vnarr12345=vnarr[1:6,0:nbeta]
        beta_star_arr12345=beta_star_arr[1:6,0:nbeta]
        grad_beta_star_arr12345=grad_beta_star_arr[1:6,0:nbeta]
        iter_arr12345=iter_arr[1:6]
        xnarr12345,vnarr12345,beta_star_arr12345,grad_beta_star_arr12345,iter_arr12345,key_it=loop_OHO_UBU4(xnarr12345,vnarr12345,beta_star_arr12345,grad_beta_star_arr12345,iter_arr12345,key_it,hper2c,extraburnin2*2)
        xnarr=dynamic_update_slice(xnarr,xnarr12345,[1,0])
        vnarr=dynamic_update_slice(vnarr,vnarr12345,[1,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr12345,[1,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr12345,[1,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr12345,[1,])

        xnarr=dynamic_update_slice(xnarr,xnarr[1:2,0:nbeta],[0,0])
        vnarr=dynamic_update_slice(vnarr,vnarr[1:2,0:nbeta],[0,0])

        xnarr012345=xnarr[0:6,0:nbeta]
        vnarr012345=vnarr[0:6,0:nbeta]
        beta_star_arr012345=beta_star_arr[0:6,0:nbeta]
        grad_beta_star_arr012345=grad_beta_star_arr[0:6,0:nbeta]
        iter_arr012345=iter_arr[0:6]
        xnarr012345,vnarr012345,beta_star_arr012345,grad_beta_star_arr012345,iter_arr012345,key_it=loop_OHO_UBU5(xnarr012345,vnarr012345,beta_star_arr012345,grad_beta_star_arr012345,iter_arr012345,key_it,hper2c,extraburnin2)
        xnarr=dynamic_update_slice(xnarr,xnarr012345,[0,0])
        vnarr=dynamic_update_slice(vnarr,vnarr012345,[0,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr012345,[0,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr012345,[0,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr012345,[0,])

        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it=jax.lax.fori_loop(0,jointburnin,multi4_inner,(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it))

        meanx_single=jnp.zeros([maxmultilevels+2,test_dim])
        meanxsquare_single=jnp.zeros([maxmultilevels+2,test_dim])
        
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,meanx_single,meanxsquare_single,hper2c,key_it=jax.lax.fori_loop(0,niter//thin,multi4_inner_thin,(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,meanx_single,meanxsquare_single,hper2c,key_it))

        meanx_single=meanx_single/(niter//thin)
        meanxsquare_single=meanxsquare_single/(niter//thin)
        meanx_diff=meanx_single[1:(maxmultilevels+2),:]-meanx_single[0:(maxmultilevels+1),:]
        meanx_diff=dynamic_update_slice(meanx_diff,jnp.zeros([1,test_dim]),[multilevels+1,0])
        meanxsquare_diff=meanxsquare_single[1:(maxmultilevels+2),:]-meanxsquare_single[0:(maxmultilevels+1),:]
        meanxsquare_diff=dynamic_update_slice(meanxsquare_diff,jnp.zeros([1,test_dim]),[multilevels+1,0])
        err=jnp.mean(jnp.abs(dynamic_slice(xnarr,[1,0],[maxmultilevels,nbeta])-dynamic_slice(xnarr,[0,0],[maxmultilevels,nbeta])),axis=1)
        return meanx_diff,meanxsquare_diff,key_it,err    

    def multi5_inner(it,inp1):
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it=inp1
        key_it,subkey_it=random.split(key_it)
        # xiarr shape is 2**(N+2) = 2**(5+2) = 128
        xiarr=random.normal(subkey_it,(128,2,nbeta)) 
        
        # --- Level 0 ---
        xnarr_up0,vnarr_up0,beta_star0,grad_beta_star0,iter0=multistep_UBU0(xnarr[0:1,0:nbeta],vnarr[0:1,0:nbeta],beta_star_arr[0:1,0:nbeta],grad_beta_star_arr[0:1,0:nbeta],iter_arr[0],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up0,[0,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up0,[0,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star0,[0,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star0,[0,0])
        iter_arr=dynamic_update_slice(iter_arr,iter0.reshape([1,]),[0,])

        # --- Level 1 ---
        xnarr_up1,vnarr_up1,beta_star1,grad_beta_star1,iter1=multistep_UBU1(xnarr[1:2,0:nbeta],vnarr[1:2,0:nbeta],beta_star_arr[1:2,0:nbeta],grad_beta_star_arr[1:2,0:nbeta],iter_arr[1],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up1,[1,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up1,[1,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star1,[1,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star1,[1,0])
        iter_arr=dynamic_update_slice(iter_arr,iter1.reshape([1,]),[1,])

        # --- Level 2 ---
        xnarr_up2,vnarr_up2,beta_star2,grad_beta_star2,iter2=multistep_UBU2(xnarr[2:3,0:nbeta],vnarr[2:3,0:nbeta],beta_star_arr[2:3,0:nbeta],grad_beta_star_arr[2:3,0:nbeta],iter_arr[2],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up2,[2,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up2,[2,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star2,[2,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star2,[2,0])
        iter_arr=dynamic_update_slice(iter_arr,iter2.reshape([1,]),[2,])

        # --- Level 3 ---
        xnarr_up3,vnarr_up3,beta_star3,grad_beta_star3,iter3=multistep_UBU3(xnarr[3:4,0:nbeta],vnarr[3:4,0:nbeta],beta_star_arr[3:4,0:nbeta],grad_beta_star_arr[3:4,0:nbeta],iter_arr[3],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up3,[3,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up3,[3,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star3,[3,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star3,[3,0])
        iter_arr=dynamic_update_slice(iter_arr,iter3.reshape([1,]),[3,])

        # --- Level 4 ---
        xnarr_up4,vnarr_up4,beta_star4,grad_beta_star4,iter4=multistep_UBU4(xnarr[4:5,0:nbeta],vnarr[4:5,0:nbeta],beta_star_arr[4:5,0:nbeta],grad_beta_star_arr[4:5,0:nbeta],iter_arr[4],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up4,[4,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up4,[4,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star4,[4,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star4,[4,0])
        iter_arr=dynamic_update_slice(iter_arr,iter4.reshape([1,]),[4,])

        # --- Level 5 ---
        xnarr_up5,vnarr_up5,beta_star5,grad_beta_star5,iter5=multistep_UBU5(xnarr[5:6,0:nbeta],vnarr[5:6,0:nbeta],beta_star_arr[5:6,0:nbeta],grad_beta_star_arr[5:6,0:nbeta],iter_arr[5],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up5,[5,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up5,[5,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star5,[5,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star5,[5,0])
        iter_arr=dynamic_update_slice(iter_arr,iter5.reshape([1,]),[5,])

        # --- Level 6 (New) ---
        xnarr_up6,vnarr_up6,beta_star6,grad_beta_star6,iter6=multistep_UBU6(xnarr[6:7,0:nbeta],vnarr[6:7,0:nbeta],beta_star_arr[6:7,0:nbeta],grad_beta_star_arr[6:7,0:nbeta],iter_arr[6],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up6,[6,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up6,[6,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star6,[6,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star6,[6,0])
        iter_arr=dynamic_update_slice(iter_arr,iter6.reshape([1,]),[6,])

        return xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it

    def multi5_inner_thin(i,inp1):
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,meanx,meanxsquare,hper2c,key_it=inp1
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it=jax.lax.fori_loop(0,thin,multi5_inner,(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it))
        test_val=vmap_test_function(xnarr)
        meanx=meanx+test_val
        meanxsquare=meanxsquare+test_val**2
        return xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,meanx,meanxsquare,hper2c,key_it
    
    def multi5(inp):
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,niter,extraburnin2, jointburnin,thin, key_it=inp
        # hper2c index is N+1 = 5+1 = 6
        hper2c=dynamic_slice(hper2c_arr,(6,0),[1,6]).reshape((6,))

        # --- Pyramid Burn-in Step 1 (Levels 5, 6) ---
        xnarr56=xnarr[5:7,0:nbeta]
        vnarr56=vnarr[5:7,0:nbeta]
        beta_star_arr56=beta_star_arr[5:7,0:nbeta]
        grad_beta_star_arr56=grad_beta_star_arr[5:7,0:nbeta]
        iter_arr56=iter_arr[5:7]
        xnarr56,vnarr56,beta_star_arr56,grad_beta_star_arr56,iter_arr56,key_it=loop_OHO_UBU(xnarr56,vnarr56,beta_star_arr56,grad_beta_star_arr56,iter_arr56,key_it,hper2c,extraburnin2*32)
        xnarr=dynamic_update_slice(xnarr,xnarr56,[5,0])
        vnarr=dynamic_update_slice(vnarr,vnarr56,[5,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr56,[5,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr56,[5,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr56,[5,])

        xnarr=dynamic_update_slice(xnarr,xnarr[5:6,0:nbeta],[4,0])
        vnarr=dynamic_update_slice(vnarr,vnarr[5:6,0:nbeta],[4,0])

        # --- Pyramid Burn-in Step 2 (Levels 4, 5, 6) ---
        xnarr456=xnarr[4:7,0:nbeta]
        vnarr456=vnarr[4:7,0:nbeta]
        beta_star_arr456=beta_star_arr[4:7,0:nbeta]
        grad_beta_star_arr456=grad_beta_star_arr[4:7,0:nbeta]
        iter_arr456=iter_arr[4:7]
        xnarr456,vnarr456,beta_star_arr456,grad_beta_star_arr456,iter_arr456,key_it=loop_OHO_UBU2(xnarr456,vnarr456,beta_star_arr456,grad_beta_star_arr456,iter_arr456,key_it,hper2c,extraburnin2*16)
        xnarr=dynamic_update_slice(xnarr,xnarr456,[4,0])
        vnarr=dynamic_update_slice(vnarr,vnarr456,[4,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr456,[4,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr456,[4,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr456,[4,])
        
        xnarr=dynamic_update_slice(xnarr,xnarr[4:5,0:nbeta],[3,0])
        vnarr=dynamic_update_slice(vnarr,vnarr[4:5,0:nbeta],[3,0])

        # --- Pyramid Burn-in Step 3 (Levels 3, 4, 5, 6) ---
        xnarr3456=xnarr[3:7,0:nbeta]
        vnarr3456=vnarr[3:7,0:nbeta]
        beta_star_arr3456=beta_star_arr[3:7,0:nbeta]
        grad_beta_star_arr3456=grad_beta_star_arr[3:7,0:nbeta]
        iter_arr3456=iter_arr[3:7]
        xnarr3456,vnarr3456,beta_star_arr3456,grad_beta_star_arr3456,iter_arr3456,key_it=loop_OHO_UBU3(xnarr3456,vnarr3456,beta_star_arr3456,grad_beta_star_arr3456,iter_arr3456,key_it,hper2c,extraburnin2*8)
        xnarr=dynamic_update_slice(xnarr,xnarr3456,[3,0])
        vnarr=dynamic_update_slice(vnarr,vnarr3456,[3,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr3456,[3,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr3456,[3,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr3456,[3,])

        xnarr=dynamic_update_slice(xnarr,xnarr[3:4,0:nbeta],[2,0])
        vnarr=dynamic_update_slice(vnarr,vnarr[3:4,0:nbeta],[2,0])

        # --- Pyramid Burn-in Step 4 (Levels 2, 3, 4, 5, 6) ---
        xnarr23456=xnarr[2:7,0:nbeta]
        vnarr23456=vnarr[2:7,0:nbeta]
        beta_star_arr23456=beta_star_arr[2:7,0:nbeta]
        grad_beta_star_arr23456=grad_beta_star_arr[2:7,0:nbeta]
        iter_arr23456=iter_arr[2:7]
        xnarr23456,vnarr23456,beta_star_arr23456,grad_beta_star_arr23456,iter_arr23456,key_it=loop_OHO_UBU4(xnarr23456,vnarr23456,beta_star_arr23456,grad_beta_star_arr23456,iter_arr23456,key_it,hper2c,extraburnin2*4)
        xnarr=dynamic_update_slice(xnarr,xnarr23456,[2,0])
        vnarr=dynamic_update_slice(vnarr,vnarr23456,[2,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr23456,[2,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr23456,[2,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr23456,[2,])

        xnarr=dynamic_update_slice(xnarr,xnarr[2:3,0:nbeta],[1,0])
        vnarr=dynamic_update_slice(vnarr,vnarr[2:3,0:nbeta],[1,0])

        # --- Pyramid Burn-in Step 5 (Levels 1, 2, 3, 4, 5, 6) ---
        xnarr123456=xnarr[1:7,0:nbeta]
        vnarr123456=vnarr[1:7,0:nbeta]
        beta_star_arr123456=beta_star_arr[1:7,0:nbeta]
        grad_beta_star_arr123456=grad_beta_star_arr[1:7,0:nbeta]
        iter_arr123456=iter_arr[1:7]
        xnarr123456,vnarr123456,beta_star_arr123456,grad_beta_star_arr123456,iter_arr123456,key_it=loop_OHO_UBU5(xnarr123456,vnarr123456,beta_star_arr123456,grad_beta_star_arr123456,iter_arr123456,key_it,hper2c,extraburnin2*2)
        xnarr=dynamic_update_slice(xnarr,xnarr123456,[1,0])
        vnarr=dynamic_update_slice(vnarr,vnarr123456,[1,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr123456,[1,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr123456,[1,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr123456,[1,])

        xnarr=dynamic_update_slice(xnarr,xnarr[1:2,0:nbeta],[0,0])
        vnarr=dynamic_update_slice(vnarr,vnarr[1:2,0:nbeta],[0,0])

        # --- Pyramid Burn-in Step 6 (Levels 0, 1, 2, 3, 4, 5, 6) ---
        xnarr0123456=xnarr[0:7,0:nbeta]
        vnarr0123456=vnarr[0:7,0:nbeta]
        beta_star_arr0123456=beta_star_arr[0:7,0:nbeta]
        grad_beta_star_arr0123456=grad_beta_star_arr[0:7,0:nbeta]
        iter_arr0123456=iter_arr[0:7]
        xnarr0123456,vnarr0123456,beta_star_arr0123456,grad_beta_star_arr0123456,iter_arr0123456,key_it=loop_OHO_UBU6(xnarr0123456,vnarr0123456,beta_star_arr0123456,grad_beta_star_arr0123456,iter_arr0123456,key_it,hper2c,extraburnin2)
        xnarr=dynamic_update_slice(xnarr,xnarr0123456,[0,0])
        vnarr=dynamic_update_slice(vnarr,vnarr0123456,[0,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr0123456,[0,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr0123456,[0,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr0123456,[0,])

        # --- Main Loops ---
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it=jax.lax.fori_loop(0,jointburnin,multi5_inner,(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it))

        meanx_single=jnp.zeros([maxmultilevels+2,test_dim])
        meanxsquare_single=jnp.zeros([maxmultilevels+2,test_dim])
        
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,meanx_single,meanxsquare_single,hper2c,key_it=jax.lax.fori_loop(0,niter//thin,multi5_inner_thin,(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,meanx_single,meanxsquare_single,hper2c,key_it))

        # --- Statistics ---
        meanx_single=meanx_single/(niter//thin)
        meanxsquare_single=meanxsquare_single/(niter//thin)
        meanx_diff=meanx_single[1:(maxmultilevels+2),:]-meanx_single[0:(maxmultilevels+1),:]
        meanx_diff=dynamic_update_slice(meanx_diff,jnp.zeros([1,test_dim]),[multilevels+1,0])
        meanxsquare_diff=meanxsquare_single[1:(maxmultilevels+2),:]-meanxsquare_single[0:(maxmultilevels+1),:]
        meanxsquare_diff=dynamic_update_slice(meanxsquare_diff,jnp.zeros([1,test_dim]),[multilevels+1,0])
        err=jnp.mean(jnp.abs(dynamic_slice(xnarr,[1,0],[maxmultilevels,nbeta])-dynamic_slice(xnarr,[0,0],[maxmultilevels,nbeta])),axis=1)
        return meanx_diff,meanxsquare_diff,key_it,err
    
    def multi6_inner(it,inp1):
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it=inp1
        key_it,subkey_it=random.split(key_it)
        # xiarr shape is 2**(N+2) = 2**(6+2) = 256
        xiarr=random.normal(subkey_it,(256,2,nbeta)) 
        
        # --- Level 0 ---
        xnarr_up0,vnarr_up0,beta_star0,grad_beta_star0,iter0=multistep_UBU0(xnarr[0:1,0:nbeta],vnarr[0:1,0:nbeta],beta_star_arr[0:1,0:nbeta],grad_beta_star_arr[0:1,0:nbeta],iter_arr[0],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up0,[0,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up0,[0,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star0,[0,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star0,[0,0])
        iter_arr=dynamic_update_slice(iter_arr,iter0.reshape([1,]),[0,])

        # --- Level 1 ---
        xnarr_up1,vnarr_up1,beta_star1,grad_beta_star1,iter1=multistep_UBU1(xnarr[1:2,0:nbeta],vnarr[1:2,0:nbeta],beta_star_arr[1:2,0:nbeta],grad_beta_star_arr[1:2,0:nbeta],iter_arr[1],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up1,[1,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up1,[1,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star1,[1,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star1,[1,0])
        iter_arr=dynamic_update_slice(iter_arr,iter1.reshape([1,]),[1,])

        # --- Level 2 ---
        xnarr_up2,vnarr_up2,beta_star2,grad_beta_star2,iter2=multistep_UBU2(xnarr[2:3,0:nbeta],vnarr[2:3,0:nbeta],beta_star_arr[2:3,0:nbeta],grad_beta_star_arr[2:3,0:nbeta],iter_arr[2],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up2,[2,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up2,[2,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star2,[2,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star2,[2,0])
        iter_arr=dynamic_update_slice(iter_arr,iter2.reshape([1,]),[2,])

        # --- Level 3 ---
        xnarr_up3,vnarr_up3,beta_star3,grad_beta_star3,iter3=multistep_UBU3(xnarr[3:4,0:nbeta],vnarr[3:4,0:nbeta],beta_star_arr[3:4,0:nbeta],grad_beta_star_arr[3:4,0:nbeta],iter_arr[3],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up3,[3,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up3,[3,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star3,[3,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star3,[3,0])
        iter_arr=dynamic_update_slice(iter_arr,iter3.reshape([1,]),[3,])

        # --- Level 4 ---
        xnarr_up4,vnarr_up4,beta_star4,grad_beta_star4,iter4=multistep_UBU4(xnarr[4:5,0:nbeta],vnarr[4:5,0:nbeta],beta_star_arr[4:5,0:nbeta],grad_beta_star_arr[4:5,0:nbeta],iter_arr[4],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up4,[4,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up4,[4,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star4,[4,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star4,[4,0])
        iter_arr=dynamic_update_slice(iter_arr,iter4.reshape([1,]),[4,])

        # --- Level 5 ---
        xnarr_up5,vnarr_up5,beta_star5,grad_beta_star5,iter5=multistep_UBU5(xnarr[5:6,0:nbeta],vnarr[5:6,0:nbeta],beta_star_arr[5:6,0:nbeta],grad_beta_star_arr[5:6,0:nbeta],iter_arr[5],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up5,[5,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up5,[5,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star5,[5,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star5,[5,0])
        iter_arr=dynamic_update_slice(iter_arr,iter5.reshape([1,]),[5,])

        # --- Level 6 ---
        xnarr_up6,vnarr_up6,beta_star6,grad_beta_star6,iter6=multistep_UBU6(xnarr[6:7,0:nbeta],vnarr[6:7,0:nbeta],beta_star_arr[6:7,0:nbeta],grad_beta_star_arr[6:7,0:nbeta],iter_arr[6],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up6,[6,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up6,[6,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star6,[6,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star6,[6,0])
        iter_arr=dynamic_update_slice(iter_arr,iter6.reshape([1,]),[6,])

        # --- Level 7 (New) ---
        xnarr_up7,vnarr_up7,beta_star7,grad_beta_star7,iter7=multistep_UBU7(xnarr[7:8,0:nbeta],vnarr[7:8,0:nbeta],beta_star_arr[7:8,0:nbeta],grad_beta_star_arr[7:8,0:nbeta],iter_arr[7],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up7,[7,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up7,[7,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star7,[7,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star7,[7,0])
        iter_arr=dynamic_update_slice(iter_arr,iter7.reshape([1,]),[7,])

        return xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it

    def multi6_inner_thin(i,inp1):
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,meanx,meanxsquare,hper2c,key_it=inp1
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it=jax.lax.fori_loop(0,thin,multi6_inner,(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it))
        test_val=vmap_test_function(xnarr)
        meanx=meanx+test_val
        meanxsquare=meanxsquare+test_val**2
        return xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,meanx,meanxsquare,hper2c,key_it
    
    def multi6(inp):
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,niter,extraburnin2, jointburnin,thin, key_it=inp
        # hper2c index is N+1 = 6+1 = 7
        hper2c=dynamic_slice(hper2c_arr,(7,0),[1,6]).reshape((6,))

        # --- Pyramid Burn-in Step 1 (Levels 6, 7) ---
        xnarr67=xnarr[6:8,0:nbeta]
        vnarr67=vnarr[6:8,0:nbeta]
        beta_star_arr67=beta_star_arr[6:8,0:nbeta]
        grad_beta_star_arr67=grad_beta_star_arr[6:8,0:nbeta]
        iter_arr67=iter_arr[6:8]
        xnarr67,vnarr67,beta_star_arr67,grad_beta_star_arr67,iter_arr67,key_it=loop_OHO_UBU(xnarr67,vnarr67,beta_star_arr67,grad_beta_star_arr67,iter_arr67,key_it,hper2c,extraburnin2*64)
        xnarr=dynamic_update_slice(xnarr,xnarr67,[6,0])
        vnarr=dynamic_update_slice(vnarr,vnarr67,[6,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr67,[6,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr67,[6,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr67,[6,])

        xnarr=dynamic_update_slice(xnarr,xnarr[6:7,0:nbeta],[5,0])
        vnarr=dynamic_update_slice(vnarr,vnarr[6:7,0:nbeta],[5,0])

        # --- Pyramid Burn-in Step 2 (Levels 5, 6, 7) ---
        xnarr567=xnarr[5:8,0:nbeta]
        vnarr567=vnarr[5:8,0:nbeta]
        beta_star_arr567=beta_star_arr[5:8,0:nbeta]
        grad_beta_star_arr567=grad_beta_star_arr[5:8,0:nbeta]
        iter_arr567=iter_arr[5:8]
        xnarr567,vnarr567,beta_star_arr567,grad_beta_star_arr567,iter_arr567,key_it=loop_OHO_UBU2(xnarr567,vnarr567,beta_star_arr567,grad_beta_star_arr567,iter_arr567,key_it,hper2c,extraburnin2*32)
        xnarr=dynamic_update_slice(xnarr,xnarr567,[5,0])
        vnarr=dynamic_update_slice(vnarr,vnarr567,[5,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr567,[5,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr567,[5,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr567,[5,])
        
        xnarr=dynamic_update_slice(xnarr,xnarr[5:6,0:nbeta],[4,0])
        vnarr=dynamic_update_slice(vnarr,vnarr[5:6,0:nbeta],[4,0])

        # --- Pyramid Burn-in Step 3 (Levels 4, 5, 6, 7) ---
        xnarr4567=xnarr[4:8,0:nbeta]
        vnarr4567=vnarr[4:8,0:nbeta]
        beta_star_arr4567=beta_star_arr[4:8,0:nbeta]
        grad_beta_star_arr4567=grad_beta_star_arr[4:8,0:nbeta]
        iter_arr4567=iter_arr[4:8]
        xnarr4567,vnarr4567,beta_star_arr4567,grad_beta_star_arr4567,iter_arr4567,key_it=loop_OHO_UBU3(xnarr4567,vnarr4567,beta_star_arr4567,grad_beta_star_arr4567,iter_arr4567,key_it,hper2c,extraburnin2*16)
        xnarr=dynamic_update_slice(xnarr,xnarr4567,[4,0])
        vnarr=dynamic_update_slice(vnarr,vnarr4567,[4,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr4567,[4,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr4567,[4,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr4567,[4,])

        xnarr=dynamic_update_slice(xnarr,xnarr[4:5,0:nbeta],[3,0])
        vnarr=dynamic_update_slice(vnarr,vnarr[4:5,0:nbeta],[3,0])

        # --- Pyramid Burn-in Step 4 (Levels 3, 4, 5, 6, 7) ---
        xnarr34567=xnarr[3:8,0:nbeta]
        vnarr34567=vnarr[3:8,0:nbeta]
        beta_star_arr34567=beta_star_arr[3:8,0:nbeta]
        grad_beta_star_arr34567=grad_beta_star_arr[3:8,0:nbeta]
        iter_arr34567=iter_arr[3:8]
        xnarr34567,vnarr34567,beta_star_arr34567,grad_beta_star_arr34567,iter_arr34567,key_it=loop_OHO_UBU4(xnarr34567,vnarr34567,beta_star_arr34567,grad_beta_star_arr34567,iter_arr34567,key_it,hper2c,extraburnin2*8)
        xnarr=dynamic_update_slice(xnarr,xnarr34567,[3,0])
        vnarr=dynamic_update_slice(vnarr,vnarr34567,[3,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr34567,[3,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr34567,[3,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr34567,[3,])

        xnarr=dynamic_update_slice(xnarr,xnarr[3:4,0:nbeta],[2,0])
        vnarr=dynamic_update_slice(vnarr,vnarr[3:4,0:nbeta],[2,0])

        # --- Pyramid Burn-in Step 5 (Levels 2, 3, 4, 5, 6, 7) ---
        xnarr234567=xnarr[2:8,0:nbeta]
        vnarr234567=vnarr[2:8,0:nbeta]
        beta_star_arr234567=beta_star_arr[2:8,0:nbeta]
        grad_beta_star_arr234567=grad_beta_star_arr[2:8,0:nbeta]
        iter_arr234567=iter_arr[2:8]
        xnarr234567,vnarr234567,beta_star_arr234567,grad_beta_star_arr234567,iter_arr234567,key_it=loop_OHO_UBU5(xnarr234567,vnarr234567,beta_star_arr234567,grad_beta_star_arr234567,iter_arr234567,key_it,hper2c,extraburnin2*4)
        xnarr=dynamic_update_slice(xnarr,xnarr234567,[2,0])
        vnarr=dynamic_update_slice(vnarr,vnarr234567,[2,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr234567,[2,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr234567,[2,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr234567,[2,])

        xnarr=dynamic_update_slice(xnarr,xnarr[2:3,0:nbeta],[1,0])
        vnarr=dynamic_update_slice(vnarr,vnarr[2:3,0:nbeta],[1,0])

        # --- Pyramid Burn-in Step 6 (Levels 1, 2, 3, 4, 5, 6, 7) ---
        xnarr1234567=xnarr[1:8,0:nbeta]
        vnarr1234567=vnarr[1:8,0:nbeta]
        beta_star_arr1234567=beta_star_arr[1:8,0:nbeta]
        grad_beta_star_arr1234567=grad_beta_star_arr[1:8,0:nbeta]
        iter_arr1234567=iter_arr[1:8]
        xnarr1234567,vnarr1234567,beta_star_arr1234567,grad_beta_star_arr1234567,iter_arr1234567,key_it=loop_OHO_UBU6(xnarr1234567,vnarr1234567,beta_star_arr1234567,grad_beta_star_arr1234567,iter_arr1234567,key_it,hper2c,extraburnin2*2)
        xnarr=dynamic_update_slice(xnarr,xnarr1234567,[1,0])
        vnarr=dynamic_update_slice(vnarr,vnarr1234567,[1,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr1234567,[1,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr1234567,[1,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr1234567,[1,])

        xnarr=dynamic_update_slice(xnarr,xnarr[1:2,0:nbeta],[0,0])
        vnarr=dynamic_update_slice(vnarr,vnarr[1:2,0:nbeta],[0,0])

        # --- Pyramid Burn-in Step 7 (Levels 0, 1, 2, 3, 4, 5, 6, 7) ---
        xnarr01234567=xnarr[0:8,0:nbeta]
        vnarr01234567=vnarr[0:8,0:nbeta]
        beta_star_arr01234567=beta_star_arr[0:8,0:nbeta]
        grad_beta_star_arr01234567=grad_beta_star_arr[0:8,0:nbeta]
        iter_arr01234567=iter_arr[0:8]
        xnarr01234567,vnarr01234567,beta_star_arr01234567,grad_beta_star_arr01234567,iter_arr01234567,key_it=loop_OHO_UBU7(xnarr01234567,vnarr01234567,beta_star_arr01234567,grad_beta_star_arr01234567,iter_arr01234567,key_it,hper2c,extraburnin2)
        xnarr=dynamic_update_slice(xnarr,xnarr01234567,[0,0])
        vnarr=dynamic_update_slice(vnarr,vnarr01234567,[0,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr01234567,[0,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr01234567,[0,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr01234567,[0,])

        # --- Main Loops ---
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it=jax.lax.fori_loop(0,jointburnin,multi6_inner,(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it))

        meanx_single=jnp.zeros([maxmultilevels+2,test_dim])
        meanxsquare_single=jnp.zeros([maxmultilevels+2,test_dim])
        
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,meanx_single,meanxsquare_single,hper2c,key_it=jax.lax.fori_loop(0,niter//thin,multi6_inner_thin,(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,meanx_single,meanxsquare_single,hper2c,key_it))

        # --- Statistics ---
        meanx_single=meanx_single/(niter//thin)
        meanxsquare_single=meanxsquare_single/(niter//thin)
        meanx_diff=meanx_single[1:(maxmultilevels+2),:]-meanx_single[0:(maxmultilevels+1),:]
        meanx_diff=dynamic_update_slice(meanx_diff,jnp.zeros([1,test_dim]),[multilevels+1,0])
        meanxsquare_diff=meanxsquare_single[1:(maxmultilevels+2),:]-meanxsquare_single[0:(maxmultilevels+1),:]
        meanxsquare_diff=dynamic_update_slice(meanxsquare_diff,jnp.zeros([1,test_dim]),[multilevels+1,0])
        err=jnp.mean(jnp.abs(dynamic_slice(xnarr,[1,0],[maxmultilevels,nbeta])-dynamic_slice(xnarr,[0,0],[maxmultilevels,nbeta])),axis=1)
        return meanx_diff,meanxsquare_diff,key_it,err

    def multi7_inner(it,inp1):
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it=inp1
        key_it,subkey_it=random.split(key_it)
        # xiarr shape is 2**(N+2) = 2**(7+2) = 512
        xiarr=random.normal(subkey_it,(512,2,nbeta)) 
        
        # --- Level 0 ---
        xnarr_up0,vnarr_up0,beta_star0,grad_beta_star0,iter0=multistep_UBU0(xnarr[0:1,0:nbeta],vnarr[0:1,0:nbeta],beta_star_arr[0:1,0:nbeta],grad_beta_star_arr[0:1,0:nbeta],iter_arr[0],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up0,[0,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up0,[0,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star0,[0,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star0,[0,0])
        iter_arr=dynamic_update_slice(iter_arr,iter0.reshape([1,]),[0,])

        # --- Level 1 ---
        xnarr_up1,vnarr_up1,beta_star1,grad_beta_star1,iter1=multistep_UBU1(xnarr[1:2,0:nbeta],vnarr[1:2,0:nbeta],beta_star_arr[1:2,0:nbeta],grad_beta_star_arr[1:2,0:nbeta],iter_arr[1],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up1,[1,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up1,[1,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star1,[1,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star1,[1,0])
        iter_arr=dynamic_update_slice(iter_arr,iter1.reshape([1,]),[1,])

        # --- Level 2 ---
        xnarr_up2,vnarr_up2,beta_star2,grad_beta_star2,iter2=multistep_UBU2(xnarr[2:3,0:nbeta],vnarr[2:3,0:nbeta],beta_star_arr[2:3,0:nbeta],grad_beta_star_arr[2:3,0:nbeta],iter_arr[2],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up2,[2,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up2,[2,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star2,[2,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star2,[2,0])
        iter_arr=dynamic_update_slice(iter_arr,iter2.reshape([1,]),[2,])

        # --- Level 3 ---
        xnarr_up3,vnarr_up3,beta_star3,grad_beta_star3,iter3=multistep_UBU3(xnarr[3:4,0:nbeta],vnarr[3:4,0:nbeta],beta_star_arr[3:4,0:nbeta],grad_beta_star_arr[3:4,0:nbeta],iter_arr[3],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up3,[3,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up3,[3,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star3,[3,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star3,[3,0])
        iter_arr=dynamic_update_slice(iter_arr,iter3.reshape([1,]),[3,])

        # --- Level 4 ---
        xnarr_up4,vnarr_up4,beta_star4,grad_beta_star4,iter4=multistep_UBU4(xnarr[4:5,0:nbeta],vnarr[4:5,0:nbeta],beta_star_arr[4:5,0:nbeta],grad_beta_star_arr[4:5,0:nbeta],iter_arr[4],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up4,[4,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up4,[4,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star4,[4,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star4,[4,0])
        iter_arr=dynamic_update_slice(iter_arr,iter4.reshape([1,]),[4,])

        # --- Level 5 ---
        xnarr_up5,vnarr_up5,beta_star5,grad_beta_star5,iter5=multistep_UBU5(xnarr[5:6,0:nbeta],vnarr[5:6,0:nbeta],beta_star_arr[5:6,0:nbeta],grad_beta_star_arr[5:6,0:nbeta],iter_arr[5],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up5,[5,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up5,[5,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star5,[5,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star5,[5,0])
        iter_arr=dynamic_update_slice(iter_arr,iter5.reshape([1,]),[5,])

        # --- Level 6 ---
        xnarr_up6,vnarr_up6,beta_star6,grad_beta_star6,iter6=multistep_UBU6(xnarr[6:7,0:nbeta],vnarr[6:7,0:nbeta],beta_star_arr[6:7,0:nbeta],grad_beta_star_arr[6:7,0:nbeta],iter_arr[6],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up6,[6,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up6,[6,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star6,[6,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star6,[6,0])
        iter_arr=dynamic_update_slice(iter_arr,iter6.reshape([1,]),[6,])

        # --- Level 7 ---
        xnarr_up7,vnarr_up7,beta_star7,grad_beta_star7,iter7=multistep_UBU7(xnarr[7:8,0:nbeta],vnarr[7:8,0:nbeta],beta_star_arr[7:8,0:nbeta],grad_beta_star_arr[7:8,0:nbeta],iter_arr[7],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up7,[7,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up7,[7,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star7,[7,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star7,[7,0])
        iter_arr=dynamic_update_slice(iter_arr,iter7.reshape([1,]),[7,])

        # --- Level 8 (New) ---
        xnarr_up8,vnarr_up8,beta_star8,grad_beta_star8,iter8=multistep_UBU8(xnarr[8:9,0:nbeta],vnarr[8:9,0:nbeta],beta_star_arr[8:9,0:nbeta],grad_beta_star_arr[8:9,0:nbeta],iter_arr[8],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up8,[8,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up8,[8,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star8,[8,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star8,[8,0])
        iter_arr=dynamic_update_slice(iter_arr,iter8.reshape([1,]),[8,])

        return xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it

    def multi7_inner_thin(i,inp1):
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,meanx,meanxsquare,hper2c,key_it=inp1
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it=jax.lax.fori_loop(0,thin,multi7_inner,(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it))
        test_val=vmap_test_function(xnarr)
        meanx=meanx+test_val
        meanxsquare=meanxsquare+test_val**2
        return xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,meanx,meanxsquare,hper2c,key_it
    
    def multi7(inp):
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,niter,extraburnin2, jointburnin,thin, key_it=inp
        # hper2c index is N+1 = 7+1 = 8
        hper2c=dynamic_slice(hper2c_arr,(8,0),[1,6]).reshape((6,))

        # --- Pyramid Burn-in Step 1 (Levels 7, 8) ---
        xnarr78=xnarr[7:9,0:nbeta]
        vnarr78=vnarr[7:9,0:nbeta]
        beta_star_arr78=beta_star_arr[7:9,0:nbeta]
        grad_beta_star_arr78=grad_beta_star_arr[7:9,0:nbeta]
        iter_arr78=iter_arr[7:9]
        xnarr78,vnarr78,beta_star_arr78,grad_beta_star_arr78,iter_arr78,key_it=loop_OHO_UBU(xnarr78,vnarr78,beta_star_arr78,grad_beta_star_arr78,iter_arr78,key_it,hper2c,extraburnin2*128)
        xnarr=dynamic_update_slice(xnarr,xnarr78,[7,0])
        vnarr=dynamic_update_slice(vnarr,vnarr78,[7,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr78,[7,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr78,[7,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr78,[7,])

        xnarr=dynamic_update_slice(xnarr,xnarr[7:8,0:nbeta],[6,0])
        vnarr=dynamic_update_slice(vnarr,vnarr[7:8,0:nbeta],[6,0])

        # --- Pyramid Burn-in Step 2 (Levels 6, 7, 8) ---
        xnarr678=xnarr[6:9,0:nbeta]
        vnarr678=vnarr[6:9,0:nbeta]
        beta_star_arr678=beta_star_arr[6:9,0:nbeta]
        grad_beta_star_arr678=grad_beta_star_arr[6:9,0:nbeta]
        iter_arr678=iter_arr[6:9]
        xnarr678,vnarr678,beta_star_arr678,grad_beta_star_arr678,iter_arr678,key_it=loop_OHO_UBU2(xnarr678,vnarr678,beta_star_arr678,grad_beta_star_arr678,iter_arr678,key_it,hper2c,extraburnin2*64)
        xnarr=dynamic_update_slice(xnarr,xnarr678,[6,0])
        vnarr=dynamic_update_slice(vnarr,vnarr678,[6,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr678,[6,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr678,[6,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr678,[6,])
        
        xnarr=dynamic_update_slice(xnarr,xnarr[6:7,0:nbeta],[5,0])
        vnarr=dynamic_update_slice(vnarr,vnarr[6:7,0:nbeta],[5,0])

        # --- Pyramid Burn-in Step 3 (Levels 5, 6, 7, 8) ---
        xnarr5678=xnarr[5:9,0:nbeta]
        vnarr5678=vnarr[5:9,0:nbeta]
        beta_star_arr5678=beta_star_arr[5:9,0:nbeta]
        grad_beta_star_arr5678=grad_beta_star_arr[5:9,0:nbeta]
        iter_arr5678=iter_arr[5:9]
        xnarr5678,vnarr5678,beta_star_arr5678,grad_beta_star_arr5678,iter_arr5678,key_it=loop_OHO_UBU3(xnarr5678,vnarr5678,beta_star_arr5678,grad_beta_star_arr5678,iter_arr5678,key_it,hper2c,extraburnin2*32)
        xnarr=dynamic_update_slice(xnarr,xnarr5678,[5,0])
        vnarr=dynamic_update_slice(vnarr,vnarr5678,[5,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr5678,[5,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr5678,[5,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr5678,[5,])

        xnarr=dynamic_update_slice(xnarr,xnarr[5:6,0:nbeta],[4,0])
        vnarr=dynamic_update_slice(vnarr,vnarr[5:6,0:nbeta],[4,0])

        # --- Pyramid Burn-in Step 4 (Levels 4, 5, 6, 7, 8) ---
        xnarr45678=xnarr[4:9,0:nbeta]
        vnarr45678=vnarr[4:9,0:nbeta]
        beta_star_arr45678=beta_star_arr[4:9,0:nbeta]
        grad_beta_star_arr45678=grad_beta_star_arr[4:9,0:nbeta]
        iter_arr45678=iter_arr[4:9]
        xnarr45678,vnarr45678,beta_star_arr45678,grad_beta_star_arr45678,iter_arr45678,key_it=loop_OHO_UBU4(xnarr45678,vnarr45678,beta_star_arr45678,grad_beta_star_arr45678,iter_arr45678,key_it,hper2c,extraburnin2*16)
        xnarr=dynamic_update_slice(xnarr,xnarr45678,[4,0])
        vnarr=dynamic_update_slice(vnarr,vnarr45678,[4,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr45678,[4,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr45678,[4,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr45678,[4,])

        xnarr=dynamic_update_slice(xnarr,xnarr[4:5,0:nbeta],[3,0])
        vnarr=dynamic_update_slice(vnarr,vnarr[4:5,0:nbeta],[3,0])

        # --- Pyramid Burn-in Step 5 (Levels 3, 4, 5, 6, 7, 8) ---
        xnarr345678=xnarr[3:9,0:nbeta]
        vnarr345678=vnarr[3:9,0:nbeta]
        beta_star_arr345678=beta_star_arr[3:9,0:nbeta]
        grad_beta_star_arr345678=grad_beta_star_arr[3:9,0:nbeta]
        iter_arr345678=iter_arr[3:9]
        xnarr345678,vnarr345678,beta_star_arr345678,grad_beta_star_arr345678,iter_arr345678,key_it=loop_OHO_UBU5(xnarr345678,vnarr345678,beta_star_arr345678,grad_beta_star_arr345678,iter_arr345678,key_it,hper2c,extraburnin2*8)
        xnarr=dynamic_update_slice(xnarr,xnarr345678,[3,0])
        vnarr=dynamic_update_slice(vnarr,vnarr345678,[3,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr345678,[3,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr345678,[3,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr345678,[3,])

        xnarr=dynamic_update_slice(xnarr,xnarr[3:4,0:nbeta],[2,0])
        vnarr=dynamic_update_slice(vnarr,vnarr[3:4,0:nbeta],[2,0])

        # --- Pyramid Burn-in Step 6 (Levels 2, 3, 4, 5, 6, 7, 8) ---
        xnarr2345678=xnarr[2:9,0:nbeta]
        vnarr2345678=vnarr[2:9,0:nbeta]
        beta_star_arr2345678=beta_star_arr[2:9,0:nbeta]
        grad_beta_star_arr2345678=grad_beta_star_arr[2:9,0:nbeta]
        iter_arr2345678=iter_arr[2:9]
        xnarr2345678,vnarr2345678,beta_star_arr2345678,grad_beta_star_arr2345678,iter_arr2345678,key_it=loop_OHO_UBU6(xnarr2345678,vnarr2345678,beta_star_arr2345678,grad_beta_star_arr2345678,iter_arr2345678,key_it,hper2c,extraburnin2*4)
        xnarr=dynamic_update_slice(xnarr,xnarr2345678,[2,0])
        vnarr=dynamic_update_slice(vnarr,vnarr2345678,[2,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr2345678,[2,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr2345678,[2,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr2345678,[2,])

        xnarr=dynamic_update_slice(xnarr,xnarr[2:3,0:nbeta],[1,0])
        vnarr=dynamic_update_slice(vnarr,vnarr[2:3,0:nbeta],[1,0])

        # --- Pyramid Burn-in Step 7 (Levels 1, 2, 3, 4, 5, 6, 7, 8) ---
        xnarr12345678=xnarr[1:9,0:nbeta]
        vnarr12345678=vnarr[1:9,0:nbeta]
        beta_star_arr12345678=beta_star_arr[1:9,0:nbeta]
        grad_beta_star_arr12345678=grad_beta_star_arr[1:9,0:nbeta]
        iter_arr12345678=iter_arr[1:9]
        xnarr12345678,vnarr12345678,beta_star_arr12345678,grad_beta_star_arr12345678,iter_arr12345678,key_it=loop_OHO_UBU7(xnarr12345678,vnarr12345678,beta_star_arr12345678,grad_beta_star_arr12345678,iter_arr12345678,key_it,hper2c,extraburnin2*2)
        xnarr=dynamic_update_slice(xnarr,xnarr12345678,[1,0])
        vnarr=dynamic_update_slice(vnarr,vnarr12345678,[1,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr12345678,[1,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr12345678,[1,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr12345678,[1,])

        xnarr=dynamic_update_slice(xnarr,xnarr[1:2,0:nbeta],[0,0])
        vnarr=dynamic_update_slice(vnarr,vnarr[1:2,0:nbeta],[0,0])

        # --- Pyramid Burn-in Step 8 (Levels 0, 1, 2, 3, 4, 5, 6, 7, 8) ---
        xnarr012345678=xnarr[0:9,0:nbeta]
        vnarr012345678=vnarr[0:9,0:nbeta]
        beta_star_arr012345678=beta_star_arr[0:9,0:nbeta]
        grad_beta_star_arr012345678=grad_beta_star_arr[0:9,0:nbeta]
        iter_arr012345678=iter_arr[0:9]
        xnarr012345678,vnarr012345678,beta_star_arr012345678,grad_beta_star_arr012345678,iter_arr012345678,key_it=loop_OHO_UBU8(xnarr012345678,vnarr012345678,beta_star_arr012345678,grad_beta_star_arr012345678,iter_arr012345678,key_it,hper2c,extraburnin2)
        xnarr=dynamic_update_slice(xnarr,xnarr012345678,[0,0])
        vnarr=dynamic_update_slice(vnarr,vnarr012345678,[0,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr012345678,[0,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr012345678,[0,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr012345678,[0,])

        # --- Main Loops ---
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it=jax.lax.fori_loop(0,jointburnin,multi7_inner,(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it))

        meanx_single=jnp.zeros([maxmultilevels+2,test_dim])
        meanxsquare_single=jnp.zeros([maxmultilevels+2,test_dim])
        
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,meanx_single,meanxsquare_single,hper2c,key_it=jax.lax.fori_loop(0,niter//thin,multi7_inner_thin,(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,meanx_single,meanxsquare_single,hper2c,key_it))

        # --- Statistics ---
        meanx_single=meanx_single/(niter//thin)
        meanxsquare_single=meanxsquare_single/(niter//thin)
        meanx_diff=meanx_single[1:(maxmultilevels+2),:]-meanx_single[0:(maxmultilevels+1),:]
        meanx_diff=dynamic_update_slice(meanx_diff,jnp.zeros([1,test_dim]),[multilevels+1,0])
        meanxsquare_diff=meanxsquare_single[1:(maxmultilevels+2),:]-meanxsquare_single[0:(maxmultilevels+1),:]
        meanxsquare_diff=dynamic_update_slice(meanxsquare_diff,jnp.zeros([1,test_dim]),[multilevels+1,0])
        err=jnp.mean(jnp.abs(dynamic_slice(xnarr,[1,0],[maxmultilevels,nbeta])-dynamic_slice(xnarr,[0,0],[maxmultilevels,nbeta])),axis=1)
        return meanx_diff,meanxsquare_diff,key_it,err

    def multi8_inner(it,inp1):
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it=inp1
        key_it,subkey_it=random.split(key_it)
        # xiarr shape is 2**(N+2) = 2**(8+2) = 1024
        xiarr=random.normal(subkey_it,(1024,2,nbeta)) 
        
        # --- Level 0 ---
        xnarr_up0,vnarr_up0,beta_star0,grad_beta_star0,iter0=multistep_UBU0(xnarr[0:1,0:nbeta],vnarr[0:1,0:nbeta],beta_star_arr[0:1,0:nbeta],grad_beta_star_arr[0:1,0:nbeta],iter_arr[0],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up0,[0,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up0,[0,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star0,[0,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star0,[0,0])
        iter_arr=dynamic_update_slice(iter_arr,iter0.reshape([1,]),[0,])

        # --- Level 1 ---
        xnarr_up1,vnarr_up1,beta_star1,grad_beta_star1,iter1=multistep_UBU1(xnarr[1:2,0:nbeta],vnarr[1:2,0:nbeta],beta_star_arr[1:2,0:nbeta],grad_beta_star_arr[1:2,0:nbeta],iter_arr[1],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up1,[1,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up1,[1,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star1,[1,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star1,[1,0])
        iter_arr=dynamic_update_slice(iter_arr,iter1.reshape([1,]),[1,])

        # --- Level 2 ---
        xnarr_up2,vnarr_up2,beta_star2,grad_beta_star2,iter2=multistep_UBU2(xnarr[2:3,0:nbeta],vnarr[2:3,0:nbeta],beta_star_arr[2:3,0:nbeta],grad_beta_star_arr[2:3,0:nbeta],iter_arr[2],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up2,[2,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up2,[2,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star2,[2,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star2,[2,0])
        iter_arr=dynamic_update_slice(iter_arr,iter2.reshape([1,]),[2,])

        # --- Level 3 ---
        xnarr_up3,vnarr_up3,beta_star3,grad_beta_star3,iter3=multistep_UBU3(xnarr[3:4,0:nbeta],vnarr[3:4,0:nbeta],beta_star_arr[3:4,0:nbeta],grad_beta_star_arr[3:4,0:nbeta],iter_arr[3],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up3,[3,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up3,[3,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star3,[3,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star3,[3,0])
        iter_arr=dynamic_update_slice(iter_arr,iter3.reshape([1,]),[3,])

        # --- Level 4 ---
        xnarr_up4,vnarr_up4,beta_star4,grad_beta_star4,iter4=multistep_UBU4(xnarr[4:5,0:nbeta],vnarr[4:5,0:nbeta],beta_star_arr[4:5,0:nbeta],grad_beta_star_arr[4:5,0:nbeta],iter_arr[4],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up4,[4,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up4,[4,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star4,[4,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star4,[4,0])
        iter_arr=dynamic_update_slice(iter_arr,iter4.reshape([1,]),[4,])

        # --- Level 5 ---
        xnarr_up5,vnarr_up5,beta_star5,grad_beta_star5,iter5=multistep_UBU5(xnarr[5:6,0:nbeta],vnarr[5:6,0:nbeta],beta_star_arr[5:6,0:nbeta],grad_beta_star_arr[5:6,0:nbeta],iter_arr[5],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up5,[5,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up5,[5,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star5,[5,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star5,[5,0])
        iter_arr=dynamic_update_slice(iter_arr,iter5.reshape([1,]),[5,])

        # --- Level 6 ---
        xnarr_up6,vnarr_up6,beta_star6,grad_beta_star6,iter6=multistep_UBU6(xnarr[6:7,0:nbeta],vnarr[6:7,0:nbeta],beta_star_arr[6:7,0:nbeta],grad_beta_star_arr[6:7,0:nbeta],iter_arr[6],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up6,[6,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up6,[6,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star6,[6,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star6,[6,0])
        iter_arr=dynamic_update_slice(iter_arr,iter6.reshape([1,]),[6,])

        # --- Level 7 ---
        xnarr_up7,vnarr_up7,beta_star7,grad_beta_star7,iter7=multistep_UBU7(xnarr[7:8,0:nbeta],vnarr[7:8,0:nbeta],beta_star_arr[7:8,0:nbeta],grad_beta_star_arr[7:8,0:nbeta],iter_arr[7],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up7,[7,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up7,[7,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star7,[7,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star7,[7,0])
        iter_arr=dynamic_update_slice(iter_arr,iter7.reshape([1,]),[7,])

        # --- Level 8 ---
        xnarr_up8,vnarr_up8,beta_star8,grad_beta_star8,iter8=multistep_UBU8(xnarr[8:9,0:nbeta],vnarr[8:9,0:nbeta],beta_star_arr[8:9,0:nbeta],grad_beta_star_arr[8:9,0:nbeta],iter_arr[8],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up8,[8,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up8,[8,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star8,[8,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star8,[8,0])
        iter_arr=dynamic_update_slice(iter_arr,iter8.reshape([1,]),[8,])

        # --- Level 9 (New) ---
        xnarr_up9,vnarr_up9,beta_star9,grad_beta_star9,iter9=multistep_UBU9(xnarr[9:10,0:nbeta],vnarr[9:10,0:nbeta],beta_star_arr[9:10,0:nbeta],grad_beta_star_arr[9:10,0:nbeta],iter_arr[9],xiarr,hper2c)
        xnarr=dynamic_update_slice(xnarr,xnarr_up9,[9,0])
        vnarr=dynamic_update_slice(vnarr,vnarr_up9,[9,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star9,[9,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star9,[9,0])
        iter_arr=dynamic_update_slice(iter_arr,iter9.reshape([1,]),[9,])

        return xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it

    def multi8_inner_thin(i,inp1):
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,meanx,meanxsquare,hper2c,key_it=inp1
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it=jax.lax.fori_loop(0,thin,multi8_inner,(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it))
        test_val=vmap_test_function(xnarr)
        meanx=meanx+test_val
        meanxsquare=meanxsquare+test_val**2
        return xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,meanx,meanxsquare,hper2c,key_it
    
    def multi8(inp):
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,niter,extraburnin2, jointburnin,thin, key_it=inp
        # hper2c index is N+1 = 8+1 = 9
        hper2c=dynamic_slice(hper2c_arr,(9,0),[1,6]).reshape((6,))

        # --- Pyramid Burn-in Step 1 (Levels 8, 9) ---
        xnarr89=xnarr[8:10,0:nbeta]
        vnarr89=vnarr[8:10,0:nbeta]
        beta_star_arr89=beta_star_arr[8:10,0:nbeta]
        grad_beta_star_arr89=grad_beta_star_arr[8:10,0:nbeta]
        iter_arr89=iter_arr[8:10]
        xnarr89,vnarr89,beta_star_arr89,grad_beta_star_arr89,iter_arr89,key_it=loop_OHO_UBU(xnarr89,vnarr89,beta_star_arr89,grad_beta_star_arr89,iter_arr89,key_it,hper2c,extraburnin2*256)
        xnarr=dynamic_update_slice(xnarr,xnarr89,[8,0])
        vnarr=dynamic_update_slice(vnarr,vnarr89,[8,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr89,[8,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr89,[8,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr89,[8,])

        xnarr=dynamic_update_slice(xnarr,xnarr[8:9,0:nbeta],[7,0])
        vnarr=dynamic_update_slice(vnarr,vnarr[8:9,0:nbeta],[7,0])

        # --- Pyramid Burn-in Step 2 (Levels 7, 8, 9) ---
        xnarr789=xnarr[7:10,0:nbeta]
        vnarr789=vnarr[7:10,0:nbeta]
        beta_star_arr789=beta_star_arr[7:10,0:nbeta]
        grad_beta_star_arr789=grad_beta_star_arr[7:10,0:nbeta]
        iter_arr789=iter_arr[7:10]
        xnarr789,vnarr789,beta_star_arr789,grad_beta_star_arr789,iter_arr789,key_it=loop_OHO_UBU2(xnarr789,vnarr789,beta_star_arr789,grad_beta_star_arr789,iter_arr789,key_it,hper2c,extraburnin2*128)
        xnarr=dynamic_update_slice(xnarr,xnarr789,[7,0])
        vnarr=dynamic_update_slice(vnarr,vnarr789,[7,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr789,[7,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr789,[7,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr789,[7,])
        
        xnarr=dynamic_update_slice(xnarr,xnarr[7:8,0:nbeta],[6,0])
        vnarr=dynamic_update_slice(vnarr,vnarr[7:8,0:nbeta],[6,0])

        # --- Pyramid Burn-in Step 3 (Levels 6, 7, 8, 9) ---
        xnarr6789=xnarr[6:10,0:nbeta]
        vnarr6789=vnarr[6:10,0:nbeta]
        beta_star_arr6789=beta_star_arr[6:10,0:nbeta]
        grad_beta_star_arr6789=grad_beta_star_arr[6:10,0:nbeta]
        iter_arr6789=iter_arr[6:10]
        xnarr6789,vnarr6789,beta_star_arr6789,grad_beta_star_arr6789,iter_arr6789,key_it=loop_OHO_UBU3(xnarr6789,vnarr6789,beta_star_arr6789,grad_beta_star_arr6789,iter_arr6789,key_it,hper2c,extraburnin2*64)
        xnarr=dynamic_update_slice(xnarr,xnarr6789,[6,0])
        vnarr=dynamic_update_slice(vnarr,vnarr6789,[6,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr6789,[6,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr6789,[6,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr6789,[6,])

        xnarr=dynamic_update_slice(xnarr,xnarr[6:7,0:nbeta],[5,0])
        vnarr=dynamic_update_slice(vnarr,vnarr[6:7,0:nbeta],[5,0])

        # --- Pyramid Burn-in Step 4 (Levels 5, 6, 7, 8, 9) ---
        xnarr56789=xnarr[5:10,0:nbeta]
        vnarr56789=vnarr[5:10,0:nbeta]
        beta_star_arr56789=beta_star_arr[5:10,0:nbeta]
        grad_beta_star_arr56789=grad_beta_star_arr[5:10,0:nbeta]
        iter_arr56789=iter_arr[5:10]
        xnarr56789,vnarr56789,beta_star_arr56789,grad_beta_star_arr56789,iter_arr56789,key_it=loop_OHO_UBU4(xnarr56789,vnarr56789,beta_star_arr56789,grad_beta_star_arr56789,iter_arr56789,key_it,hper2c,extraburnin2*32)
        xnarr=dynamic_update_slice(xnarr,xnarr56789,[5,0])
        vnarr=dynamic_update_slice(vnarr,vnarr56789,[5,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr56789,[5,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr56789,[5,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr56789,[5,])

        xnarr=dynamic_update_slice(xnarr,xnarr[5:6,0:nbeta],[4,0])
        vnarr=dynamic_update_slice(vnarr,vnarr[5:6,0:nbeta],[4,0])

        # --- Pyramid Burn-in Step 5 (Levels 4, 5, 6, 7, 8, 9) ---
        xnarr456789=xnarr[4:10,0:nbeta]
        vnarr456789=vnarr[4:10,0:nbeta]
        beta_star_arr456789=beta_star_arr[4:10,0:nbeta]
        grad_beta_star_arr456789=grad_beta_star_arr[4:10,0:nbeta]
        iter_arr456789=iter_arr[4:10]
        xnarr456789,vnarr456789,beta_star_arr456789,grad_beta_star_arr456789,iter_arr456789,key_it=loop_OHO_UBU5(xnarr456789,vnarr456789,beta_star_arr456789,grad_beta_star_arr456789,iter_arr456789,key_it,hper2c,extraburnin2*16)
        xnarr=dynamic_update_slice(xnarr,xnarr456789,[4,0])
        vnarr=dynamic_update_slice(vnarr,vnarr456789,[4,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr456789,[4,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr456789,[4,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr456789,[4,])

        xnarr=dynamic_update_slice(xnarr,xnarr[4:5,0:nbeta],[3,0])
        vnarr=dynamic_update_slice(vnarr,vnarr[4:5,0:nbeta],[3,0])

        # --- Pyramid Burn-in Step 6 (Levels 3, 4, 5, 6, 7, 8, 9) ---
        xnarr3456789=xnarr[3:10,0:nbeta]
        vnarr3456789=vnarr[3:10,0:nbeta]
        beta_star_arr3456789=beta_star_arr[3:10,0:nbeta]
        grad_beta_star_arr3456789=grad_beta_star_arr[3:10,0:nbeta]
        iter_arr3456789=iter_arr[3:10]
        xnarr3456789,vnarr3456789,beta_star_arr3456789,grad_beta_star_arr3456789,iter_arr3456789,key_it=loop_OHO_UBU6(xnarr3456789,vnarr3456789,beta_star_arr3456789,grad_beta_star_arr3456789,iter_arr3456789,key_it,hper2c,extraburnin2*8)
        xnarr=dynamic_update_slice(xnarr,xnarr3456789,[3,0])
        vnarr=dynamic_update_slice(vnarr,vnarr3456789,[3,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr3456789,[3,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr3456789,[3,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr3456789,[3,])

        xnarr=dynamic_update_slice(xnarr,xnarr[3:4,0:nbeta],[2,0])
        vnarr=dynamic_update_slice(vnarr,vnarr[3:4,0:nbeta],[2,0])

        # --- Pyramid Burn-in Step 7 (Levels 2, 3, 4, 5, 6, 7, 8, 9) ---
        xnarr23456789=xnarr[2:10,0:nbeta]
        vnarr23456789=vnarr[2:10,0:nbeta]
        beta_star_arr23456789=beta_star_arr[2:10,0:nbeta]
        grad_beta_star_arr23456789=grad_beta_star_arr[2:10,0:nbeta]
        iter_arr23456789=iter_arr[2:10]
        xnarr23456789,vnarr23456789,beta_star_arr23456789,grad_beta_star_arr23456789,iter_arr23456789,key_it=loop_OHO_UBU7(xnarr23456789,vnarr23456789,beta_star_arr23456789,grad_beta_star_arr23456789,iter_arr23456789,key_it,hper2c,extraburnin2*4)
        xnarr=dynamic_update_slice(xnarr,xnarr23456789,[2,0])
        vnarr=dynamic_update_slice(vnarr,vnarr23456789,[2,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr23456789,[2,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr23456789,[2,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr23456789,[2,])

        xnarr=dynamic_update_slice(xnarr,xnarr[2:3,0:nbeta],[1,0])
        vnarr=dynamic_update_slice(vnarr,vnarr[2:3,0:nbeta],[1,0])

        # --- Pyramid Burn-in Step 8 (Levels 1, 2, 3, 4, 5, 6, 7, 8, 9) ---
        xnarr123456789=xnarr[1:10,0:nbeta]
        vnarr123456789=vnarr[1:10,0:nbeta]
        beta_star_arr123456789=beta_star_arr[1:10,0:nbeta]
        grad_beta_star_arr123456789=grad_beta_star_arr[1:10,0:nbeta]
        iter_arr123456789=iter_arr[1:10]
        xnarr123456789,vnarr123456789,beta_star_arr123456789,grad_beta_star_arr123456789,iter_arr123456789,key_it=loop_OHO_UBU8(xnarr123456789,vnarr123456789,beta_star_arr123456789,grad_beta_star_arr123456789,iter_arr123456789,key_it,hper2c,extraburnin2*2)
        xnarr=dynamic_update_slice(xnarr,xnarr123456789,[1,0])
        vnarr=dynamic_update_slice(vnarr,vnarr123456789,[1,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr123456789,[1,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr123456789,[1,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr123456789,[1,])

        xnarr=dynamic_update_slice(xnarr,xnarr[1:2,0:nbeta],[0,0])
        vnarr=dynamic_update_slice(vnarr,vnarr[1:2,0:nbeta],[0,0])

        # --- Pyramid Burn-in Step 9 (Levels 0, 1, 2, 3, 4, 5, 6, 7, 8, 9) ---
        xnarr0123456789=xnarr[0:10,0:nbeta]
        vnarr0123456789=vnarr[0:10,0:nbeta]
        beta_star_arr0123456789=beta_star_arr[0:10,0:nbeta]
        grad_beta_star_arr0123456789=grad_beta_star_arr[0:10,0:nbeta]
        iter_arr0123456789=iter_arr[0:10]
        xnarr0123456789,vnarr0123456789,beta_star_arr0123456789,grad_beta_star_arr0123456789,iter_arr0123456789,key_it=loop_OHO_UBU9(xnarr0123456789,vnarr0123456789,beta_star_arr0123456789,grad_beta_star_arr0123456789,iter_arr0123456789,key_it,hper2c,extraburnin2)
        xnarr=dynamic_update_slice(xnarr,xnarr0123456789,[0,0])
        vnarr=dynamic_update_slice(vnarr,vnarr0123456789,[0,0])
        beta_star_arr=dynamic_update_slice(beta_star_arr,beta_star_arr0123456789,[0,0])
        grad_beta_star_arr=dynamic_update_slice(grad_beta_star_arr,grad_beta_star_arr0123456789,[0,0])
        iter_arr=dynamic_update_slice(iter_arr,iter_arr0123456789,[0,])

        # --- Main Loops ---
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it=jax.lax.fori_loop(0,jointburnin,multi8_inner,(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,hper2c,key_it))

        meanx_single=jnp.zeros([maxmultilevels+2,test_dim])
        meanxsquare_single=jnp.zeros([maxmultilevels+2,test_dim])
        
        xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,meanx_single,meanxsquare_single,hper2c,key_it=jax.lax.fori_loop(0,niter//thin,multi8_inner_thin,(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,meanx_single,meanxsquare_single,hper2c,key_it))

        # --- Statistics ---
        meanx_single=meanx_single/(niter//thin)
        meanxsquare_single=meanxsquare_single/(niter//thin)
        meanx_diff=meanx_single[1:(maxmultilevels+2),:]-meanx_single[0:(maxmultilevels+1),:]
        meanx_diff=dynamic_update_slice(meanx_diff,jnp.zeros([1,test_dim]),[multilevels+1,0])
        meanxsquare_diff=meanxsquare_single[1:(maxmultilevels+2),:]-meanxsquare_single[0:(maxmultilevels+1),:]
        meanxsquare_diff=dynamic_update_slice(meanxsquare_diff,jnp.zeros([1,test_dim]),[multilevels+1,0])
        err=jnp.mean(jnp.abs(dynamic_slice(xnarr,[1,0],[maxmultilevels,nbeta])-dynamic_slice(xnarr,[0,0],[maxmultilevels,nbeta])),axis=1)
        return meanx_diff,meanxsquare_diff,key_it,err                



    key_it=key
    [key_it, subkey_it]=random.split(key_it)    
    x=x0
    v=random.normal(subkey_it,[1,nbeta])
    xnarr=jnp.ones([maxmultilevels+2,1])@stop_gradient(x)
    vnarr=jnp.ones([maxmultilevels+2,1])@stop_gradient(v)
    beta_star_arr=xnarr
    grad_beta_star_arr=jnp.zeros([maxmultilevels+2,nbeta])
    iter_arr=jnp.zeros(maxmultilevels+2,dtype=int)

    xnarr=jnp.ones([maxmultilevels+2,1])@stop_gradient(x)
    vnarr=jnp.ones([maxmultilevels+2,1])@stop_gradient(v)
    key_it=key
    match maxmultilevels:
        case 0:
            meanx,meanxsquare,key_it,err=multi0((xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,niter,extraburnin2, jointburnin,thin, key_it))
        case 1:
            meanx,meanxsquare,key_it,err=jax.lax.switch(multilevels,(multi0,multi1),(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,niter,extraburnin2, jointburnin,thin, key_it))
        case 2:
            meanx,meanxsquare,key_it,err=jax.lax.switch(multilevels,(multi0,multi1,multi2),(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,niter,extraburnin2, jointburnin,thin, key_it))
        case 3:
            meanx,meanxsquare,key_it,err=jax.lax.switch(multilevels,(multi0,multi1,multi2,multi3),(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,niter,extraburnin2, jointburnin,thin, key_it))
        case 4:
            meanx,meanxsquare,key_it,err=jax.lax.switch(multilevels,(multi0,multi1,multi2,multi3,multi4),(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,niter,extraburnin2, jointburnin,thin, key_it))
        case 5:
            meanx,meanxsquare,key_it,err=jax.lax.switch(multilevels,(multi0,multi1,multi2,multi3,multi4,multi5),(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,niter,extraburnin2, jointburnin,thin, key_it))
        case 6:
            meanx,meanxsquare,key_it,err=jax.lax.switch(multilevels,(multi0,multi1,multi2,multi3,multi4,multi5,multi6),(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,niter,extraburnin2, jointburnin,thin, key_it))
        case 7:
            meanx,meanxsquare,key_it,err=jax.lax.switch(multilevels,(multi0,multi1,multi2,multi3,multi4,multi5,multi6,multi7),(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,niter,extraburnin2, jointburnin,thin, key_it))
        case _:
            meanx,meanxsquare,key_it,err=jax.lax.switch(multilevels,(multi0,multi1,multi2,multi3,multi4,multi5,multi6,multi7,multi8),(xnarr,vnarr,beta_star_arr,grad_beta_star_arr,iter_arr,niter,extraburnin2, jointburnin,thin, key_it))    
    
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

def multilevel_ubu_approx(niter,burnin,rep,h,gam, repfullgrad,vmap_grad_lpost, vmap_test_function, invcholHprodv,Hprodv,exp_hM,r,c,nbeta,test_dim,beta_min,maxlevel,max_parallel_chain, max_Gaussian_samp, multilevels,key):
    
    key_it,sub_key0=random.split(key)

    doubleMCMC_levels01_jit=jax.jit(doubleMCMC_levels01,static_argnames=['vmap_grad_lpost','Hprodv','exp_hM','vmap_test_function','rep','nbeta','test_dim'])
    doubleMCMC_jit=jax.jit(doubleMCMC,static_argnames=['vmap_grad_lpost','Hprodv','exp_hM','vmap_test_function','rep','nbeta','test_dim'])


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
    
    ngradtot=0
    for lev in range(detlevels):
        extraburnin2=burnin*(2**(lev+1))
        jointburnin=burnin*(2**lev)*(lev+1)*(2+(lev>0))
        ngradtot+=repruns[lev+1]*(extraburnin2+jointburnin+niter*(2**lev)*(2+(lev>0)))/repfullgrad
    
    for lev in range(detlevels,maxlevel):
        extraburnin2=burnin*(2**(lev+1))
        jointburnin=burnin*(2**lev)*(lev+1)*3
        ngradtot+=problevels[lev+1]*(extraburnin2+jointburnin+niter*(2**lev)*3)/repfullgrad

    repruns[(detlevels+1):(maxlevel+1)]=0    

      
   
    rho=0.5
    means=jnp.zeros([maxruns,test_dim])
    squaremeans=jnp.zeros([maxruns,test_dim])
           
    
    no_batches=np.asarray(np.ceil(rep/max_parallel_chain), dtype=int)
    for repit in range(no_batches):
        if(repit<(no_batches-1)):
            par_chain=max_parallel_chain
        else:
            par_chain=rep-max_parallel_chain*(no_batches-1)
        key_it,subkey_it=random.split(key_it)
        means_up,squaremeans_up,_=Gaussian_samples(nbeta,par_chain,niter,test_dim,vmap_test_function,beta_min,invcholHprodv,max_Gaussian_samp,subkey_it)
        means=dynamic_update_slice(means,means_up,(repit*max_parallel_chain,0))
        squaremeans=dynamic_update_slice(squaremeans,squaremeans_up,(repit*max_parallel_chain,0))

    lev=0
    no_batches=np.asarray(np.ceil(repruns[lev+1]/max_parallel_chain), dtype=int)
    hper4c=jnp.asarray(hper2constarr(h/(2**(lev+1)),gam))
    for repit in range(no_batches):
        if(repit<(no_batches-1)):
            par_chain=max_parallel_chain
        else:
            par_chain=repruns[lev+1]-max_parallel_chain*(no_batches-1)
        key_it,subkey_it,subkey_it2=random.split(key_it,3)
        extraburnin2=burnin*2
        jointburnin=burnin
        thin=2**lev
        x0=jnp.ones([par_chain,1])@beta_min.reshape([1,nbeta])+invcholHprodv(random.normal(subkey_it,(par_chain,nbeta)))
        means_up,squaremeans_up,_,err=doubleMCMC_levels01_jit(nbeta,par_chain,niter,extraburnin2, jointburnin,thin, x0, hper4c, test_dim, beta_min,vmap_grad_lpost,vmap_test_function, Hprodv,exp_hM,repfullgrad,subkey_it2)
        means=dynamic_update_slice(means,means_up,(blockstart[lev+1]+repit*max_parallel_chain,0))
        squaremeans=dynamic_update_slice(squaremeans,squaremeans_up,(blockstart[lev+1]+repit*max_parallel_chain,0))
        
        print("Lev:",lev,"/",lev+1," xn/x2n error:", err)    

    for lev in range(1,detlevels-1):
        no_batches=np.asarray(np.ceil(repruns[lev+1]/max_parallel_chain), dtype=int)
        hper4c=jnp.asarray(hper2constarr(h/(2**(lev+1)),gam))
        for repit in range(no_batches):
            if(repit<(no_batches-1)):
                par_chain=max_parallel_chain
            else:
                par_chain=repruns[lev+1]-max_parallel_chain*(no_batches-1)
            key_it,subkey_it,subkey_it2=random.split(key_it,3)
            extraburnin2=burnin*(2**lev)
            jointburnin=burnin*(2**lev)*(lev+1)
            thin=2**lev
            x0=jnp.ones([par_chain,1])@beta_min.reshape([1,nbeta])+invcholHprodv(random.normal(subkey_it,(par_chain,nbeta)))
            means_up,squaremeans_up,_,err=doubleMCMC_jit(nbeta,par_chain,niter*(2**(lev)),extraburnin2, jointburnin,thin, x0,hper4c,test_dim, beta_min,vmap_grad_lpost,vmap_test_function, Hprodv,exp_hM,repfullgrad,subkey_it2)
            means=dynamic_update_slice(means,means_up,(blockstart[lev+1]+repit*max_parallel_chain,0))
            squaremeans=dynamic_update_slice(squaremeans,squaremeans_up,(blockstart[lev+1]+repit*max_parallel_chain,0))
            print("Lev:",lev,"/",lev+1,"xn/x2n error:", err)
            print("h:",h/(2**(lev)),"h/2:",h/(2**(lev+1)))

    repruns=jnp.asarray(repruns)

    if(detlevels<maxlevel):
        repruns_up=jnp.asarray(random.uniform(sub_key0,[1,randlevels])<problevels[(detlevels+1):(detlevels+1+randlevels)],dtype=int).reshape([randlevels,])
        repruns=dynamic_update_slice(repruns,repruns_up[0:multilevels],[detlevels+1,])

    maxmultilevels=randlevels
    hper2c_list=[None]*(maxmultilevels+2)
    print("Lev:",detlevels-1,"/",detlevels,"...")
    hm=h/(2**(detlevels-1))
    print("hm:",hm)
    for it in range(maxmultilevels+2):    
        hper2c_list[it]=hper2constarr(hm/(2**(it)),gam)
    hper2c_arr=jnp.stack(hper2c_list)
    
    key_it,subkey_it=random.split(key_it,2)
    multiMCMC_jit=jax.jit(multiMCMC,static_argnames=['nbeta','vmap_grad_lpost','vmap_test_function','Hprodv', 'exp_hM', 'test_dim','maxmultilevels'])
    x0=beta_min.reshape([1,nbeta])+invcholHprodv(random.normal(subkey_it,[1,nbeta]))
    means_up,squaremeans_up,key_it,err=multiMCMC_jit(nbeta,niter*(2**(detlevels-1)),extraburnin2, jointburnin,thin,multilevels,maxmultilevels,x0,vmap_grad_lpost,Hprodv, exp_hM, beta_min,vmap_test_function,test_dim, repfullgrad,key_it,hper2c_arr)

    print("Multi error:",err)
    means=dynamic_update_slice(means,means_up,(blockstart[detlevels],0))
    squaremeans=dynamic_update_slice(squaremeans,squaremeans_up,(blockstart[detlevels],0))
    
       
    return (rep,niter,detlevels,randlevels,repruns,maxlevel,problevels,ngradtot,nbeta,means,squaremeans,blockstart,blockend,test_dim,maxruns,rho)


vmultilevel_ubu_approx=jax.vmap(multilevel_ubu_approx,[None]*20+[0])



def vmap_multilevel_ubu_approx(niter,burnin,rep,h,gam, repfullgrad,vmap_grad_lpost, vmap_test_function, invcholHprodv,Hprodv,exp_hM,r,c,nbeta,test_dim,beta_min,maxlevel,max_parallel_chain, max_Gaussian_samp, keys, chunk_size):
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
        res[wit]=jax.device_put(vmultilevel_ubu_approx(niter,burnin,rep,h,gam, repfullgrad,vmap_grad_lpost, vmap_test_function, invcholHprodv,Hprodv,exp_hM,r,c,nbeta,test_dim,beta_min,maxlevel,max_parallel_chain, max_Gaussian_samp, max(multilevels_arr[wit*chunk_size:(wit+1)*chunk_size]), batch_key),cpus[0])
    if(par_runs%chunk_size>0):
        batch_key=keys[(par_runs//chunk_size)*chunk_size:]
        res[par_runs//chunk_size]=jax.device_put(vmultilevel_ubu_approx(niter,burnin,rep,h,gam, repfullgrad,vmap_grad_lpost, vmap_test_function, invcholHprodv,Hprodv,exp_hM,r,c,nbeta,test_dim,beta_min,maxlevel,max_parallel_chain, max_Gaussian_samp, max(multilevels_arr[(par_runs//chunk_size)*chunk_size:]), batch_key),cpus[0])

    res1=[]
    num_res=len(res)
    for it in range(len(res[0])):
        res0it=[]
        for it2 in range(num_res):
            res0it.append(res[it2][it])
        stacked=jnp.concatenate(res0it,axis=0)
        res1.append(stacked)
    return res1