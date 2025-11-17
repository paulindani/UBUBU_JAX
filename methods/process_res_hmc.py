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

def process_res_hmc(res,bind):
    means,squaremeans,_,ngradtot=res
    rep=means.shape[0]
    meansn=means[bind,]
    squaremeansn=squaremeans[bind,]
    test_mean=jnp.mean(meansn,0)
    test_mean_var=jnp.var(meansn,0)
    
    test_squaremean=jnp.mean(squaremeansn,0)
    test_post_var=test_squaremean-jnp.square(test_mean)
    ess=test_post_var/test_mean_var

    grad_per_ess=ngradtot/ess
    return ess,grad_per_ess