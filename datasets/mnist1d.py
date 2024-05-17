#! /usr/bin/env python3

import numpy as np

from scipy.ndimage import gaussian_filter

def mask(x, x0, x1):
    return 1.*(x >= x0)*(x < x1)

base_funcs = [
    #lambda x: np.abs(np.sin(x*2.*np.pi)) * mask(x, 0., 1.),
    #lambda x: -np.abs(np.sin(x*2.*np.pi)) * mask(x, 0., 1.),
    lambda x: np.sin(x*4.*np.pi) * mask(x, 0., 1.),
    lambda x: -np.sin(x*4.*np.pi) * mask(x, 0., 1.),
    lambda x: np.sin(x*8.*np.pi) * mask(x, 0., 1.),
    lambda x: -np.sin(x*8.*np.pi) * mask(x, 0., 1.),
    lambda x: np.sqrt(np.maximum(0.,0.25-(x-0.5)**2.)) * mask(x, 0., 1.),
    lambda x: -np.sqrt(np.maximum(0.,0.25-(x-0.5)**2.)) * mask(x, 0., 1.),
    #lambda x: 5.*x*mask(x,0.,0.2) + (1.-(x-.2)*10./3.)*mask(x,0.2,0.8) + (5.*(x-.8) - 1.)*mask(x,0.8,1.),
    #lambda x: -(5.*x*mask(x,0.,0.2) + (1.-(x-.2)*10./3.)*mask(x,0.2,0.8) + (5.*(x-.8) - 1.)*mask(x,0.8,1.)),
    lambda x: np.sign(np.sin(x*2.*np.pi)) * mask(x, 0., 1.),
    lambda x: -np.sign(np.sin(x*2.*np.pi)) * mask(x, 0., 1.),
    lambda x: 5.*x*mask(x,0.,0.2) + mask(x,0.2,0.8) + (1.-5.*(x-.8))*mask(x,0.8,1.),
    lambda x: -(5.*x*mask(x,0.,0.2) + mask(x,0.2,0.8) + (1.-5.*(x-.8))*mask(x,0.8,1.))
]

#base_funcs = base_funcs[4:6]
#base_funcs = [base_funcs[0], base_funcs[1], base_funcs[4], base_funcs[5]]

def mnist1d_dataset_array(samples,
                          list_base_funcs=np.arange(10).tolist(),
                          t=5., t_pattern=1.,
						  t0_attention=1., t1_attention=5.,
                          t_output_act=2.5,
                          width_output_act=1.0,
						  dt=0.01,
                          sigm_uncorr_noise = 0.1,
                          sigm_corr_noise = 0.1,
                          width_corr_noise = 0.1,
                          seed=50):
    
    _base_funcs = [base_funcs[k] for k in list_base_funcs]
    n_o = len(_base_funcs)

    np.random.seed(seed)

    nt = int(t/dt)

    if type(samples) is list:
        n_samples = len(samples)
        labels = samples
    elif type(samples) is int:
        n_samples = samples
        labels = np.repeat(np.arange(n_o), 1 + n_samples//n_o)

        np.random.shuffle(labels)
        labels = labels[:n_samples]
    else:
        raise TypeError("samples should be int or list")

    X = np.ndarray((n_samples, nt))
    Y = np.zeros((n_samples, nt, n_o))
    Z = np.zeros((n_samples, nt))

    t_ax = np.arange(nt) * dt

    for k in range(n_samples):
        X[k] = _base_funcs[labels[k]](t_ax/t_pattern)

        uncorr_noise = np.random.normal(0.,1.,(nt))
        uncorr_noise = (uncorr_noise - uncorr_noise.mean()) * sigm_uncorr_noise / uncorr_noise.std()
        corr_noise = gaussian_filter(np.random.normal(0.,1.,(nt)), sigma=width_corr_noise/dt, mode="constant", cval=0.)
        corr_noise = (corr_noise - corr_noise.mean()) * sigm_corr_noise / corr_noise.std()

        X[k] += uncorr_noise + corr_noise


        Y[k,:,labels[k]] = mask(t_ax, 0., t_pattern)
        #Y[k,:,labels[k]] = np.exp(-(t_ax-t_output_act)**2./(2.*width_output_act**2.))
        #Y[k,:,10] = 1.-Y[k,:,labels[k]]

        Z[k] = mask(t_ax, t0_attention, t1_attention)

    X_rs = np.expand_dims(X.flatten(), 1)
    Y_rs = np.reshape(Y, (Y.shape[0] * Y.shape[1], Y.shape[2]))
    Z_rs = Z.flatten()

    return X_rs, Y_rs, Z_rs