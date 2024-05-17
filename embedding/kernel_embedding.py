import numpy as np

def step_kernel_embedding(x, d_embedding, r_points, r_fact_kernel=1.25, seed=None):

    assert x.ndim in [2, 3]

    # if x is 2d, we assume it represents a single batch and add a batch dimension
    if x.ndim == 2:
        x = x[np.newaxis, :, :]

    n_batch = x.shape[0]

    if seed is not None:
        np.random.seed(seed)
    
    n_samples = x.shape[1]

    d_input = x.shape[2]

    # generate d_embedding random points uniformly distributed in a sphere in the input space
    p = np.random.randn(d_embedding, d_input)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    p = r_points * (p.T * (np.random.rand(d_embedding)**(1/d_input))).T

    # we want to express the squared distance between x and p in terms of a dot product plus a constant.
    # this is achieved by the following identity:
    # ||x - p||^2 = ||x||^2 + ||p||^2 - 2 * x * p.
    # we can express this as a dot product by augmenting the input space with the squared norm of x and a 1.
    # the weights are then given by -2 * p, augmented with 1 and the squared norm of p.

    x_augm = np.concatenate([x, np.linalg.norm(x, axis=2)[:,:,np.newaxis]**2., np.ones((n_batch, n_samples, 1))], axis=2)
    p_augm = np.concatenate([-2.*p, np.ones((d_embedding, 1)), np.linalg.norm(p, axis=1)[:,np.newaxis]**2.], axis=1)

    x_emb = x_augm @ p_augm.T

    # we now apply the distance based kernel function to the embedding.
    # in this case, we simply use a step function that is 1 if the distance is less than r_fact_kernel multiplied by
    # the average smallest distance between points, and 0 otherwise.
    # Intuitively, this means that each point in the input space is represented by a number of "active" random points
    # that are close to it in the input space. In principle, it is possible that a point is not close enough to any points,
    # in which case it will not be represented by any random points. One way to circumvent this would be to always pick the
    # k closest points, but this would require computing all pairwise distances, which is expensive.

    dpp = np.linalg.norm(p[np.newaxis,:,:] - p[:,np.newaxis,:], axis=2)
    dpp[range(d_embedding), range(d_embedding)] = np.inf
    mean_closest = np.min(dpp, axis=1).mean()

    y_emb = (x_emb < (r_fact_kernel * mean_closest)**2.).astype(np.float64)


    return y_emb
