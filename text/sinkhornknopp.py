import numpy as np


def optimize_l_sk(prob, lmd, a=None, b=None, ddtype=np.float64):
    """
    Sinkhorn-Knopp with optional marginals.

    prob : (n_samples, n_latents) cost/score matrix.
    a : (n_latents,) row marginal target after transpose (latent), sum=1.
    b : (n_samples,) column marginal target (sample), sum=1.
    """
    n_samples = prob.shape[0]
    k = prob.shape[1]

    prob = ddtype(prob)
    prob = prob.T  # (k, n_samples) = (r, n)

    if a is None:
        a_vec = np.full(k, ddtype(1.0 / k), dtype=ddtype)
    else:
        a_vec = np.asarray(a, dtype=ddtype).reshape(-1)
        if a_vec.shape[0] != k:
            raise ValueError(f"a length {a_vec.shape[0]} != n_latents {k}")

    if b is None:
        b_vec = np.full(n_samples, ddtype(1.0 / n_samples), dtype=ddtype)
    else:
        b_vec = np.asarray(b, dtype=ddtype).reshape(-1)
        if b_vec.shape[0] != n_samples:
            raise ValueError(f"b length {b_vec.shape[0]} != n_samples {n_samples}")

    a_col = a_vec.reshape(k, 1)
    b_col = b_vec.reshape(n_samples, 1)

    prob **= lmd  # (k, n)
    err = 1e6
    cnt = 0
    c = np.ones((n_samples, 1), dtype=ddtype) / n_samples

    while err > 1e-1:
        r = a_col / (prob @ c)  # (k, 1)
        c_new = b_col / (r.T @ prob).T  # (n, 1)
        if cnt % 10 == 0:
            err = np.nansum(np.abs(c / np.clip(c_new, 1e-12, None) - 1))
        c = c_new
        cnt += 1

    prob *= np.squeeze(c)
    prob = prob.T
    prob *= np.squeeze(r)  # (n_samples, k)
    argmaxes = np.nanargmax(prob, axis=1)

    return prob, argmaxes
