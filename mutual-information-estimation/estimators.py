import numpy as np
import torch
import torch.nn.functional as F


def logmeanexp_diag(x, device='cuda'):
    """Compute logmeanexp over the diagonal elements of x."""
    batch_size = x.size(0)

    logsumexp = torch.logsumexp(x.diag(), dim=(0,))
    num_elem = batch_size

    return logsumexp - torch.log(torch.tensor(num_elem).float()).to(device)


def logmeanexp_nodiag(x, dim=None, device='cuda'):
    batch_size = x.size(0)
    if dim is None:
        dim = (0, 1)

    logsumexp = torch.logsumexp(
        x - torch.diag(np.inf * torch.ones(batch_size).to(device)), dim=dim)

    try:
        if len(dim) == 1:
            num_elem = batch_size - 1.
        else:
            num_elem = batch_size * (batch_size - 1.)
    except ValueError:
        num_elem = batch_size - 1
    return logsumexp - torch.log(torch.tensor(num_elem)).to(device)


def tuba_lower_bound(scores, log_baseline=None):
    if log_baseline is not None:
        scores -= log_baseline[:, None]

    # First term is an expectation over samples from the joint,
    # which are the diagonal elmements of the scores matrix.
    joint_term = scores.diag().mean()

    # Second term is an expectation over samples from the marginal,
    # which are the off-diagonal elements of the scores matrix.
    marg_term = logmeanexp_nodiag(scores).exp()
    return 1. + joint_term - marg_term


def nwj_lower_bound(scores):
    return tuba_lower_bound(scores - 1.)


def infonce_lower_bound(f, alpha=1.0, **kwargs):
    device = f.device
    n = f.size(0)
    nll = f.diag().mean()
    
    aij = torch.ones(n, n).to(device) * \
            np.log((n - alpha) / (n * (n-1)))
    aii = torch.log(torch.tensor((alpha / n,))).to(device)
    mij = (torch.ones(n, n) - torch.ones(n).diag()).to(device)
    mii = torch.ones(n).diag().to(device)
    weighted_f = f + mii * aii + mij * aij
    
    mi = f.diag() - weighted_f.logsumexp(dim=1)
    mi = mi.mean()
    
    return mi


def js_fgan_lower_bound(f):
    """Lower bound on Jensen-Shannon divergence from Nowozin et al. (2016)."""
    f_diag = f.diag()
    first_term = -F.softplus(-f_diag).mean()
    n = f.size(0)
    second_term = (torch.sum(F.softplus(f)) -
                   torch.sum(F.softplus(f_diag))) / (n * (n - 1.))
    return first_term - second_term


def js_lower_bound(f):
    """Obtain density ratio from JS lower bound then output MI estimate from NWJ bound."""
    nwj = nwj_lower_bound(f)
    js = js_fgan_lower_bound(f)

    with torch.no_grad():
        nwj_js = nwj - js

    return js + nwj_js


def dv_upper_lower_bound(f):
    """
    Donsker-Varadhan lower bound, but upper bounded by using log outside. 
    Similar to MINE, but did not involve the term for moving averages.
    """
    first_term = f.diag().mean()
    second_term = logmeanexp_nodiag(f)

    return first_term - second_term


def mine_lower_bound(f, buffer=None, momentum=0.9):
    """
    MINE lower bound based on DV inequality. 
    """
    if buffer is None:
        buffer = torch.tensor(1.0).cuda()
    first_term = f.diag().mean()

    buffer_update = logmeanexp_nodiag(f).exp()
    with torch.no_grad():
        second_term = logmeanexp_nodiag(f)
        buffer_new = buffer * momentum + buffer_update * (1 - momentum)
        buffer_new = torch.clamp(buffer_new, min=1e-4)
        third_term_no_grad = buffer_update / buffer_new

    third_term_grad = buffer_update / buffer_new

    return first_term - second_term - third_term_grad + third_term_no_grad, buffer_update


def smile_lower_bound(f, clip=None):
    """
    SMILE estimator. Technically not a lower bound due to bias in clipping.
    """
    if clip is not None:
        f_ = torch.clamp(f, -clip, clip)
    else:
        f_ = f
    z = logmeanexp_nodiag(f_, dim=(0, 1))
    dv = f.diag().mean() - z

    js = js_fgan_lower_bound(f)

    with torch.no_grad():
        dv_js = dv - js

    return js + dv_js


def multilabel_infonce_lower_bound(f, alpha=1.0, g=None):
    """
    Multi-label, multi-class versions of InfoNCE.
    """
    device = f.device
    n = f.size(0)
    first_term = f.diag().mean()
    if g is None:
        aij = torch.ones(n, n).to(device) * \
            np.log((n - alpha) / (n * n * (n-1)))
        aii = torch.log(torch.tensor((alpha / (n * n),))).to(device)
        mij = (torch.ones(n, n) - torch.ones(n).diag()).to(device)
        mii = torch.ones(n).diag().to(device)
        weighted_f = f + mii * aii + mij * aij
        second_term = weighted_f.logsumexp(dim=1).logsumexp(dim=0)
    else:
        second_term = torch.tensor((0.0,)).to(device)
        if alpha > 0.0:
            second_term += (f + np.log(alpha)).logsumexp(dim=1).exp()
        if alpha < 1.0:
            second_term += (g + np.log(1 - alpha)).logsumexp(dim=1).exp()
        second_term = second_term.log().logsumexp(
            dim=0) - torch.tensor((n)).float().log() * 2.0

    return first_term - second_term


def estimate_mutual_information(estimator, x, y, critic_fn,
                                baseline_fn=None, alpha_logit=None, **kwargs):
    """Estimate variational lower bounds on mutual information.

  Args:
    estimator: string specifying estimator, one of:
      'nwj', 'infonce', 'tuba', 'js', 'interpolated'
    x: [batch_size, dim_x] Tensor
    y: [batch_size, dim_y] Tensor
    critic_fn: callable that takes x and y as input and outputs critic scores
      output shape is a [batch_size, batch_size] matrix
    baseline_fn (optional): callable that takes y as input 
      outputs a [batch_size]  or [batch_size, 1] vector
    alpha_logit (optional): logit(alpha) for interpolated bound

  Returns:
    scalar estimate of mutual information
    """
    x, y = x.cuda(), y.cuda()
    scores = critic_fn(x, y)

    if baseline_fn is not None:
        # Some baselines' output is (batch_size, 1) which we remove here.
        log_baseline = torch.squeeze(baseline_fn(y))
    if estimator == 'infonce':
        mi = infonce_lower_bound(scores, **kwargs)
    elif estimator == 'nwj':
        mi = nwj_lower_bound(scores)
    elif estimator == 'tuba':
        mi = tuba_lower_bound(scores, log_baseline)
    elif estimator == 'js':
        mi = js_lower_bound(scores)
    elif estimator == 'smile':
        mi = smile_lower_bound(scores, **kwargs)
    elif estimator == 'dv':
        mi = dv_upper_lower_bound(scores)
    elif estimator == 'ml_infonce':
        if kwargs.get('y_', None) is not None:
            y_ = kwargs['y_']
            y_ = y_.cuda()
            scores_ = critic_fn(x, y_)
        else:
            scores_ = None
        mi = multilabel_infonce_lower_bound(scores, g=scores_, alpha=kwargs.get('alpha', 1.0))
    return mi
