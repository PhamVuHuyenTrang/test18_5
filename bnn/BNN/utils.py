import torch
import torch.nn.functional as F
import random


def kl_Gauss_Gauss(mean_q, std_q, mean_p, std_p):
    first = torch.log(std_p/std_q)
    second = (std_q ** 2 + (mean_q - mean_p) ** 2) / (2. * std_p ** 2)
    return (first + second - 0.5).sum()


def gumbel_sample(size, eps=1e-20, device='cpu'):
    uniform = torch.rand(size, device=device)
    return -torch.log(-torch.log(uniform + eps) + eps)


def sample_gumbel_mask(size, droprates, temperature=0.01):
    omdrs = 1. - droprates
    probs = torch.stack((droprates, omdrs))
    probs = probs.transpose(0, 1)
    logits = torch.log(probs)
    logits = logits.unsqueeze(0).repeat((size[0], 1, 1))
    y = logits + gumbel_sample(logits.size(), device=droprates.device)
    y = F.softmax(y / temperature, dim=-1)
    return y[:, :, -1]


def sample_gumbel_mask_for_image(shape, droprates, temperature=0.01):
    # shape: (batch_size, channels, height, width)
    omdrs = 1. - droprates
    probs = torch.stack((droprates, omdrs)).transpose(0, 1)
    logits = torch.log(probs)
    y = logits + gumbel_sample(logits.size(), device=droprates.device)
    y = F.softmax(y / temperature, dim=-1)
    y = y.unsqueeze(0).repeat((shape[0], shape[2], shape[3], 1, 1))
    y = torch.transpose(y, 1, 3)
    y = torch.transpose(y, 2, 3)
    y = y[:, :, :, :, -1]
    return y


def droprate_to_mean_mask_image(shape, droprates):
    # shape: (batch_size, channels, height, width)
    omdrs = 1. - droprates
    mask = omdrs.unsqueeze(0).repeat((shape[0], shape[2], shape[3], 1))
    mask = torch.transpose(mask, 1, 3)
    mask = torch.transpose(mask, 2, 3)
    return mask

def droprate_to_mean_mask(shape, droprates):
    omdrs = 1. - droprates
    mask = omdrs.repeat((shape[0], 1))
    return mask


def sample_bernoulli_mask_for_image(shape, droprates):
    # shape: (batch_size, channels, height, width)
    omdrs = 1. - droprates
    mask = omdrs.unsqueeze(0).repeat((shape[0], shape[2], shape[3], 1))
    mask = torch.bernoulli(mask)
    mask = torch.transpose(mask, 1, 3)
    mask = torch.transpose(mask, 2, 3)
    return mask


def decode_diag_gauss(rho, logvar_enc=False, return_var=False,
                      return_logvar=False):
    r"""Decode the standard deviation for a Gaussian distribution with diagonal
    covariance.

    We consider a Gaussian distribution where the covariance is encoded in
    :math:`\rho` (``rho``) as real numbers. We can extract the standard
    deviation from :math:`\rho` as described in the documentation of
    :func:`decode_and_sample_diag_gauss`.

    Args:
        (....): See docstring of function :func:`decode_and_sample_diag_gauss`.
        return_var (bool, optional): If ``True``, the variance :math:`\sigma^2`
            will be returned as well.
        return_logvar (bool, optional): If ``True``, the log-variance
            :math:`\log\sigma^2` will be returned as well.

    Returns:
        (tuple): Tuple containing:

        - **std** (list): The standard deviation :math:`\sigma`.
        - **var** (list, optional): The variance :math:`\sigma^2`. See argument
          ``return_var``.
        - **logvar** (list, optional): The log-variance :math:`\log\sigma^2`.
          See argument ``return_logvar``.
    """
    ret_std = []
    ret_var = []
    ret_logvar = []

    for i in range(len(rho)):
        if logvar_enc:
            std = torch.exp(0.5 * rho[i])
            logvar = rho[i]
        else:
            std = F.softplus(rho[i])
            logvar = 2 * torch.log(std)

        ret_std.append(std)
        ret_logvar.append(logvar)

        if return_var:
            ret_var.append(std**2)

    if return_var and return_logvar:
        return ret_std, ret_var, ret_logvar
    elif return_var:
        return ret_std, ret_var
    elif return_logvar:
        return ret_std, ret_logvar

    return ret_std


def kl_diag_gaussians(mean_a, logvar_a, mean_b, logvar_b):
    r"""Compute the KL divergence between 2 diagonal Gaussian distributions.

    .. math::
        KL \big( p_a(\cdot) \mid\mid  p_b(\cdot) \big)

    Args:
        mean_a: Mean tensors of the first distribution (see argument `mean` of
            method :func:`sample_diag_gauss`).
        logvar_a: Log-variance tensors with the same shapes as the `mean_a`
            tensors.
        mean_b: Same as `mean_a` for second distribution.
        logvar_b: Same as `logvar_a` for second distribution.

    Returns:
        The analytically computed KL divergence between these distributions.
    """
    mean_a_flat = torch.cat([t.view(-1) for t in mean_a])
    logvar_a_flat = torch.cat([t.view(-1) for t in logvar_a])
    mean_b_flat = torch.cat([t.view(-1) for t in mean_b])
    logvar_b_flat = torch.cat([t.view(-1) for t in logvar_b])

    ### Using our own implementation ###
    kl = 0.5 * torch.sum(-1 +
                         (logvar_a_flat.exp() + (mean_b_flat - mean_a_flat).pow(2)) /
                         logvar_b_flat.exp() + logvar_b_flat - logvar_a_flat)

    return kl


def inv_softplus(x):
    positive_mask = (x > 0)
    negative_mask = (x <= 0)
    if negative_mask.sum() > 0:
        Warning("Negative values in input of inv_softplus. Returning -100000.")
    ans = x.clone().detach()
    ans[positive_mask] = torch.log(torch.exp(x[positive_mask]) - 1)
    ans[negative_mask] = -100000.
    return ans


if __name__ == "__main__":
    dr = torch.nn.Parameter(torch.Tensor([-1]))
    droprate = torch.sigmoid(dr)
    optimizer = torch.optim.Adam([dr], lr=0.01)
    shape = (1, 1)
    mcSamples = 10
    for epoch in range(100000):
        droprate = torch.sigmoid(dr)
        optimizer.zero_grad()
        sum = torch.Tensor([[0.]])
        for i in range(mcSamples):
            sample = sample_gumbel_mask(shape, droprate)
            sum += sample
        print(droprate)
        loss = (sum/mcSamples - 0.5)**2
        loss.backward()
        optimizer.step()