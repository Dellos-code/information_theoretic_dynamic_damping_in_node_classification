"""Float64 CPU torch implementation of the EDSP iteration."""

from __future__ import annotations


def edsp_torch(*args, **kwargs):
    """Run EDSP natively with torch tensor operations."""
    import torch  # type: ignore[import-not-found]

    edges, probabilities, heterophily, priors = args[:4] if len(args) >= 4 else (
        kwargs["edges"], kwargs["probabilities"], kwargs["heterophily"], kwargs["priors"]
    )
    prior_trust = kwargs.get("prior_trust", float("inf"))
    edge_trust = kwargs.get("edge_trust", float("inf"))
    epsilon = kwargs.get("epsilon", 1e-6)
    max_iter = kwargs.get("max_iter", 1000)
    v = torch.as_tensor(priors, dtype=torch.float64)
    if v.ndim == 1:
        v = v.unsqueeze(0)
    v = v.clamp_min(0)
    v = v / v.sum(1, keepdim=True).clamp_min(1e-300)
    p = torch.as_tensor(probabilities, dtype=torch.float64)
    prior_trust_tensor = torch.as_tensor(prior_trust, dtype=torch.float64)
    edge_trust_tensor = torch.as_tensor(edge_trust, dtype=torch.float64)
    prior_decay = torch.exp(-prior_trust_tensor)
    edge_decay = torch.exp(-edge_trust_tensor)
    v = (1 - prior_decay) * v + prior_decay / v.shape[1]
    p = p * (1 - edge_decay)
    p = p.clamp(0, 1 - 1e-12)
    h = torch.as_tensor(heterophily, dtype=torch.int64)
    edge = torch.as_tensor(edges, dtype=torch.long).reshape(-1, 2)
    if edge.numel():
        edge = torch.cat((edge, edge.flip(1)))
        p = torch.cat((p, p))
        h = torch.cat((h, h))
        order = torch.argsort(edge[:, 1], stable=True)
        edge, p, h = edge[order], p[order], h[order]
    source, target = edge[:, 0], edge[:, 1]
    n, k = v.shape
    log_odds = -torch.log1p(-p)
    den = torch.zeros(n, dtype=torch.float64).index_add(0, target, log_odds)
    weights = torch.where(den[target] > 0, log_odds / den[target], torch.zeros_like(log_odds))
    p_esc = -torch.expm1(-den)
    base_entropy = -(torch.where(v > 0, v * torch.log(v), torch.zeros_like(v))).sum(1)
    hmax = float(torch.log(torch.tensor(k, dtype=torch.float64)))
    current = v.clone()
    converged = False
    residual = 0.0
    for iteration in range(1, max_iter + 1):
        transformed = torch.where(
            h[:, None] == 0,
            current[source],
            (1 - current[source]) / (k - 1),
        )
        aggregate = torch.zeros_like(current).index_add(0, target, weights[:, None] * transformed)
        aggregate[den == 0] = 1.0 / k
        aggregate_entropy = -(torch.where(
            aggregate > 0, aggregate * torch.log(aggregate), torch.zeros_like(aggregate)
        )).sum(1)
        numerator = (hmax - aggregate_entropy) * p_esc
        denominator = numerator + (hmax - base_entropy)
        alpha = torch.where(denominator > 1e-15, numerator / denominator, torch.zeros_like(denominator))
        boundary = (
            ((torch.abs(hmax - base_entropy) <= 1e-12 * hmax)
             & (torch.abs(hmax - aggregate_entropy) <= 1e-12 * hmax))
            | ((p_esc == 0) & (torch.abs(hmax - base_entropy) <= 1e-12 * hmax))
        )
        next_values = (1 - alpha[:, None]) * v + alpha[:, None] * aggregate
        next_values[boundary] = 1.0 / k
        next_values = next_values.clamp_min(0)
        next_values /= next_values.sum(1, keepdim=True)
        residual = float(torch.max(torch.sum(torch.abs(next_values - current), dim=1)))
        current = next_values
        if residual < epsilon:
            converged = True
            break
    return {
        "probabilities": current,
        "predictions": torch.argmax(current, dim=1),
        "p_esc": p_esc,
        "weights": weights,
        "iterations": iteration,
        "converged": converged,
        "final_residual": residual,
    }
