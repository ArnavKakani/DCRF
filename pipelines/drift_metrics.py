"""Drift metrics — single source of truth for the two versions.

Two genuinely different geometries, used by the two ablation scripts:

  l2_distance       — POINT-to-POINT. Distance between the current residual state
                      and the (averaged) healthy anchor vector. This is what a
                      geomloss `sinkhorn_loss(h_t, v_anchor)` call on single vectors
                      computes: with one point on each side there is nothing to
                      transport, so entropic OT collapses to the ground cost
                      ‖h_t − v_anchor‖². We compute that directly (no solver).

  sinkhorn_distance — SET-to-SET. Real entropic optimal transport between two point
                      clouds: mu = the set of healthy-window states, nu_t = a sliding
                      window of recent states. Unlike the point distance this responds
                      to how the *distribution* of recent states has moved, not to a
                      single outlier token.

Both consume L2-normalized states (unit vectors), matching the existing pipeline.
"""

import torch
from geomloss import SamplesLoss

# Real entropic-OT solver (only meaningful for the set-vs-set version).
# blur=0.1, p=2.
_sinkhorn = SamplesLoss(loss="sinkhorn", p=2, blur=0.1, backend="tensorized")


def l2_distance(h_t_norm, v_anchor):
    """Squared L2 between the current normalized state and the anchor mean.

    h_t_norm : (1, D) or (D,) unit vector — current residual state.
    v_anchor : (1, D) or (D,) unit vector — mean of the healthy window.
    Returns a Python float. Monotonic with the geomloss(p=2) value on
    single vectors.
    """
    return torch.sum((h_t_norm - v_anchor) ** 2).item()


def sinkhorn_distance(recent_set, healthy_set):
    """Real set-vs-set entropic optimal transport (Sinkhorn).

    recent_set  : (k, D) cloud of recent normalized states            -> nu_t
    healthy_set : (m, D) cloud of healthy-window normalized states    -> mu
    Returns a Python float. NOTE: this lives on a different numeric scale than the
    point distance, so its Z-threshold must be re-calibrated (see issue #9 sweep).
    """
    return _sinkhorn(recent_set, healthy_set).item()
