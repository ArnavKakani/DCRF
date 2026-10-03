"""Drift metrics for the automated pipeline — one registry, five geometries.

Every metric exposes the same small interface so the detection engine in
runtime.py can drive any of them without special-casing:

    build_reference(healthy_norm, healthy_raw) -> ref
        Called once, after the HEALTHY_WINDOW tokens, to summarise the healthy
        regime. `healthy_norm` / `healthy_raw` are lists of (1, D) tensors
        (L2-normalised and raw residual states respectively).

    score(h_norm, h_raw, recent_norm, ref) -> float
        Called every post-healthy token. `recent_norm` is the list of the most
        recent RECENT_WINDOW normalised states (only meaningful when uses_set).

Flags:
    uses_set    metric needs the sliding window as a *cloud* (only scores once the
                window is full) — true for Sinkhorn, false for the point metrics.
    is_control  metric has no detection geometry; it is the random-position
                baseline and is handled directly by Stage 2 (skips layer/threshold
                selection). Kept in the registry so the chain stays uniform.

The four real metrics differ only in geometry; the random control differs in
*mechanism*. Together they answer "is the latent signal real, and does this
particular geometry matter?" (TODO #7 random baseline, #8 metric ablation).
"""

import torch
from geomloss import SamplesLoss

# Real entropic-OT solver; only the set-vs-set metric actually transports mass.
_sinkhorn_solver = SamplesLoss(loss="sinkhorn", p=2, blur=0.1, backend="tensorized")


class Metric:
    name = "base"
    uses_set = False
    is_control = False

    def build_reference(self, healthy_norm, healthy_raw):
        raise NotImplementedError

    def score(self, h_norm, h_raw, recent_norm, ref):
        raise NotImplementedError


class L2Metric(Metric):
    """Point-to-point squared L2 between the current state and the healthy mean.

    This is exactly what the original geomloss(h_t, v_anchor) call degenerated to
    (one point per side -> nothing to transport -> ground cost ||h - v||^2).
    """
    name = "l2"

    def build_reference(self, healthy_norm, healthy_raw):
        v = torch.stack(healthy_norm).mean(dim=0)
        return v / (v.norm(p=2, dim=-1, keepdim=True) + 1e-8)

    def score(self, h_norm, h_raw, recent_norm, ref):
        return torch.sum((h_norm - ref) ** 2).item()


class CosineMetric(Metric):
    """Cosine distance (1 - cos) between the current state and the healthy mean.

    Same anchor as L2 but scale-invariant: responds only to *direction* change.
    """
    name = "cosine"

    def build_reference(self, healthy_norm, healthy_raw):
        v = torch.stack(healthy_norm).mean(dim=0)
        return v / (v.norm(p=2, dim=-1, keepdim=True) + 1e-8)

    def score(self, h_norm, h_raw, recent_norm, ref):
        # both operands are unit vectors, so the dot product is the cosine
        return (1.0 - torch.sum(h_norm * ref)).item()


class ResidualNormMetric(Metric):
    """Deviation of the raw residual-stream magnitude from its healthy mean.

    A direction-free control: ignores *where* the state points and looks only at
    how large the residual got. Uses the UN-normalised states (the norm is the
    signal) — the only metric that consumes h_raw.
    """
    name = "residual_norm"

    def build_reference(self, healthy_norm, healthy_raw):
        norms = torch.stack([h.norm(p=2) for h in healthy_raw])
        return norms.mean().item()

    def score(self, h_norm, h_raw, recent_norm, ref):
        return abs(h_raw.norm(p=2).item() - ref)


class SinkhornMetric(Metric):
    """Real set-vs-set entropic optimal transport.

    mu   = cloud of healthy-window normalised states (kept, not averaged)
    nu_t = sliding window of the RECENT_WINDOW most recent normalised states
    D_t  = Sinkhorn(nu_t, mu)

    Responds to how the *distribution* of recent states has moved, not to a single
    outlier token — the only setting where OT is a meaningful choice over L2.
    Lives on a different numeric scale than the point metrics, which is fine
    because triggering is on the standardised Z-score, calibrated in Stage 1.
    """
    name = "sinkhorn"
    uses_set = True

    def build_reference(self, healthy_norm, healthy_raw):
        return torch.cat(healthy_norm, dim=0)          # (w, D) cloud

    def score(self, h_norm, h_raw, recent_norm, ref):
        recent_set = torch.cat(recent_norm, dim=0)     # (k, D) cloud
        return _sinkhorn_solver(recent_set, ref).item()


class RandomControlMetric(Metric):
    """Matched-position control: no geometry, trigger at a random token.

    Detection is bypassed (Stage 2 draws a uniform trigger index per problem and
    runs the same rollback+reprompt). Establishes that the latent signal beats
    intervening at a random position — the highest-leverage baseline (TODO #7).
    """
    name = "random"
    is_control = True


METRICS = {
    m.name: m
    for m in (
        L2Metric(),
        CosineMetric(),
        ResidualNormMetric(),
        SinkhornMetric(),
        RandomControlMetric(),
    )
}
