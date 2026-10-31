import torch, numpy as np
from plyfile import PlyData
from gsplat import rasterization


# ---------------------------------------------------------------------------
# Loading and covariance
# ---------------------------------------------------------------------------

def load_3dgs_ply(path, dev="cuda"):
    v = PlyData.read(path)["vertex"]
    g = lambda *n: torch.tensor(np.stack([v[x] for x in n], -1), dtype=torch.float32, device=dev)
    means = g("x", "y", "z")
    scales = torch.exp(g("scale_0", "scale_1", "scale_2"))  # stored as log-scale
    quats = torch.nn.functional.normalize(g("rot_0", "rot_1", "rot_2", "rot_3"), dim=-1)  # wxyz
    opac = torch.sigmoid(g("opacity")[:, 0])  # stored as logit
    return means, quats, scales, opac


def quat_to_R(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], -1).view(-1, 3, 3)


def covariance(quats, scales):
    M = quat_to_R(quats) @ torch.diag_embed(scales)  # R S
    return M @ M.transpose(1, 2)  # Σ = R S Sᵀ Rᵀ


# ---------------------------------------------------------------------------
# Ray-based path (debugging / future ray-traced renderer)
# [M, N] broadcasting is for prototyping; cull candidates first in production.
# ---------------------------------------------------------------------------

def pixel_rays(K, c2w, uv):  # c2w: OpenCV convention, uv: [M,2] pixel indices
    pix = torch.cat([uv + 0.5, torch.ones_like(uv[:, :1])], -1)
    d = torch.nn.functional.normalize((torch.linalg.inv(K) @ pix.T).T @ c2w[:3, :3].T, dim=-1)
    return c2w[:3, 3].expand_as(d), d


def ray_gaussian_terms(o, d, mu, Sinv):  # o,d: [M,3] mu: [N,3] Sinv: [N,3,3]
    m = mu[None] - o[:, None]  # [M,N,3]
    a = torch.einsum("mi,nij,mj->mn", d, Sinv, d)  # dᵀΣ⁻¹d
    b = torch.einsum("mi,nij,mnj->mn", d, Sinv, m)  # dᵀΣ⁻¹(μ−o)
    c = torch.einsum("mni,nij,mnj->mn", m, Sinv, m)  # (μ−o)ᵀΣ⁻¹(μ−o)
    t_star = b / a  # depth of peak response
    dM2_min = c - b * b / a  # min Mahalanobis² along the ray
    peak = torch.exp(-0.5 * dM2_min)  # 3DGRT-style response
    integ = torch.sqrt(2 * torch.pi / a) * peak  # exact line integral of the density
    return t_star, dM2_min, peak, integ


def ray_ellipsoid_hit(dM2_min, t_star, k=3.0):  # identical to thresholded Mahalanobis
    return (dM2_min <= k * k) & (t_star > 0)


def ray_compositing_weights(peak, t_star, opac):  # per-ray transmittance, sorted by depth
    alpha = (opac[None] * peak).clamp(max=0.99)
    order = t_star.argsort(1)
    a_s = alpha.gather(1, order)
    T = torch.cumprod(torch.cat([torch.ones_like(a_s[:, :1]), 1 - a_s[:, :-1]], 1), 1)
    return (T * a_s).scatter(1, order, T * a_s)  # w_i = T_i α_i, back in Gaussian order


# ---------------------------------------------------------------------------
# Rasterizer path (the production association step)
# ---------------------------------------------------------------------------

def detection_prob_image(dets, H, W, beta, f, edge_frac=0.08, dev="cuda"):
    """dets: [(x0,y0,x1,y1,q_calibrated)] at render resolution → π_v(p) ∈ [f, 1]."""
    P = torch.full((H, W), f, device=dev)
    yy, xx = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
    for x0, y0, x1, y1, q in dets:
        ex, ey = edge_frac * (x1 - x0), edge_frac * (y1 - y0)
        kx = ((torch.minimum(xx - x0, x1 - xx) / ex).clamp(0, 1) * torch.pi / 2).sin()
        ky = ((torch.minimum(yy - y0, y1 - yy) / ey).clamp(0, 1) * torch.pi / 2).sin()
        P = torch.maximum(P, f + (q * beta - f) * kx * ky)
    return P


def per_view_evidence(means, quats, scales, opac, viewmat, K, H, W, prob_imgs, settings):
    """prob_imgs: [H,W,Kc] one π_v,k per class.
    Returns num [N,Kc] = Σ_p w_i(p) π_v,k(p) and vis [N] = Σ_p w_i(p).
    Exact because rendering is linear in the per-Gaussian feature once geometry is frozen."""
    N, Kc = means.shape[0], prob_imgs.shape[-1]
    feat = torch.ones(N, Kc + 1, device=means.device, requires_grad=True)
    img, _, _ = rasterization(means, quats, scales, opac, feat,
                              viewmat[None], K[None], W, H, sh_degree=None,
                              **settings)  # MUST match training settings
    target = torch.cat([prob_imgs, torch.ones_like(prob_imgs[..., :1])], -1)
    (img[0] * target).sum().backward()
    num, vis = feat.grad[:, :Kc], feat.grad[:, Kc]
    return num.detach(), vis.detach()


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def logit(p):
    p = p.clamp(1e-4, 1 - 1e-4)
    return torch.log(p / (1 - p))


def view_reliability(means, quats, scales, cam_center):
    R = quat_to_R(quats)
    normal = R[torch.arange(len(R)), :, scales.argmin(-1)]  # thinnest axis ≈ surface normal
    view = torch.nn.functional.normalize(cam_center - means, dim=-1)
    return (normal * view).sum(-1).abs().clamp(0.2, 1.0)  # grazing views down-weighted


class DefectField:
    def __init__(self, N, prior, dev):  # prior: [Kc] e.g. [0.03, 0.01]
        self.L = torch.zeros(N, len(prior), device=dev)
        self.n_obs = torch.zeros(N, device=dev)
        self.l0 = logit(torch.tensor(prior, device=dev))

    def update(self, num, vis, omega, v_min=0.3, L_max=2.5):
        obs = vis > v_min  # unobserved Gaussians: no update
        p = num[obs] / vis[obs, None]  # inverse sensor model, per class
        dL = (logit(p) - self.l0).clamp(-L_max, L_max)  # relative to prior; negative if unseen in boxes
        self.L[obs] += omega[obs, None] * dL
        self.n_obs[obs] += omega[obs]

    def posterior(self):
        return torch.sigmoid(self.l0 + self.L)


# ---------------------------------------------------------------------------
# Surface-graph smoothing: solves (I + λ(D − A)) L̂ = L by Jacobi iteration
# ---------------------------------------------------------------------------

def smooth_on_surface(L, means, Sigma, knn_idx, lam=0.3, iters=30):
    diff = means[knn_idx] - means[:, None]  # [N,k,3]
    S = Sigma[:, None] + Sigma[knn_idx]
    A = torch.exp(-0.5 * torch.einsum("nki,nkij,nkj->nk", diff, torch.linalg.inv(S), diff))
    Lh = L.clone()
    for _ in range(iters):
        Lh = (L + lam * (A[..., None] * Lh[knn_idx]).sum(1)) / (1 + lam * A.sum(1, keepdim=True))
    return Lh


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
# The following must be supplied by you (they are not defined in the source):
#   keyframes : iterable of views with .dets (dict class -> boxes), .w2c, .K,
#               .cam_center, .cluster_size
#   classes   : list of class keys, length Kc (same order as `prior`)
#   beta      : dict class -> detector confidence scale
#   f         : floor detection probability
#   H, W      : render resolution
#   settings  : dict of extra kwargs for gsplat rasterization (must match training)
#   knn_idx   : [N,k] long tensor of nearest-neighbour indices per Gaussian

if __name__ == "__main__":
    means, quats, scales, opac = load_3dgs_ply("scene.ply")
    field = DefectField(len(means), prior=[0.03, 0.01], dev="cuda")
    for v in keyframes:  # pose-decorrelated, sharp frames
        pi = torch.stack([detection_prob_image(v.dets[k], H, W, beta[k], f) for k in classes], -1)
        num, vis = per_view_evidence(means, quats, scales, opac, v.w2c, v.K, H, W, pi, settings)
        omega = view_reliability(means, quats, scales, v.cam_center) / v.cluster_size
        field.update(num, vis, omega)
    L_hat = smooth_on_surface(field.l0 + field.L, means, covariance(quats, scales), knn_idx)
    P = torch.sigmoid(L_hat)  # [N, Kc] defect probability field