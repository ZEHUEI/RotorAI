"""CPU-only mock of 270-degree ray-based Gaussian rendering.
Install: pip install numpy scipy matplotlib
Run: python mahalanobis_mock.py
"""
from pathlib import Path
import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

WIDTH, HEIGHT = 480, 240
SIGMA_CUTOFF = 3.0
OUT = Path(__file__).with_name('mahalanobis_mock.png')


def rot_y(degrees):
    a = np.deg2rad(degrees)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# Center, standard deviations along local axes, rotation, color, opacity strength.
GAUSSIANS = [
    (np.array([0., 0., -4.]), np.array([1.0, .75, .45]), rot_y(35), np.array([1.0, .45, .2]), 1.7),
    (np.array([3., .0, -1.4]), np.array([.60, 1.0, .50]), rot_y(-25), np.array([.20, .75, 1.0]), 1.4),
    (np.array([-3., -.1, -1.5]), np.array([.85, .55, .35]), rot_y(65), np.array([.8, .35, 1.0]), 1.5),
]

# Equidistant angular projection: 270 deg horizontal FOV, 80 deg vertical FOV.
# Crucially, ray tracing does NOT require pinhole/perspective-only projection.
az = np.deg2rad(np.linspace(-135, 135, WIDTH))
el = np.deg2rad(np.linspace(40, -40, HEIGHT))
AZ, EL = np.meshgrid(az, el)
rays = np.stack([np.sin(AZ) * np.cos(EL), np.sin(EL),
                 -np.cos(AZ) * np.cos(EL)], axis=-1)

records = []
for center, scales, rotation, color, strength in GAUSSIANS:
    # World -> Gaussian local normalized coordinates: y = R.T @ (x - mu) / scales
    # For row-vector arrays: y = (x - mu) @ R / scales.
    origin_local = (-center @ rotation) / scales
    direction_local = (rays @ rotation) / scales  # DO NOT normalize: t remains world-ray distance.
    A = np.sum(direction_local**2, axis=-1)
    B = np.sum(direction_local * origin_local, axis=-1)
    C = np.sum(origin_local**2) - SIGMA_CUTOFF**2
    delta = B**2 - A * C
    intersects = delta >= 0
    root = np.sqrt(np.maximum(delta, 0.0))
    near = (-B - root) / A
    far = (-B + root) / A
    valid = intersects & (far > 0)
    start = np.maximum(near, 0)
    end = np.maximum(far, start)

    # Gaussian density on the ray: exp(-0.5*(A*t*t + 2*B*t + ||origin||^2))
    # Integrate analytically over its clipped [start,end] interval.
    closest_t = -B / A
    min_d2 = np.sum(origin_local**2) - B**2 / A
    z0 = np.sqrt(A / 2) * (start - closest_t)
    z1 = np.sqrt(A / 2) * (end - closest_t)
    integral = np.exp(-0.5 * np.maximum(min_d2, 0)) * np.sqrt(np.pi / (2 * A)) * (erf(z1) - erf(z0))
    soft_alpha = np.where(valid, 1 - np.exp(-strength * np.maximum(integral, 0)), 0)
    hard_alpha = np.where(valid, np.minimum(strength / 1.8, 1), 0)
    records.append((np.where(valid, start, np.inf), soft_alpha, hard_alpha, color))

# Approximate depth composition: sort primitive entry distances per pixel.
# A full volumetric renderer would integrate interleaved density contributions in depth order.
depths = np.stack([r[0] for r in records], -1)
order = np.argsort(depths, axis=-1)
background = np.array([.07, .08, .12])

def compose(which):
    rgb = np.zeros((HEIGHT, WIDTH, 3))
    trans = np.ones((HEIGHT, WIDTH))
    alphas = np.stack([r[1 if which == 'soft' else 2] for r in records], -1)
    colors = np.stack([r[3] for r in records])
    for j in range(len(records)):
        idx = order[..., j]
        a = np.take_along_axis(alphas, idx[..., None], axis=-1)[..., 0]
        rgb += (trans * a)[..., None] * colors[idx]
        trans *= (1 - a)
    return np.clip(rgb + trans[..., None] * background, 0, 1)

fig, axs = plt.subplots(2, 1, figsize=(13, 7), constrained_layout=True)
for ax, mode, title in zip(axs, ('hard', 'soft'),
                           ('Hard ray–ellipsoid cutoff (binary-looking edges)',
                            'Mahalanobis-bounded integrated Gaussian (soft alpha)')):
    ax.imshow(compose(mode), extent=[-135, 135, -40, 40], aspect='auto')
    ax.set_title(title)
    ax.set_ylabel('Elevation (degrees)')
    ax.set_xlabel('Azimuth (degrees) — horizontal 270° field of view')
fig.savefig(OUT, dpi=150)
print(f'Saved {OUT}')