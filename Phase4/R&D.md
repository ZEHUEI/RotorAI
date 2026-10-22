# TODO: Mahalanobis-Masked Ray–Ellipsoid Rendering for 3D Gaussian Splatting

## Goal
Implement a custom Three.js/WebGL ray-based rendering path for 3D Gaussian Splatting (3DGS) to investigate mask-induced artifacts and improve alignment under a ~270° field of view.

> **Technical note:** A hard binary mask does not automatically destroy all 3DGS gradients; whether it causes optimization issues depends on where and how it is applied. Likewise, clipping at 3σ is an approximation, not the exact physical surface of an object. Validate these hypotheses against the current renderer before treating them as root causes.

## 1. Investigate the current problem
- [ ] Reproduce the black-and-white mask failure with a fixed scene, camera, and seed.
- [ ] Identify where the binary mask enters the pipeline (training loss, alpha compositing, shader, or preprocessing).
- [ ] Record symptoms: disappearing Gaussians, enlarged scales, halos, and 270° misalignment.
- [ ] Capture baseline screenshots and metrics for later comparison.
- [ ] Check for unrelated causes, including camera calibration, coordinate conventions, and spherical/fisheye projection.

## 2. Parse Gaussian attributes from `.ply`
- [ ] Load binary `.ply` files into the Three.js pipeline.
- [ ] Read positions (`x`, `y`, `z`) for each Gaussian center, **μ**.
- [ ] Read scale attributes and apply the format's scale decoding (e.g., exponentiation if log-scales are stored).
- [ ] Read quaternion rotations and normalize them.
- [ ] Read opacity and apply the format's decoding (e.g., sigmoid if logits are stored).
- [ ] Verify axis ordering, matrix conventions, and quaternion component order.
- [ ] Upload the required per-Gaussian attributes to GPU buffers/textures.

## 3. Implement local-space ray transformations
- [ ] Convert each Gaussian's quaternion to a rotation matrix **R**.
- [ ] Transform the ray origin and direction using the inverse Gaussian rotation and scale.
- [ ] Keep the transformed direction **unnormalized**, or correctly rescale ray parameters after normalization.
- [ ] Handle extremely small Gaussian scales with numerical safeguards.
- [ ] Confirm that a transformed Gaussian has identity covariance in local coordinates, under the chosen scale convention.

## 4. Implement the Mahalanobis-distance test
- [ ] Implement the squared Mahalanobis distance:

  `D_M²(x) = (x − μ)^T Σ⁻¹ (x − μ)`

- [ ] Use local coordinates to compute the same quantity without explicitly inverting the covariance matrix.
- [ ] Find the closest point on the **forward ray** (clamp closest-approach `t` to `t >= 0`).
- [ ] Reject Gaussians whose closest forward-ray distance exceeds a configurable threshold, initially `D_M > 3`.
- [ ] Test threshold values (e.g., 2σ, 3σ, and 4σ) for quality/performance trade-offs.
- [ ] Remember: `3σ = 99.7%` applies to a one-dimensional normal distribution interval, **not** the enclosed probability of a 3D ellipsoid.

## 5. Implement ray–ellipsoid intersection and rendering
- [ ] Solve the ray/ellipsoid quadratic for intersections with `D_M² = threshold²`.
- [ ] Handle rays starting inside the ellipsoid, tangent rays, and intersections behind the camera.
- [ ] Use intersections as **support bounds** for the Gaussian; do not assume the threshold surface is a true physical object boundary.
- [ ] Decide whether to evaluate volume density/integrated opacity along the intersecting ray segment or use another documented splatting approximation.
- [ ] Preserve continuous alpha/transmittance compositing where appropriate.
- [ ] Implement depth ordering/occlusion consistently for overlapping Gaussians.
- [ ] Avoid using fragment `discard` as a substitute for a complete intersection-and-compositing algorithm.

## 6. Address the 270° camera alignment
- [ ] Verify the camera's projection model (perspective, fisheye, equirectangular, or custom spherical mapping).
- [ ] Generate a correct world-space ray for every output pixel under that projection.
- [ ] Check seams, extreme angles, ray directions, and coordinate transforms.
- [ ] Compare the original rasterization output against ray-based output at the same camera pose.
- [ ] Determine whether remaining errors arise from projection, Gaussian representation, or alpha/depth handling.
- [ ] Evaluate non-linear projection methods such as 3D Gaussian Unscented Transform if applicable.

## 7. Tests and validation
- [ ] Unit-test quaternion-to-matrix conversion and covariance/local-space equivalence.
- [ ] Test spheres, highly stretched ellipsoids, and rotated ellipsoids with known intersections.
- [ ] Test numerical stability for tiny/large scales and near-tangent rays.
- [ ] Compare image quality, alignment, alpha halos, and frame rate before and after changes.
- [ ] Profile GPU cost and investigate acceleration/culling if per-ray-per-Gaussian testing is too expensive.
- [ ] If modifying training, confirm that gradients remain meaningful through all differentiable operations.

## 8. Deliverables
- [ ] Reusable `.ply` parser for Gaussian attributes.
- [ ] GLSL quaternion-to-inverse-rotation helper.
- [ ] GLSL ray-to-local-Gaussian transform helper.
- [ ] GLSL Mahalanobis-bound / ray–ellipsoid intersection helper.
- [ ] Three.js example integrating the helpers with Gaussian data.
- [ ] Before/after renders and performance benchmarks for the ~270° case.
- [ ] Short write-up explaining findings, limitations, and chosen threshold.

## Reference reading
- [ ] Kerbl et al. (2023), *3D Gaussian Splatting for Real-Time Radiance Field Rendering* — Gaussian covariance representation.
- [ ] *RaySplats: Ray Tracing based Gaussian Splatting* (arXiv:2501.19196) — ray-based Gaussian rendering approaches.
- [ ] Research on *3D Gaussian Ray Tracing* — validate the precise paper/authors before citation.
- [ ] Research on *3D Gaussian Unscented Transform* — non-linear and wide-angle projection approaches.

## Suggested implementation order
1. Reproduce and diagnose the existing mask/alignment issue.
2. Parse and validate `.ply` attributes.
3. Implement/test local-space rays and Mahalanobis bounds.
4. Add correct ray–ellipsoid support intersections and alpha compositing.
5. Validate spherical/fisheye camera rays for ~270° output.
6. Benchmark, refine, and document results.
