import taichi as ti
import numpy as np
import math

ti.init(arch=ti.cuda)

# ==========================================
# GLOBAL PHYSICAL PARAMETERS
# ==========================================
E_val = 5000.0
nu = 0.2
n_grid = 128
dx = 1.0 / n_grid
inv_dx = float(n_grid)
dt = 2.5e-5
p_vol = (dx * 0.5) ** 3
p_rho = 1.0
p_mass = p_vol * p_rho

MAX_PHYS_POINTS = 100000
MAX_RENDER_POINTS = 3000000
CRUSHER_POINTS = 225

# ==========================================
# AVAILABLE MATERIALS
# ==========================================
MATERIAL_ELASTIC = 0
MATERIAL_METAL = 1
MATERIAL_FOAM = 2
MATERIAL_SAND = 3
MATERIAL_NAMES = [
    "Elastic (rubber/hard gel)",
    "Metal (ductile)",
    "Foam/Gel (viscoplastic)",
    "Granular (sand)",
]

# ==========================================
# TAICHI FIELDS (STATE)
# ==========================================
active_N = ti.field(int, shape=())
active_N_total = ti.field(int, shape=())

active = ti.field(dtype=int, shape=MAX_PHYS_POINTS)

x = ti.Vector.field(3, dtype=float, shape=MAX_PHYS_POINTS)
v = ti.Vector.field(3, dtype=float, shape=MAX_PHYS_POINTS)
C = ti.Matrix.field(3, 3, dtype=float, shape=MAX_PHYS_POINTS)
F = ti.Matrix.field(3, 3, dtype=float, shape=MAX_PHYS_POINTS)
Jp = ti.field(dtype=float, shape=MAX_PHYS_POINTS)
ep = ti.field(dtype=float, shape=MAX_PHYS_POINTS)
grid_v = ti.Vector.field(3, dtype=float, shape=(n_grid, n_grid, n_grid))
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid, n_grid)) 

enable_crusher_ti = ti.field(int, shape=())
crusher_y_ti = ti.field(float, shape=())

# Floor collision plane setup (used by GUI but defined here for consistency)
floor_pts = []
res_floor = 15
for i in range(res_floor):
    for k in range(res_floor):
        fx = i / (res_floor - 1.0) * 0.8 + 0.1
        fk = k / (res_floor - 1.0) * 0.8 + 0.1
        floor_pts.append([fx, 0.9, fk])
floor_np = np.array(floor_pts, dtype=np.float32)
floor_field = ti.Vector.field(3, dtype=float, shape=len(floor_np))
floor_field.from_numpy(floor_np)

# Initializes MPM particle data (positions, deformation gradient, velocities)
@ti.kernel
def init_data(pos_arr: ti.types.ndarray()):
    for i in range(MAX_PHYS_POINTS):
        if i >= active_N_total[None]:
            continue
        for j in ti.static(range(3)):
            x[i][j] = pos_arr[i, j]
        F[i] = ti.Matrix.identity(float, 3)
        Jp[i] = 1.0
        ep[i] = 0.0
        v[i] = ti.Vector.zero(float, 3)
        C[i] = ti.Matrix.zero(float, 3, 3)
        active[i] = 1

# Main MPM Substep: Particle-to-Grid (P2G), Grid Operations, Grid-to-Particle (G2P)
@ti.kernel
def mpm_substep(
    mu_0: float,               # Initial shear modulus (Lamé 1st param), shear resistance
    lam_0: float,              # Initial bulk modulus (Lamé 2nd param), volume resistance
    grav: float,               # Gravity acceleration applied along +Y (downwards)
    material: int,             # Selected material model (Elastic, Metal, Foam, Sand)
    hardening: float,          # Metal hardening coeff; increases yield strength
    friction_deg: float,       # Sand friction angle (deg) for Drucker-Prager cone
    yield_stress: float,       # Plastic yield threshold (von Mises radius / foam limit)
    floor_friction: float,     # Boundary friction (0-1) slowing tangential movement
    crusher_mass_val: float,   # High mass for crusher particles to displace material
    fracture_threshold: float, # Stretch threshold for disabling particles
):
    # 1. Reset Grid
    for i, j, k in grid_m:
        grid_v[i, j, k] = [0, 0, 0]
        grid_m[i, j, k] = 0

    # Drucker-Prager friction cone coefficient (only for sand material)
    friction_rad = friction_deg * math.pi / 180.0
    sand_alpha = (
        ti.sqrt(2.0 / 3.0) * (2.0 * ti.sin(friction_rad)) / (3.0 - ti.sin(friction_rad))
    )

    # 2. Particle to Grid (P2G)
    for p in x:
        # Quadratic B-spline weights w_ip for interpolation from particle p
        # to its 27 nearby grid nodes i.
        # Normalized pos: x_p / dx. Base: bottom-left-back node in 3x3x3.
        if p >= active_N_total[None]:
            continue
        if active[p] == 0:
            continue
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

        # Update deformation gradient F with velocity field C_p (grad v_p):
        # $F_{n+1} = (I + \Delta t C_p) F_n$
        F[p] = (ti.Matrix.identity(float, 3) + dt * C[p]) @ F[p]

        stress = ti.Matrix.zero(float, 3, 3)

        if p >= active_N[None]:
            # Crusher uses stiff elasticity. Kirchhoff stress formula:
            # $\tau = \mu (F F^T - I) + \lambda J (J - 1) I$, where J = det(F)
            J = F[p].determinant()
            mu_crusher = 10000.0
            lam_crusher = 10000.0
            stress = mu_crusher * (
                F[p] @ F[p].transpose() - ti.Matrix.identity(float, 3)
            ) + ti.Matrix.identity(float, 3) * lam_crusher * J * (J - 1)
        elif material == MATERIAL_ELASTIC:
            # Elastic material Kirchhoff stress:
            # $\tau = \mu_0 (F F^T - I) + \lambda_0 J (J - 1) I$, where J = det(F)
            J = F[p].determinant()
            
            stress = mu_0 * (
                F[p] @ F[p].transpose() - ti.Matrix.identity(float, 3)
            ) + ti.Matrix.identity(float, 3) * lam_0 * J * (J - 1)
        else:
            # Plasticity models: Corotated elasticity with return mapping on
            # principal stretches via SVD: $F = U \Sigma V^T$
            h = 1.0
            if material == MATERIAL_METAL:
                h = ti.exp(hardening * (1.0 - Jp[p]))
            mu, lam = mu_0 * h, lam_0 * h

            if material == MATERIAL_FOAM:
                mu = mu_0 * 0.3  # Reduced shear modulus -> softer/more fluid

            U, sig, V = ti.svd(F[p])

            max_stretch = ti.max(sig[0, 0], ti.max(sig[1, 1], sig[2, 2]))
            if max_stretch > fracture_threshold:
                active[p] = 0
                continue

            if material == MATERIAL_SAND:
                # Drucker-Prager plasticity. Principal stretches sigma_i map to
                # Hencky (logarithmic) strains: $\epsilon_i = \ln(\sigma_i)$
                e0 = ti.log(ti.max(ti.abs(sig[0, 0]), 1e-6))
                e1 = ti.log(ti.max(ti.abs(sig[1, 1]), 1e-6))
                e2 = ti.log(ti.max(ti.abs(sig[2, 2]), 1e-6))
                eps = ti.Vector([e0, e1, e2])
                trace_eps = eps[0] + eps[1] + eps[2]
                eps_hat = eps - trace_eps / 3.0
                eps_hat_norm = eps_hat.norm() + 1e-20
                delta_gamma = (
                    eps_hat_norm
                    + (3.0 * lam + 2.0 * mu) / (2.0 * mu) * trace_eps * sand_alpha
                )

                new_eps = eps
                if trace_eps >= 0.0:
                    new_eps = ti.Vector(
                        [0.0, 0.0, 0.0]
                    )  # cannot resist tension -> collapse
                elif delta_gamma > 0.0:
                    new_eps = (
                        eps - (delta_gamma / eps_hat_norm) * eps_hat
                    )  # project onto the cone

                ns0, ns1, ns2 = ti.exp(new_eps[0]), ti.exp(new_eps[1]), ti.exp(new_eps[2])
                Jp[p] *= (sig[0, 0] * sig[1, 1] * sig[2, 2]) / (ns0 * ns1 * ns2)
                sig[0, 0], sig[1, 1], sig[2, 2] = ns0, ns1, ns2

            elif material == MATERIAL_METAL:
                # Standard von Mises (J2) Plasticity. Principal stretches sigma_i
                # map to Hencky strains: $\epsilon_i = \ln(\sigma_i)$
                e0 = ti.log(ti.max(ti.abs(sig[0, 0]), 1e-6))
                e1 = ti.log(ti.max(ti.abs(sig[1, 1]), 1e-6))
                e2 = ti.log(ti.max(ti.abs(sig[2, 2]), 1e-6))
                eps = ti.Vector([e0, e1, e2])
                
                trace_eps = eps[0] + eps[1] + eps[2]
                eps_hat = eps - trace_eps / 3.0
                eps_hat_norm = eps_hat.norm() + 1e-20
                
                # PhysGaussian maps yield_stress to the radius of the elastic region
                max_strain = yield_stress * 0.05
                
                if eps_hat_norm > max_strain:
                    # Return mapping to the yield cylinder
                    eps_hat = eps_hat * (max_strain / eps_hat_norm)
                
                new_eps = eps_hat + trace_eps / 3.0
                ns0, ns1, ns2 = ti.exp(new_eps[0]), ti.exp(new_eps[1]), ti.exp(new_eps[2])
                
                Jp[p] *= (sig[0, 0] * sig[1, 1] * sig[2, 2]) / (ns0 * ns1 * ns2)
                
                # Explicitly clamp principal ELASTIC stretches to prevent Gaussian artifacts (spikes)
                # as described in the PhysGaussian paper.
                sig[0, 0] = ti.min(ti.max(ns0, 0.4), 2.5)
                sig[1, 1] = ti.min(ti.max(ns1, 0.4), 2.5)
                sig[2, 2] = ti.min(ti.max(ns2, 0.4), 2.5)

            elif material == MATERIAL_FOAM:
                # Viscoplastic: elastic under threshold, relaxes towards volume over threshold
                dev_norm = ti.sqrt(
                    (sig[0, 0] - 1) ** 2 + (sig[1, 1] - 1) ** 2 + (sig[2, 2] - 1) ** 2
                )
                if dev_norm > yield_stress:
                    relax = yield_stress / dev_norm
                    for d in ti.static(range(3)):
                        sig[d, d] = 1.0 + (sig[d, d] - 1.0) * relax

            J = sig[0, 0] * sig[1, 1] * sig[2, 2]
            F[p] = U @ sig @ V.transpose()
            stress = 2.0 * mu * (F[p] - U @ V.transpose()) @ F[
                p
            ].transpose() + ti.Matrix.identity(float, 3) * lam * J * (J - 1)

        mass = p_mass
        if p >= active_N[None]:
            mass = crusher_mass_val
            
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        # Affine momentum matrix for APIC transfer:
        # $\text{affine} = \tau + m_p C_p$
        affine = stress + mass * C[p]

        # Splat mass and momentum to 27 neighboring grid nodes using w_ip.
        # Grid momentum update:
        # $(mv)_i = \sum_p w_{ip} [m_p v_p + m_p C_p (x_i - x_p) + \tau (x_i - x_p)]$
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1] * w[k][2]
            grid_v[base + offset] += weight * (mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * mass

    # 3. Grid Operations (Gravity and Boundaries)
    tangential_keep = 1.0 - floor_friction
    for i, j, k in grid_m:
        if grid_m[i, j, k] > 0:
            grid_v[i, j, k] = (1 / grid_m[i, j, k]) * grid_v[i, j, k]
            # Y-down, gravity towards +Y
            grid_v[i, j, k][1] += dt * grav

            # Floor at y=0.9 (n_grid * 0.9)
            if j > 0.9 * inv_dx and grid_v[i, j, k][1] > 0:
                grid_v[i, j, k][1] = 0
                grid_v[i, j, k][0] *= tangential_keep
                grid_v[i, j, k][2] *= tangential_keep
            # Optional ceiling
            if j < 0.1 * inv_dx and grid_v[i, j, k][1] < 0:
                grid_v[i, j, k][1] = 0

            # Walls (padding of 3 grid nodes)
            bound = 3
            if i < bound and grid_v[i, j, k][0] < 0:
                grid_v[i, j, k][0] = 0
            if i > n_grid - bound and grid_v[i, j, k][0] > 0:
                grid_v[i, j, k][0] = 0
            if k < bound and grid_v[i, j, k][2] < 0:
                grid_v[i, j, k][2] = 0
            if k > n_grid - bound and grid_v[i, j, k][2] > 0:
                grid_v[i, j, k][2] = 0

            # Kinematic Crusher
            if enable_crusher_ti[None] == 1:
                cy = crusher_y_ti[None] * inv_dx
                if j < cy and grid_v[i, j, k][1] < 1.0:
                    grid_v[i, j, k][1] = 1.0
                    grid_v[i, j, k][0] *= tangential_keep
                    grid_v[i, j, k][2] *= tangential_keep

    # 4. Grid to Particle (G2P)
    for p in x:
        if p >= active_N_total[None]:
            continue
        if active[p] == 0:
            continue
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 3)
        new_C = ti.Matrix.zero(float, 3, 3)

        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dpos = offset.cast(float) - fx
            weight = w[i][0] * w[j][1] * w[k][2]
            g_v = grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)

        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]

        # Enforce limits on particle positions
        margin = 3.0 * dx
        for j in ti.static(range(3)):
            if x[p][j] < margin:
                x[p][j] = margin
                v[p][j] = 0.0
            elif x[p][j] > 1.0 - margin:
                x[p][j] = 1.0 - margin
                v[p][j] = 0.0
