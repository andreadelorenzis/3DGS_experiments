import torch

def compute_cov3d_deformed(scale, rot, F):
    """
    Calculates the new 3D covariance matrix (scales & rotations) based on the deformation gradient (F).
    Used in MPM and physics skinning.
    """
    rot = rot / torch.norm(rot, dim=1, keepdim=True)
    r, i, j, k = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]

    R = torch.zeros((scale.shape[0], 3, 3), device=scale.device)
    R[:, 0, 0] = 1.0 - 2.0 * (j * j + k * k)
    R[:, 0, 1] = 2.0 * (i * j - r * k)
    R[:, 0, 2] = 2.0 * (i * k + r * j)
    R[:, 1, 0] = 2.0 * (i * j + r * k)
    R[:, 1, 1] = 1.0 - 2.0 * (i * i + k * k)
    R[:, 1, 2] = 2.0 * (j * k - r * i)
    R[:, 2, 0] = 2.0 * (i * k - r * j)
    R[:, 2, 1] = 2.0 * (j * k + r * i)
    R[:, 2, 2] = 1.0 - 2.0 * (i * i + j * j)

    S = torch.zeros((scale.shape[0], 3, 3), device=scale.device)
    S[:, 0, 0] = scale[:, 0]
    S[:, 1, 1] = scale[:, 1]
    S[:, 2, 2] = scale[:, 2]

    M_mat = torch.bmm(R, S)
    Sigma_old = torch.bmm(M_mat, M_mat.transpose(1, 2))
    Sigma_new = torch.bmm(torch.bmm(F, Sigma_old), F.transpose(1, 2))

    cov3D_precomp = torch.zeros((scale.shape[0], 6), device=scale.device)
    cov3D_precomp[:, 0] = Sigma_new[:, 0, 0]
    cov3D_precomp[:, 1] = Sigma_new[:, 0, 1]
    cov3D_precomp[:, 2] = Sigma_new[:, 0, 2]
    cov3D_precomp[:, 3] = Sigma_new[:, 1, 1]
    cov3D_precomp[:, 4] = Sigma_new[:, 1, 2]
    cov3D_precomp[:, 5] = Sigma_new[:, 2, 2]
    return cov3D_precomp
