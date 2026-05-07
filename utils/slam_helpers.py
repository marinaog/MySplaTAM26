import torch
import torch.nn.functional as F
import torch.nn as nn
from utils.slam_external import build_rotation

_FINAL_ACT_MAP = {
    "torch.exp": torch.exp,
    "torch.sigmoid": torch.sigmoid,
    "torch.relu": torch.relu,
    "none": lambda x: x,
}

class TinyColorMLP(nn.Module):
    def __init__(self, in_feats=16, dir_feats=3,
                 mid_feats_list=[16, 16], out_feats=3,
                 final_bias=None, final_act='torch.exp',
                 act='leaky_relu',detach=False) -> None:
        super().__init__()
        self.detach = detach
        if act == 'leaky_relu':
            self.act = nn.LeakyReLU(0.1, inplace=True)
        elif act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            raise NotImplementedError
        assert dir_feats == 3 or dir_feats == 9 or dir_feats == 19
        self.dir_feats = dir_feats
        feats = [in_feats+dir_feats] + mid_feats_list + [out_feats]
        self.linears = nn.ModuleList([nn.Linear(feats[i], feats[i+1]) for i in range(len(feats)-2)])
        self.linears.append(nn.Linear(feats[-2], feats[-1], bias=False))

        # ── Weight initialisation ────────────────────────────────────────────
        # PyTorch Linear defaults to kaiming_uniform_(a=√5), which is calibrated
        # for plain ReLU.  Our hidden activation is LeakyReLU(0.1), so we must
        # use a=0.1 to keep the variance of activations stable across layers.
        for layer in self.linears[:-1]:
            nn.init.kaiming_uniform_(layer.weight, a=0.1, nonlinearity='leaky_relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        # Final layer: zero-init weights so the MLP output (before adding
        # features_dc) starts at exactly 0.  This means the initial rendered
        # colour = exp(0 + features_dc), i.e. features_dc alone determines the
        # starting prediction — correct once features_dc is initialised from
        # log(point-cloud colour).  Large random final-layer weights would
        # produce chaotic initial renders and waste the first N optimisation steps.
        nn.init.zeros_(self.linears[-1].weight)

        if final_act not in _FINAL_ACT_MAP:
            raise ValueError(f"Unknown final_act '{final_act}'. Supported: {list(_FINAL_ACT_MAP)}")
        self.final_act = _FINAL_ACT_MAP[final_act]
        if final_bias is not None:
            self.final_bias = nn.Parameter(torch.ones(1, out_feats) * final_bias)
        else:
            self.register_buffer('final_bias', torch.zeros(1, out_feats))

    def forward(self, color_feats, dirs, w_bias=True):
        if self.detach:
            dirs = dirs.detach()
        if self.dir_feats >= 9:
            x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            dirs = torch.cat([dirs, xx, yy, zz, xy, yz, xz], dim=-1)
            if self.dir_feats == 19:
                xxx, xxy, xyy = xx * x, xx * y, xy * y
                zzz, xzz, xxz = zz * z, xz * z, xx * z
                yyy, yzz, yyz = yy * y, yz * z, yy * z
                xyz = xy * z
                dirs = torch.cat([dirs, xxx, xxy, xyy,
                                        zzz, xzz, xxz,
                                        yyy, yzz, yyz,
                                        xyz
                                 ], dim=-1)
        final_bias = self.final_bias.to(dirs.device)
        if isinstance(color_feats, tuple):
            final_bias = color_feats[0]
            color_feats = color_feats[1]

        # Squeeze in case of 3D shapes from legacy logic
        if color_feats.dim() == 3:
            color_feats = color_feats.squeeze(1)
        if final_bias.dim() == 3:
            final_bias = final_bias.squeeze(1)

        out = torch.cat([color_feats, dirs], dim=-1)
        for linear in self.linears[:-1]:
            out = self.act(linear(out))

        return self.linears[-1](out) + (final_bias if w_bias else 0)


def l1_loss_v1(x, y):
    return torch.abs((x - y)).mean()


def l1_loss_v2(x, y):
    return (torch.abs(x - y).sum(-1)).mean()


def rawnerf_loss(rgb_render_clip, gt, mask=None, eps=1e-2):
    """
    Reweighted L2 loss based on the gradient of the log tonemapping curve.
    This effectively penalizes relative error rather than absolute error.
    """
    # 1. Comparison in linear space
    resid_sq = (rgb_render_clip - gt) ** 2

    # 2. Scaling by the gradient of the log curve: 1 / (x + eps)
    # We detach the denominator so it acts as a fixed weight per pixel
    scaling_grad = 1.0 / (rgb_render_clip.detach() + eps)

    loss = resid_sq * (scaling_grad ** 2)

    if mask is not None:
        return loss[mask].mean()
    return loss.mean()


def weighted_l2_loss_v1(x, y, w):
    return torch.sqrt(((x - y) ** 2) * w + 1e-20).mean()


def weighted_l2_loss_v2(x, y, w):
    return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
rawnerf_eps    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def params2rendervar(params):
    # Check if Gaussians are Isotropic
    if params['log_scales'].shape[1] == 1:
        log_scales = torch.tile(params['log_scales'], (1, 3))
    else:
        log_scales = params['log_scales']
    # Initialize Render Variables
    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': params['rgb_colors'],
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar


def transformed_params2rendervar(params, transformed_gaussians, variables=None, tracking=False):
    # Check if Gaussians are Isotropic
    if params['log_scales'].shape[1] == 1:
        log_scales = torch.tile(params['log_scales'], (1, 3))
    else:
        log_scales = params['log_scales']

    # Check if we're using MLP for colors
    if variables is not None and 'color_mlp' in variables:
        # During tracking, detach means3D before computing view directions so the MLP
        # colour output is treated as a fixed appearance signal.
        means_for_dirs = transformed_gaussians['means3D'].detach() if tracking else transformed_gaussians['means3D']
        view_dirs = torch.nn.functional.normalize(means_for_dirs, dim=-1)
        colors_precomp = variables['color_mlp']((params['features_dc'], params['features_rest']), view_dirs)
    else:
        colors_precomp = params['rgb_colors']

    # Initialize Render Variables
    rendervar = {
        'means3D': transformed_gaussians['means3D'],
        'colors_precomp': colors_precomp,
        'rotations': F.normalize(transformed_gaussians['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar


def project_points(points_3d, intrinsics):
    """
    Function to project 3D points to image plane.
    params:
    points_3d: [num_gaussians, 3]
    intrinsics: [3, 3]
    out: [num_gaussians, 2]
    """
    points_2d = torch.matmul(intrinsics, points_3d.transpose(0, 1))
    points_2d = points_2d.transpose(0, 1)
    points_2d = points_2d / points_2d[:, 2:]
    points_2d = points_2d[:, :2]
    return points_2d

def params2silhouette(params):
    # Check if Gaussians are Isotropic
    if params['log_scales'].shape[1] == 1:
        log_scales = torch.tile(params['log_scales'], (1, 3))
    else:
        log_scales = params['log_scales']
    # Initialize Render Variables
    sil_color = torch.zeros_like(params['rgb_colors'])
    sil_color[:, 0] = 1.0
    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': sil_color,
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar


def transformed_params2silhouette(params, transformed_gaussians):
    # Check if Gaussians are Isotropic
    if params['log_scales'].shape[1] == 1:
        log_scales = torch.tile(params['log_scales'], (1, 3))
    else:
        log_scales = params['log_scales']
    # Initialize Render Variables
    sil_color = torch.zeros_like(params['rgb_colors'])
    sil_color[:, 0] = 1.0
    rendervar = {
        'means3D': transformed_gaussians['means3D'],
        'colors_precomp': sil_color,
        'rotations': F.normalize(transformed_gaussians['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar


def get_depth_and_silhouette(pts_3D, w2c):
    """
    Function to compute depth and silhouette for each gaussian.
    These are evaluated at gaussian center.
    """
    # Depth of each gaussian center in camera frame
    pts4 = torch.cat((pts_3D, torch.ones_like(pts_3D[:, :1])), dim=-1)
    pts_in_cam = (w2c @ pts4.transpose(0, 1)).transpose(0, 1)
    depth_z = pts_in_cam[:, 2].unsqueeze(-1) # [num_gaussians, 1]
    depth_z_sq = torch.square(depth_z) # [num_gaussians, 1]

    # Depth and Silhouette
    depth_silhouette = torch.zeros((pts_3D.shape[0], 3)).cuda().float()
    depth_silhouette[:, 0] = depth_z.squeeze(-1)
    depth_silhouette[:, 1] = 1.0
    depth_silhouette[:, 2] = depth_z_sq.squeeze(-1)

    return depth_silhouette


def params2depthplussilhouette(params, w2c):
    # Check if Gaussians are Isotropic
    if params['log_scales'].shape[1] == 1:
        log_scales = torch.tile(params['log_scales'], (1, 3))
    else:
        log_scales = params['log_scales']
    # Initialize Render Variables
    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': get_depth_and_silhouette(params['means3D'], w2c),
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar


def transformed_params2depthplussilhouette(params, w2c, transformed_gaussians):
    # Check if Gaussians are Isotropic
    if params['log_scales'].shape[1] == 1:
        log_scales = torch.tile(params['log_scales'], (1, 3))
    else:
        log_scales = params['log_scales']
    # Initialize Render Variables
    rendervar = {
        'means3D': transformed_gaussians['means3D'],
        'colors_precomp': get_depth_and_silhouette(transformed_gaussians['means3D'], w2c),
        'rotations': F.normalize(transformed_gaussians['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar


def transform_to_frame(params, time_idx, gaussians_grad, camera_grad):
    """
    Function to transform Isotropic or Anisotropic Gaussians from world frame to camera frame.

    Args:
        params: dict of parameters
        time_idx: time index to transform to
        gaussians_grad: enable gradients for Gaussians
        camera_grad: enable gradients for camera pose

    Returns:
        transformed_gaussians: Transformed Gaussians (dict containing means3D & unnorm_rotations)
    """
    # Get Frame Camera Pose
    if camera_grad:
        cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx])
        cam_tran = params['cam_trans'][..., time_idx]
    else:
        cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        cam_tran = params['cam_trans'][..., time_idx].detach()
    rel_w2c = torch.eye(4).cuda().float()
    rel_w2c[:3, :3] = build_rotation(cam_rot)
    rel_w2c[:3, 3] = cam_tran

    # Check if Gaussians need to be rotated (Isotropic or Anisotropic)
    if params['log_scales'].shape[1] == 1:
        transform_rots = False # Isotropic Gaussians
    else:
        transform_rots = True # Anisotropic Gaussians

    # Get Centers and Unnorm Rots of Gaussians in World Frame
    if gaussians_grad:
        pts = params['means3D']
        unnorm_rots = params['unnorm_rotations']
    else:
        pts = params['means3D'].detach()
        unnorm_rots = params['unnorm_rotations'].detach()

    transformed_gaussians = {}
    # Transform Centers of Gaussians to Camera Frame
    pts_ones = torch.ones(pts.shape[0], 1).cuda().float()
    pts4 = torch.cat((pts, pts_ones), dim=1)
    transformed_pts = (rel_w2c @ pts4.T).T[:, :3]
    transformed_gaussians['means3D'] = transformed_pts
    # Transform Rots of Gaussians to Camera Frame
    if transform_rots:
        norm_rots = F.normalize(unnorm_rots)
        transformed_rots = quat_mult(cam_rot, norm_rots)
        transformed_gaussians['unnorm_rotations'] = transformed_rots
    else:
        transformed_gaussians['unnorm_rotations'] = unnorm_rots

    return transformed_gaussians