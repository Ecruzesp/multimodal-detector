def depth_to_pointcloud(depth, K):
    H, W = depth.shape
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W))
    
    z = depth
    x3d = (x - K[0,2]) * z / K[0,0]
    y3d = (y - K[1,2]) * z / K[1,1]
    
    points = torch.stack([x3d, y3d, z], dim=-1)  # (H, W, 3)
    return points.reshape(-1, 3)

def transform_points(points, R, t):
    return (R @ points.T).T + t

def project_to_image(points, K):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    u = (K[0,0] * x / z) + K[0,2]
    v = (K[1,1] * y / z) + K[1,2]

    return torch.stack([u, v], dim=-1)

def sample_features(feature_map, uv):
    # normalize to [-1, 1] for grid_sample
    H, W = feature_map.shape[-2:]
    uv_norm = uv.clone()
    uv_norm[:, 0] = (uv[:, 0] / W) * 2 - 1
    uv_norm[:, 1] = (uv[:, 1] / H) * 2 - 1
    
    grid = uv_norm.view(1, -1, 1, 2)
    sampled = F.grid_sample(feature_map, grid, align_corners=True)
    
    return sampled.squeeze().T  # (N, C)
