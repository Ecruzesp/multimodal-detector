def depth_to_pointcloud(depth, K):
    H, W = depth.shape
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W))
    
    z = depth
    x3d = (x - K[0,2]) * z / K[0,0]
    y3d = (y - K[1,2]) * z / K[1,1]
    
    points = torch.stack([x3d, y3d, z], dim=-1)  # (H, W, 3)
    return points.reshape(-1, 3)
