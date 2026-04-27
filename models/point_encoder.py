def transform_points(points, R, t):
    return (R @ points.T).T + t
