class PointEncoder(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, points):
        return self.mlp(points)
