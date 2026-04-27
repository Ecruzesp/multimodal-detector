class FusionTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead)
    
    def forward(self, queries, rgb_feats, depth_feats):
        # concatenate modalities
        keys = torch.cat([rgb_feats, depth_feats], dim=0)
        values = keys
        
        out, _ = self.cross_attn(queries, keys, values)
        return out
