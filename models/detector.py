class DetectionHead(nn.Module):
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.class_head = nn.Linear(d_model, num_classes)
        self.box_head = nn.Linear(d_model, 4)  # x,y,w,h
    
    def forward(self, x):
        return self.class_head(x), self.box_head(x)
