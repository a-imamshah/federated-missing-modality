"""The small model that sits on top of the frozen CLIP features. This is the only part that is trained."""
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, dim=512, hidden=512, num_classes=8):
        super().__init__()
        self.img_adapter = nn.Sequential(nn.Linear(dim, hidden), nn.GELU())
        self.txt_adapter = nn.Sequential(nn.Linear(dim, hidden), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Dropout(0.1))
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(self, img, txt, has_img, has_txt):
        # A missing input is zero-masked: its adapter output is multiplied by 0.
        h_img = self.img_adapter(img) * has_img[:, None]
        h_txt = self.txt_adapter(txt) * has_txt[:, None]
        return self.classifier(self.fusion(torch.cat([h_img, h_txt], dim=1)))
