"""The small model that sits on top of the frozen CLIP features. This is the only part that is trained."""
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, dim=512, hidden=512, num_classes=8, impute=False):
        super().__init__()
        self.impute = impute
        self.img_adapter = nn.Sequential(nn.Linear(dim, hidden), nn.GELU())
        self.txt_adapter = nn.Sequential(nn.Linear(dim, hidden), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Dropout(0.1))
        self.classifier = nn.Linear(hidden, num_classes)
        if impute:   # baseline: predict the missing CLIP vector from the one that is present
            self.txt2img = nn.Linear(dim, dim)
            self.img2txt = nn.Linear(dim, dim)

    def forward(self, img, txt, has_img, has_txt):
        if self.impute:
            hi, ht = has_img[:, None], has_txt[:, None]
            h_img = self.img_adapter(hi * img + (1 - hi) * self.txt2img(txt))
            h_txt = self.txt_adapter(ht * txt + (1 - ht) * self.img2txt(img))
        else:
            # A missing input is zero-masked: its adapter output is multiplied by 0.
            h_img = self.img_adapter(img) * has_img[:, None]
            h_txt = self.txt_adapter(txt) * has_txt[:, None]
        return self.classifier(self.fusion(torch.cat([h_img, h_txt], dim=1)))

    def recon_loss(self, img, txt, has_img, has_txt):
        """Imputation baseline: train the two predictors on the samples that have both inputs (cosine distance)."""
        both = (has_img == 1) & (has_txt == 1)
        if not both.any():
            return img.new_zeros(())
        cos = nn.functional.cosine_similarity
        return (1 - cos(self.txt2img(txt[both]), img[both])).mean() + (1 - cos(self.img2txt(img[both]), txt[both])).mean()
