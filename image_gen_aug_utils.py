import torch
from torch import nn

# CODE IS FROM https://github.com/hamidkazemi22/vit-visualization/blob/main/augmentation/pre.py

class ColorJitter(nn.Module):
    def __init__(self, batch_size: int, shuffle_every: bool = False, mean: float = 1., std: float = 1.):
        super().__init__()
        self.batch_size, self.mean_p, self.std_p = batch_size, mean, std
        self.mean = self.std = None
        self.shuffle()
        self.shuffle_every = shuffle_every

    def shuffle(self):
        self.mean = (torch.rand((self.batch_size, 3, 1, 1,)).cuda() - 0.5) * 2 * self.mean_p
        self.std = ((torch.rand((self.batch_size, 3, 1, 1,)).cuda() - 0.5) * 2 * self.std_p).exp()

    def forward(self, img: torch.tensor) -> torch.tensor:
        if self.shuffle_every:
            self.shuffle()
        return (img - self.mean) / self.std
    
class GaussianNoise(nn.Module):
    def __init__(self, batch_size: int, shuffle_every: bool = False, std: float = 1., max_iter: int = 2000):
        super().__init__()
        self.batch_size, self.std_p, self.max_iter = batch_size, std, max_iter
        self.std = None
        self.rem = max_iter - 1
        self.shuffle()
        self.shuffle_every = shuffle_every

    def shuffle(self):
        self.std = torch.randn(self.batch_size, 3, 1, 1).cuda() * self.rem * self.std_p / self.max_iter
        self.rem = (self.rem - 1 + self.max_iter) % self.max_iter

    def forward(self, img: torch.tensor) -> torch.tensor:
        if self.shuffle_every:
            self.shuffle()
        return img + self.std
    
class Centering(nn.Module):
    def __init__(self, std: float, max_iter: int, centering_size: list):
        super().__init__()
        self.std = std
        self.max_iter = max_iter
        self.centering_size = centering_size
        self.iter = 0

    def forward(self, img: torch.tensor) -> torch.tensor:
        now_sz_idx = self.iter // (self.max_iter)
        self.size = self.centering_size[now_sz_idx]
        self.iter+=1
        
        pert = (torch.rand(2) * 2 - 1) * self.std
        w, h = img.shape[-2:]
        x = (pert[0] + w // 2 - self.size // 2).long().clamp(min=0, max=w - self.size)
        y = (pert[1] + h // 2 - self.size // 2).long().clamp(min=0, max=h - self.size)
        return img[:, :, x:x + self.size, y:y + self.size]


class Zoom(nn.Module):
    def __init__(self, out_size: int = 384):
        super().__init__()
        self.up = torch.nn.Upsample(size=(out_size, out_size), mode='bilinear', align_corners=False).cuda()

    def forward(self, img: torch.tensor) -> torch.tensor:
        return self.up(img)