import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

# ─────────────────────────────────────────────
# UNet Building Blocks
# ─────────────────────────────────────────────


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        # Residual projection if channel dims differ
        self.res = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.res(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ─────────────────────────────────────────────
# UNet Classifier
# ─────────────────────────────────────────────


class UNetClassifier(nn.Module):
    """
    UNet encoder-decoder with global average pooling head for classification.
    Input:  (B, 1, 28, 28)
    Output: (B, num_classes)  — raw logits
    """

    def __init__(self, num_classes=10, base_ch=16, dropout=0.1):
        super().__init__()
        # Encoder
        self.down1 = DownBlock(1, base_ch)  # 28→14
        self.down2 = DownBlock(base_ch, base_ch * 2)  # 14→7
        self.down3 = DownBlock(base_ch * 2, base_ch * 4)  # 7→3

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8, dropout=dropout)

        # Decoder (reconstructs spatial structure — good for feature richness)
        self.up1 = UpBlock(base_ch * 8, base_ch * 4)
        self.up2 = UpBlock(base_ch * 4, base_ch * 2)
        self.up3 = UpBlock(base_ch * 2, base_ch)

        # Classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_ch, base_ch * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(base_ch * 2, num_classes),
        )

    def forward(self, x):
        # Pad 28x28 → 32x32 for clean power-of-2 pooling
        x = F.pad(x, (2, 2, 2, 2))  # → (B,1,32,32)

        x, s1 = self.down1(x)  # x:(B,32,16,16)  s1:(B,32,32,32)
        x, s2 = self.down2(x)  # x:(B,64,8,8)    s2:(B,64,16,16)
        x, s3 = self.down3(x)  # x:(B,128,4,4)   s3:(B,128,8,8)

        x = self.bottleneck(x)  # (B,256,4,4)

        x = self.up1(x, s3)  # (B,128,8,8)
        x = self.up2(x, s2)  # (B,64,16,16)
        x = self.up3(x, s1)  # (B,32,32,32)

        return self.head(x)

    def get_features(self, x):
        """Return bottleneck features for FID computation."""
        x = F.pad(x, (2, 2, 2, 2))
        x, _ = self.down1(x)
        x, _ = self.down2(x)
        x, _ = self.down3(x)
        x = self.bottleneck(x)
        return F.adaptive_avg_pool2d(x, 1).flatten(1)  # (B, base_ch*8)
