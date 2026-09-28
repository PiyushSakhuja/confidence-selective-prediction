"""SmallCNN: the single model used in this study. Do not modify after locking."""
import torch
import torch.nn as nn


def _block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 32x32 -> 16x16 -> 8x8 -> 4x4
        self.features = nn.Sequential(_block(3, 32), _block(32, 64), _block(64, 128))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))  # raw logits (no softmax)


if __name__ == "__main__":
    m = SmallCNN()
    out = m(torch.randn(1, 3, 32, 32))
    print(out.shape)  # expect torch.Size([1, 10])
    print("params:", sum(p.numel() for p in m.parameters()))
