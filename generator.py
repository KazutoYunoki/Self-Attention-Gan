# Generator Code
import torch.nn as nn

from self_attention import Self_Attention


class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.ngf = ngf
        self.nc = nc

        self.layer1 = nn.Sequential(
            # 入力Zを畳み込み層へ
            # Spectral Normalizationを追加
            nn.utils.spectral_norm(
                nn.ConvTranspose2d(
                    self.nz,
                    self.ngf * 8,
                    kernel_size=4,
                    stride=1,
                    padding=0,
                    bias=False,
                )
            ),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
        )

        self.layer2 = nn.Sequential(
            # state size (self.ngf*8) * 4 * 4
            # Spectoral Normalizationを追加
            nn.utils.spectral_norm(
                nn.ConvTranspose2d(
                    self.ngf * 8,
                    self.ngf * 4,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            ),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
        )

        self.layer3 = nn.Sequential(
            # state size. (self.ngf*4) x 8 x 8
            # Spectral Normalizationを追加
            nn.utils.spectral_norm(
                nn.ConvTranspose2d(
                    self.ngf * 4,
                    self.ngf * 2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            ),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
        )

        # Self-Attention層を追加
        self.self_attention1 = Self_Attention(in_dim=ngf * 2)

        self.layer4 = nn.Sequential(
            # state size. (self.ngf*2) x 16 x 16
            nn.utils.spectral_norm(
                nn.ConvTranspose2d(
                    self.ngf * 2,
                    self.ngf,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            ),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
        )

        # Self-Attention層を追加
        self.self_attention2 = Self_Attention(in_dim=ngf)

        self.last = nn.Sequential(
            # state size. (self.ngf) x 32 x 32
            nn.ConvTranspose2d(
                self.ngf, self.nc, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, z):
        out = self.layer1(z)
        out = self.layer2(out)
        out = self.layer3(out)
        out, attention_map1 = self.self_attention1(out)
        out = self.layer4(out)
        out, attention_map2 = self.self_attention2(out)
        out = self.last(out)

        return out, attention_map1, attention_map2


if __name__ == "__main__":
    G = Generator(ngpu=1, nz=100, ngf=64, nc=3)
    print(G)
