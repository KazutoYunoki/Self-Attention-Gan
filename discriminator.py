import torch.nn as nn

from self_attention import Self_Attention


class Discriminator(nn.Sequential):
    def __init__(self, ngpu, ndf, nc):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.ndf = ndf
        self.nc = nc

        self.layer1 = nn.Sequential(
            # input is (nc) * 64 * 64
            # Spectral Normalizationを追加
            nn.utils.spectral_norm(nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer2 = nn.Sequential(
            # state size. (ndf) * 32 * 32
            # Spectral Normalizationを追加
            nn.utils.spectral_norm(
                nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False)
            ),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer3 = nn.Sequential(
            # state size. (ndf * 2) * 16 * 16
            # Spectral Normalizationを追加
            nn.utils.spectral_norm(
                nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False)
            ),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Self-Attention層を追加
        self.self_attention1 = Self_Attention(in_dim=ndf * 4)

        self.layer4 = nn.Sequential(
            # state size. (ndf * 4) * 8 * 8
            # Spectral Normalizationを追加
            nn.utils.spectral_norm(
                nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False)
            ),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Self-Attention層の追加
        self.self_attention2 = Self_Attention(in_dim=ndf * 8)

        self.last = nn.Sequential(
            # state size. (ndf * 8) * 4 * 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid(), # 本ではSigmoid関数を通してないためいったんコメントアウト
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
    D = Discriminator(ngpu=1, ndf=64, nc=3)
    print(D)
