import torch
from generator import Generator
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils


def generate_img():
    # 潜在ベクトルを生成
    nz = 100
    for i in range(100):
        fixed_noise = torch.randn(1, nz, 1, 1)

        # Generator作成
        netG = Generator(0, nz, 64, 3)

        # Pytorchのネットワークパラメータのロード
        # 重みをロード
        load_path = "./melanoma_generator.pth"
        load_weights = torch.load(load_path, map_location=torch.device("cpu"))
        netG.load_state_dict(load_weights)

        with torch.no_grad():
            fake, _, _ = netG(fixed_noise)

        print(fake.shape)

        vutils.save_image(fake[0], "generate_img_" + str(i) + ".jpg", normalize=True)

        """
        # Real imagesのプロット
        plt.figure()
        plt.axis("off")
        plt.imshow(np.transpose(fake[0], (1, 2, 0)))
        plt.savefig(
            "generate_img_" + str(nz) + ".jpg", bbox_inches="tight", pad_inches=0
        )
        plt.show()
        vutils.save_image()
        """


if __name__ == "__main__":
    generate_img()
