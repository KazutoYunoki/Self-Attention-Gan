from __future__ import print_function

# %matplotlib inline
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt

import hydra
import logging
from generator import Generator
from discriminator import Discriminator
from dataset import make_dataset

# A logger for this file
log = logging.getLogger(__name__)

# GeneratorとDiscriminatorの重みの初期化


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


@hydra.main(config_path="conf/config.yaml")
def main(cfg):

    # 再現性のためにシード値をセット
    manualSeed = 999

    # manualSeed = random.randint(1, 10000) # 新しい結果がほしい場合に使用
    log.info("Random Seed: " + str(manualSeed))
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # datsetの作成
    dataset = make_dataset(cfg.dataroot, cfg.image_size)

    # データローダの作成
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers
    )

    # どのデバイスで実行するか決定
    device = torch.device(
        "cuda:0" if (torch.cuda.is_available() and cfg.ngpu > 0) else "cpu"
    )

    # 訓練画像をプロット

    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(
                real_batch[0].to(device)[:64], padding=2, normalize=True
            ).cpu(),
            (1, 2, 0),
        )
    )
    plt.savefig("training_img")

    # Generator作成
    netG = Generator(cfg.ngpu, cfg.nz, cfg.ngf, cfg.nc).to(device)

    # マルチGPUを望むなら
    if (device.type == "cuda") and (cfg.ngpu > 1):
        netG = nn.DataParallel(netG, list(range(cfg.ngpu)))

    # 重みの初期化関数を適用
    netG.apply(weights_init)

    # モデルの印字
    log.info(netG)

    # Discriminator作成
    netD = Discriminator(cfg.ngpu, cfg.ndf, cfg.nc).to(device)

    # マルチGPUを望むなら
    if (device.type == "cuda") and (cfg.ngpu > 1):
        netD = nn.DataParallel(netD, list(range(cfg.ngpu)))

    # 重みの初期化関数を適用
    netD.apply(weights_init)

    # モデルの印字
    log.info(netD)

    # 損失関数の定義
    # criterion = nn.BCELoss()

    # 潜在ベクトルを作成 Generatorの進歩を可視化するため
    fixed_noise = torch.randn(64, cfg.nz, 1, 1, device=device)

    # 学習中の本物と偽物のラベルを作成
    # real_label = 1
    # fake_label = 0

    # 最適化関数Adamを設定
    optimizerD = optim.Adam(netD.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))

    # 学習ループ

    # 結果を保存しておくリスト
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(cfg.num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ##########
            # 1. Discriminatorの更新
            ##########

            # 本物のバッチの学習
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            # label = torch.full((b_size,), real_label, device=device)

            # 本物のバッチをDiscriminatorへ
            output, _, _ = netD(real_cpu)
            output = output.view(-1)

            # 損失値の計算 → hinge version of the adversarial lossに変更
            # errD_real = criterion(output, label)
            # 誤差 output が1以上で誤差0になる。output>1で、1.0-outputが負の場合ReLUで0
            errD_real = torch.nn.ReLU()(1.0 - output).mean()

            # bpによる勾配計算
            errD_real.backward()
            D_x = output.mean().item()

            # 偽物のバッチ学習　潜在ベクトル生成
            noise = torch.randn(b_size, cfg.nz, 1, 1, device=device)

            # 偽画像の生成
            fake, _, _ = netG(noise)
            # label.fill_(fake_label)

            # 偽画像の分類
            output, _, _ = netD(fake.detach())
            output = output.view(-1)

            # 偽画像の損失値の計算　hinge version of the adversarial lossに変更
            # errD_fake = criterion(output, label)
            # 誤差 outpuが―1以下なら誤差0になる。output<-1で、1.0+outputが負の場合ReLUで0にする
            errD_fake = torch.nn.ReLU()(1.0 + output).mean()

            # 勾配の計算
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            # 本物と偽物の勾配を足す
            errD = errD_real + errD_fake

            # Discriminatorの更新
            optimizerD.step()

            ##########
            # 2.Generatorの更新
            ##########
            netG.zero_grad()
            # label.fill_(real_label)

            output, _, _ = netD(fake)
            output = output.view(-1)

            # GeneratorのLoss計算 hinge version of the adversarial lossに変更
            # errG = criterion(output, label)
            errG = -output.mean()

            # Generatorの勾配計算
            errG.backward()
            D_G_z2 = output.mean().item()

            # Generatorの更新
            optimizerG.step()

            # 学習の状態を出力

            if i % 50 == 0:
                print(
                    "[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f"
                    % (
                        epoch,
                        cfg.num_epochs,
                        i,
                        len(dataloader),
                        errD.item(),
                        errG.item(),
                        D_x,
                        D_G_z1,
                        D_G_z2,
                    )
                )

            # Lossを保存する
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Generatorの動作確認と出力画像を保存
            if (iters % 500 == 0) or (
                (epoch == cfg.num_epochs - 1) and (i == len(dataloader) - 1)
            ):
                with torch.no_grad():
                    fake, _, _ = netG(fixed_noise)
                    fake = fake.detach().cpu()

                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig("Genarator_Discriminator_Loss.png")

    # データローダから本物の画像を取得
    real_batch = next(iter(dataloader))

    # Real images のplot
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(
                real_batch[0].to(device)[:64], padding=5, normalize=True
            ).cpu(),
            (1, 2, 0),
        )
    )

    # 最後のエポックの偽画像を表示
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.show()
    plt.savefig("result_img.png")


if __name__ == "__main__":
    main()
