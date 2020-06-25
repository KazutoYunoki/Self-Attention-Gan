import pathlib
import torchvision.datasets as dset
import torchvision.transforms as transforms


def make_dataset(dataroot, image_size):
    current_dir = pathlib.Path(__file__).resolve().parent
    print(current_dir)

    # データセットの作成
    dataset = dset.ImageFolder(
        root=str(current_dir) + dataroot,
        transform=transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                # 白黒
                # transforms.Normalize((0.5), (0.5,))
                # カラー
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )

    return dataset


if __name__ == "__main__":
    dataset = make_dataset(dataroot="/data", image_size=64)
    print(dataset)
