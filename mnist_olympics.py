import torch, gzip, os, requests
from tqdm import trange
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

image_size = 28

device = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)


def download_file(url, save_path):
    response = requests.get(url, stream=True)
    with open(save_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=128):
            file.write(chunk)


def get_file_from_url(url, local_path):
    if not os.path.exists(local_path):
        print(f"Downloading file from {url}...")
        download_file(url, local_path)
        print("Download complete.")
    else:
        print(f"File {local_path} already exists.")


get_file_from_url(
    "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    "train-images-idx3-ubyte.gz",
)
get_file_from_url(
    "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
)
get_file_from_url(
    "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
)
get_file_from_url(
    "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)


train_images = (
    np.frombuffer(
        gzip.open("train-images-idx3-ubyte.gz").read(),
        dtype=np.uint8,
    )[16:]
    .reshape(-1, 1, 28, 28)
    .astype(np.float32)
)
train_labels = np.frombuffer(
    gzip.open("train-labels-idx1-ubyte.gz").read(),
    dtype=np.uint8,
)[8:]
test_images = (
    np.frombuffer(
        gzip.open("t10k-images-idx3-ubyte.gz").read(),
        dtype=np.uint8,
    )[16:]
    .reshape(-1, 1, 28, 28)
    .astype(np.float32)
)
test_labels = np.frombuffer(
    gzip.open("t10k-labels-idx1-ubyte.gz").read(),
    dtype=np.uint8,
)[8:]

n = int(0.9 * train_images.shape[0])

X_tr, Y_tr = train_images[:n], train_labels[:n]
X_val, Y_val = train_images[n:], train_labels[n:]
X_test, Y_test = test_images, test_labels


class BroNet(nn.Module):
    def __init__(self):
        k = 5
        super().__init__()
        self.conv1 = nn.Conv2d(
            1,
            32,
            kernel_size=k,
            stride=1,
        )
        self.conv2 = nn.Conv2d(
            32,
            64,
            kernel_size=k,
            stride=1,
        )

        self.bn = nn.BatchNorm2d(64)
        self.dp = nn.Dropout(0.2)
        self.maxp = nn.MaxPool2d(3, 3)
        self.ff1 = nn.Linear(64 * 6 * 6, 128)
        self.ff2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = F.relu(self.conv2(x))
        x = self.bn(x)
        x = self.dp(x)
        # print(x.shape)
        x = self.maxp(x)
        # print(x.shape)
        x = torch.flatten(x, -3)
        x = F.relu(self.ff1(x))
        x = self.ff2(x)
        return x


model = BroNet()
model.to(device)

batch_size = 512
total_steps = 1000
optim = torch.optim.AdamW(model.parameters())
loss_i, acc_i = [], []


def train():
    for i in (t := trange(total_steps)):
        samp = torch.randint(0, X_tr.shape[0], (batch_size,))
        logits = model(torch.tensor(X_tr[samp], device=device, dtype=torch.float))
        targets = F.one_hot(
            torch.tensor(Y_tr[samp], dtype=torch.long, device=device), num_classes=10
        ).float()
        # print(logits.squeeze(1).shape, targets.shape)
        loss = F.cross_entropy(logits.squeeze(1), targets)
        optim.zero_grad()
        loss.backward()
        optim.step()
        acc = (
            torch.sum(
                torch.argmax(logits.squeeze(1), dim=1)
                == torch.tensor(Y_tr[samp], device=device)
            )
            / batch_size
        )
        t.set_description(f"loss {loss:8.4f} acc {acc:8.4f}")
        loss_i.append(loss.item())
        acc_i.append(acc.item())


def eval(mode):
    x, y = {"train": (X_tr, Y_tr), "val": (X_val, Y_val), "test": (X_test, Y_test)}[
        mode
    ]
    logits = model(torch.tensor(x, device=device))
    acc = (
        sum(torch.argmax(logits.squeeze(1), dim=1) == torch.tensor(y, device=device))
        / x.shape[0]
    )
    print(f"{mode} {acc:8.4f}")


train()
eval("train")
eval("val")
eval("test")
# plt.plot(loss_i)
# plt.plot(acc_i)
# plt.show()
