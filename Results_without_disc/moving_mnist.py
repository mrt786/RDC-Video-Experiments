import socket
import glob
import os
import numpy as np
from PIL import Image
from torchvision import datasets, transforms

class MovingMNIST(object):

    """Data Handler that creates Bouncing MNIST dataset on the fly."""

    def __init__(self, train, data_root, seq_len=20, num_digits=2, image_size=64, deterministic=False):
        path = data_root
        self.seq_len = seq_len
        self.num_digits = num_digits
        self.image_size = image_size
        self.step_length = 0.1
        self.digit_size = int(image_size/2)
        self.deterministic = deterministic
        self.seed_is_set = False # multi threaded loading
        self.channels = 1

        self.data = datasets.MNIST(
            path,
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(self.digit_size),
                 transforms.ToTensor()]))

        self.N = len(self.data)

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        self.set_seed(index)
        image_size = self.image_size
        digit_size = self.digit_size
        x = np.zeros((self.seq_len,
                      image_size,
                      image_size,
                      self.channels),
                    dtype=np.float32)
        for n in range(self.num_digits):
            idx = np.random.randint(self.N)
            # digit,  = self.data[idx]
            # ---   Classification   ---
            digit, label = self.data[idx]
            # ---                   ---

            sx = np.random.randint(image_size-digit_size)
            sy = np.random.randint(image_size-digit_size)
            dx = np.random.randint(-4, 5)
            dy = np.random.randint(-4, 5)
            for t in range(self.seq_len):
                dx = np.random.randint(-4, 5)
                dy = np.random.randint(-4, 5)
                if sy < 0:
                    sy = 0
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(1, 5)
                        dx = np.random.randint(-4, 5)
                elif sy >= image_size-self.digit_size:
                    sy = image_size-self.digit_size-1
                    if self.deterministic:
                        dy = -dy
                    else:
                        dy = np.random.randint(-4, 0)
                        dx = np.random.randint(-4, 5)

                if sx < 0:
                    sx = 0
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(1, 5)
                        dy = np.random.randint(-4, 5)
                elif sx >= image_size-self.digit_size:
                    sx = image_size-self.digit_size-1
                    if self.deterministic:
                        dx = -dx
                    else:
                        dx = np.random.randint(-4, 0)
                        dy = np.random.randint(-4, 5)

                x[t, sy:sy+self.digit_size, sx:sx+self.digit_size, 0] += digit.numpy().squeeze()
                sy += dy
                sx += dx

        x[x>1] = 1.
        return x, label


class kth(object):

    """Data handler for grayscale KTH action-recognition video sequences."""

    def __init__(self, train, data_root, seq_len=20, image_size=64):
        self.seq_len = seq_len
        self.image_size = image_size
        self.channels = 1
        self.classes = [
            "boxing",
            "handclapping",
            "handwaving",
            "jogging",
            "running",
            "walking",
        ]
        self.samples = []

        persons = range(1, 21) if train else range(21, 26)
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        for label, action in enumerate(self.classes):
            action_dir = os.path.join(data_root, action)
            for person in persons:
                person_dir = os.path.join(action_dir, f"person{person:02d}")
                for video_dir in sorted(glob.glob(os.path.join(person_dir, "*d4"))):
                    frame_paths = sorted(
                        glob.glob(os.path.join(video_dir, "image-*.png"))
                    )
                    for start in range(len(frame_paths) - seq_len + 1):
                        self.samples.append(
                            (frame_paths[start:start + seq_len], label)
                        )
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        frame_paths, label = self.samples[index]
        frames = []

        for frame_path in frame_paths:
            with Image.open(frame_path) as frame:
                frame = self.transform(frame.convert("L"))
            frames.append(frame.squeeze(0).numpy())

        sequence = np.stack(frames, axis=0)[..., np.newaxis]
        return sequence.astype(np.float32), label




