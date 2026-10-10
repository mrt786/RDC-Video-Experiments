import os
import argparse
import numpy as np
from PIL import Image
from moving_mnist import MovingMNIST


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--data-root", default="./data/")
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--num-digits", type=int, default=1)
    parser.add_argument("--out-dir", default="dataset_preview")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dataset = MovingMNIST(
        train=True,
        data_root=args.data_root,
        seq_len=args.seq_len,
        num_digits=args.num_digits,
        image_size=64,
    )

    for i in range(args.n):
        video, label = dataset[i]
        frames = (video.squeeze(-1) * 255).astype(np.uint8)
        pil_frames = [Image.fromarray(f).convert("L") for f in frames]
        out_path = os.path.join(args.out_dir, f"sample_{i}_label_{label}.gif")
        pil_frames[0].save(out_path, save_all=True, append_images=pil_frames[1:], duration=100, loop=0)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
