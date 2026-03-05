"""
按 YOLO 数据集格式建立 val 划分：在 images 与 labels 下创建 val 文件夹，
从 train 中按 9:1 比例挑选样本到 val（移动图像与对应标签）。
"""

import random
from pathlib import Path

# 常见图片扩展名
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def split_train_val(
    root: str | Path,
    val_ratio: float = 0.1,
    seed: int | None = 42,
) -> tuple[int, int]:
    """
    在 root（含 images/train、labels/train）下创建 images/val、labels/val，
    从 train 中按比例随机选取样本移动到 val，保持图像与标签一一对应。

    Args:
        root: 数据集根路径，如 F:\\datesets\\train_cross_detect_dataset\\train_detect_dataset
        val_ratio: 验证集比例，默认 0.1（即 9:1）
        seed: 随机种子，默认 42 保证可复现

    Returns:
        (移动到 val 的图像数, 移动到 val 的标签数)
    """
    root = Path(root)
    images_train = root / "images" / "train"
    labels_train = root / "labels" / "train"
    images_val = root / "images" / "val"
    labels_val = root / "labels" / "val"

    if not images_train.is_dir():
        raise FileNotFoundError(f"不存在: {images_train}")
    if not labels_train.is_dir():
        raise FileNotFoundError(f"不存在: {labels_train}")

    images_val.mkdir(parents=True, exist_ok=True)
    labels_val.mkdir(parents=True, exist_ok=True)

    # 收集 train 中所有图片（按 stem 去重，只认一个扩展名）
    stems = set()
    stem_to_image = {}
    for p in images_train.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            stem = p.stem
            stems.add(stem)
            stem_to_image[stem] = p

    if not stems:
        return 0, 0

    stem_list = sorted(stems)
    if seed is not None:
        random.seed(seed)
    random.shuffle(stem_list)

    n_val = max(1, int(len(stem_list) * val_ratio))
    val_stems = set(stem_list[:n_val])

    moved_images = 0
    moved_labels = 0

    for stem in val_stems:
        img_src = stem_to_image.get(stem)
        if img_src and img_src.is_file():
            img_dst = images_val / img_src.name
            img_src.rename(img_dst)
            moved_images += 1

        label_src = labels_train / f"{stem}.txt"
        if label_src.is_file():
            label_dst = labels_val / label_src.name
            label_src.rename(label_dst)
            moved_labels += 1

    return moved_images, moved_labels


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="按 YOLO 格式建立 val，从 train 中 9:1 划分并移动文件"
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=r"F:\datesets\train_cross_detect_dataset\train_detect_dataset",
        help="数据集根路径（含 images/train、labels/train）",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="验证集比例，默认 0.1（9:1）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，默认 42",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将移动的文件数量与示例，不实际移动",
    )
    args = parser.parse_args()

    root = Path(args.root)
    images_train = root / "images" / "train"
    labels_train = root / "labels" / "train"

    if not images_train.is_dir():
        print(f"错误: 不存在 {images_train}")
        return 1
    if not labels_train.is_dir():
        print(f"错误: 不存在 {labels_train}")
        return 1

    stems = set()
    stem_to_image = {}
    for p in images_train.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            stem = p.stem
            stems.add(stem)
            stem_to_image[stem] = p

    if not stems:
        print("images/train 下没有图片")
        return 0

    stem_list = sorted(stems)
    random.seed(args.seed)
    random.shuffle(stem_list)
    n_val = max(1, int(len(stem_list) * args.val_ratio))
    val_stems = set(stem_list[:n_val])

    if args.dry_run:
        print(f"train 样本数: {len(stem_list)}, 将移至 val: {n_val} (比例 {args.val_ratio})")
        for i, stem in enumerate(list(val_stems)[:5]):
            img = stem_to_image.get(stem)
            lb = labels_train / f"{stem}.txt"
            print(f"  图像: {img.name if img else '?'} -> images/val/")
            print(f"  标签: {lb.name} -> labels/val/ {'(存在)' if lb.is_file() else '(不存在)'}")
        if n_val > 5:
            print(f"  ... 共 {n_val} 组")
        print("未执行移动 (--dry-run)")
        return 0

    images_val = root / "images" / "val"
    labels_val = root / "labels" / "val"
    images_val.mkdir(parents=True, exist_ok=True)
    labels_val.mkdir(parents=True, exist_ok=True)

    moved_images = 0
    moved_labels = 0
    for stem in val_stems:
        img_src = stem_to_image.get(stem)
        if img_src and img_src.is_file():
            img_src.rename(images_val / img_src.name)
            moved_images += 1
        label_src = labels_train / f"{stem}.txt"
        if label_src.is_file():
            label_src.rename(labels_val / label_src.name)
            moved_labels += 1

    print(f"已创建 images/val、labels/val，移动图像 {moved_images} 张，标签 {moved_labels} 个")
    return 0


if __name__ == "__main__":
    exit(main())
