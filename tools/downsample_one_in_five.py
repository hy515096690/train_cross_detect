"""
每五张图片保留一张，其余四张删除。
处理路径下 1-6 文件夹中的 *.jpg，存储路径不变（原地删除）。
"""

import os
from pathlib import Path


def downsample_one_in_five(root: str | Path) -> tuple[int, int]:
    """
    在 root 下的 1-6 文件夹中，每五张 jpg 保留一张，删除其余四张。

    Args:
        root: 数据集根路径，如 F:\\datesets\\train_cross_detect_dataset

    Returns:
        (保留数量, 删除数量)
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"路径不存在或不是目录: {root}")

    kept = 0
    deleted = 0

    for folder_name in ("1", "2", "3", "4", "5", "6"):
        folder = root / folder_name
        if not folder.is_dir():
            continue

        jpgs = sorted(folder.glob("*.jpg"))
        if not jpgs:
            continue

        # 每五张保留一张：索引 0, 5, 10, ...
        for i, p in enumerate(jpgs):
            if i % 5 == 0:
                kept += 1
            else:
                try:
                    p.unlink()
                    deleted += 1
                except OSError as e:
                    print(f"删除失败 {p}: {e}")

    return kept, deleted


def main():
    import argparse

    parser = argparse.ArgumentParser(description="每五张 jpg 保留一张，其余删除（1-6 文件夹）")
    parser.add_argument(
        "root",
        nargs="?",
        default=r"F:\datesets\train_cross_detect_dataset",
        help="数据集根路径（默认: F:\\datesets\\train_cross_detect_dataset）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将保留/删除的文件，不实际删除",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"错误: 路径不存在或不是目录: {root}")
        return 1

    if args.dry_run:
        total_keep = 0
        total_del = 0
        for folder_name in ("1", "2", "3", "4", "5", "6"):
            folder = root / folder_name
            if not folder.is_dir():
                continue
            jpgs = sorted(folder.glob("*.jpg"))
            to_keep = [p for i, p in enumerate(jpgs) if i % 5 == 0]
            to_del = [p for i, p in enumerate(jpgs) if i % 5 != 0]
            total_keep += len(to_keep)
            total_del += len(to_del)
            print(f"[{folder_name}] 保留 {len(to_keep)} 张, 删除 {len(to_del)} 张")
            for p in to_del[:5]:
                print(f"  将删除: {p.name}")
            if len(to_del) > 5:
                print(f"  ... 共 {len(to_del)} 个文件")
        print(f"合计: 保留 {total_keep}, 删除 {total_del} (未执行删除)")
        return 0

    kept, deleted = downsample_one_in_five(root)
    print(f"完成: 保留 {kept} 张, 删除 {deleted} 张")
    return 0


if __name__ == "__main__":
    exit(main())
