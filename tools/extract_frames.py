"""
从 /videos 文件夹读取 1-7 的 MP4 视频，原分辨率截取视频帧保存到 /frames，
每三帧保存一张图像。
"""

import cv2
from pathlib import Path

# 路径配置（相对于脚本所在目录）
SCRIPT_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = SCRIPT_DIR / "videos"
FRAMES_DIR = SCRIPT_DIR / "frames"

# 每 N 帧保存一张
SAVE_EVERY_N_FRAMES = 3


def extract_frames_from_video(video_path: Path, frames_dir: Path) -> int:
    """
    从单个视频中每 N 帧保存一张图像（原分辨率）。

    Args:
        video_path: 视频文件路径
        frames_dir: 帧保存目录

    Returns:
        保存的图像数量
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  警告：无法打开 {video_path.name}，跳过")
        return 0

    video_id = video_path.stem  # 如 "1", "2", ...
    saved_count = 0
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % SAVE_EVERY_N_FRAMES == 0:
            # 原分辨率保存，文件名如 1_000000.jpg
            out_name = f"{video_id}_{saved_count:06d}.jpg"
            out_path = frames_dir / out_name
            cv2.imwrite(str(out_path), frame)
            saved_count += 1

        frame_index += 1

    cap.release()
    return saved_count


def main() -> None:
    frames_dir = FRAMES_DIR
    frames_dir.mkdir(parents=True, exist_ok=True)

    if not VIDEOS_DIR.is_dir():
        print(f"错误：视频目录不存在 '{VIDEOS_DIR}'")
        return

    total_saved = 0
    for i in range(1, 7):
        video_name = f"{i}.mp4"
        video_path = VIDEOS_DIR / video_name
        if not video_path.exists():
            print(f"跳过：未找到 {video_name}")
            continue

        print(f"处理 {video_name} ...")
        n = extract_frames_from_video(video_path, frames_dir)
        total_saved += n
        print(f"  保存 {n} 张图像")

    print(f"\n完成。共保存 {total_saved} 张图像到 {frames_dir}")


if __name__ == "__main__":
    main()
