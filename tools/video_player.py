"""
使用 OpenCV 加载 MP4 视频并逐帧显示。
按 'q' 退出，按空格键暂停/继续。
"""

import cv2
import sys


def play_video(video_path: str) -> None:
    """
    加载并逐帧显示 MP4 视频。

    Args:
        video_path: 视频文件路径（支持 .mp4 等格式）
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"错误：无法打开视频文件 '{video_path}'")
        sys.exit(1)

    # 获取视频基本信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"视频信息: {width}x{height}, {fps:.1f} FPS, 共 {total_frames} 帧")
    print("按 'q' 退出, 按空格键 暂停/继续")

    paused = False
    frame_index = 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("视频播放完毕或读取失败")
                break
            frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        else:
            # 暂停时保持显示当前帧
            pass

        # 在画面上显示帧序号
        display_frame = frame.copy()
        cv2.putText(
            display_frame,
            f"Frame: {frame_index}/{total_frames}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        if paused:
            cv2.putText(
                display_frame,
                "PAUSED",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Video Player", display_frame)

        # 根据是否暂停决定等待时间
        if paused:
            key = cv2.waitKey(0) & 0xFF
        else:
            delay = max(1, int(1000 / fps)) if fps > 0 else 33
            key = cv2.waitKey(delay) & 0xFF

        if key == ord("q"):
            print("用户退出")
            break
        elif key == ord(" "):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("用法: python video_player.py <视频文件路径>")
    #     print("示例: python video_player.py video.mp4")
    #     sys.exit(1)

    video_path = r'./video/1.mp4'
    play_video(video_path)
