"""
使用训练好的分割权重对视频做推理（model.predict）；
对类别 train（角标 2）在分割 mask 区域内用光流（像素级运动）判断左/右/停止。
YOLO 在 GPU 上推理；光流优先用 OpenCV CUDA（若可用），否则在降分辨率图上算以减轻 CPU 负担。
"""
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.plotting import Annotator, colors

# 训练好的分割权重（请按实际最新 run 修改）
WEIGHTS = Path(__file__).resolve().parent.parent / "runs" / "segment" / "train2" / "weights" / "best.pt"
# 待检测视频
SOURCE = Path(__file__).resolve().parent.parent / "videos" / "1.mp4"
# 结果视频与可视化输出目录
SAVE_DIR = Path(r"D:\PYTHON_PROJECT\train_cross_detect\pre_result")

# 类别 train 的角标（与 train-seg.yaml 中 names 一致）
TRAIN_CLS_ID = 2

# ---------- 光流“是否在动”的判定阈值 ----------
# 水平光流（像素/帧）超过此值判为 右移动，低于负值判为 左移动，否则 停止。
# 调小 → 更敏感，轻微位移即判为移动；调大 → 更迟钝，只有明显位移才判为移动。
# FLOW_THRESHOLD_PX = 0.2
FLOW_THRESHOLD_PX = 0.1

# 光流在 CPU 上计算时的最大宽度（降低分辨率以提速），若用 CUDA 则按原图
FLOW_MAX_WIDTH_CPU = 320

# ---------- Farneback 光流算法参数（影响精度与速度） ----------
# pyr_scale: 金字塔相邻层缩放比（0~1）。越小金字塔层间变化越细，对小位移更敏感，但更耗时。
FLOW_PYR_SCALE = 0.5
# levels: 金字塔层数。越多越能捕捉大位移，但小位移在高层可能被平滑掉；2~3 对小位移较敏感。
FLOW_LEVELS = 2
# winsize: 每像素邻域窗口边长（奇数）。越小对局部小运动越敏感，但噪声更敏感；越大更平滑、更稳。
# FLOW_WINSIZE = 15
FLOW_WINSIZE = 11
# iterations: 每层迭代次数。越多光流越精细，略更敏感，但更慢。
FLOW_ITERATIONS = 2
# 若要“轻微变动就判为移动”：可把 FLOW_THRESHOLD_PX 再调小（如 0.1），或把 FLOW_WINSIZE 调小（如 11）。

# 每 N 帧才计算一次光流，中间帧复用上一段光流结果，可明显降低 CPU（光流在 CPU 上时有效）
FLOW_EVERY_N_FRAMES = 1  # 设为 2 则每 2 帧算一次光流，CPU 约减半

# 是否尝试使用 OpenCV CUDA 光流（需安装带 CUDA 的 OpenCV，如 opencv-contrib-python 且编译了 CUDA）
_USE_CUDA_FLOW = getattr(cv2, "cuda", None) is not None
if _USE_CUDA_FLOW:
    try:
        _ = cv2.cuda.FarnebackOpticalFlow.create()
    except Exception:
        _USE_CUDA_FLOW = False


def _get_mask_at_orig_shape(mask_tensor, target_h: int, target_w: int) -> np.ndarray:
    """将 mask 转为 numpy 并缩放到 (target_h, target_w)。"""
    if isinstance(mask_tensor, torch.Tensor):
        mask = mask_tensor.cpu().numpy()
    else:
        mask = np.asarray(mask_tensor)
    if mask.ndim > 2:
        mask = mask.squeeze()
    if mask.shape[0] != target_h or mask.shape[1] != target_w:
        mask = cv2.resize(mask.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return mask


def _flow_x_in_mask(flow: np.ndarray, mask: np.ndarray, scale: float = 1.0, use_median: bool = True) -> float:
    """
    在 mask 区域内取水平光流中位数；scale 将光流从计算分辨率换算到原图像素/帧。
    """
    h, w = flow.shape[:2]
    if mask.shape[0] != h or mask.shape[1] != w:
        mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    valid = (mask > 0.5).flatten()
    flow_x = flow[:, :, 0].flatten()
    flow_x = flow_x[valid]
    if flow_x.size == 0:
        return 0.0
    raw = float(np.median(flow_x) if use_median else np.mean(flow_x))
    return raw * scale


def _compute_flow(prev_gray: np.ndarray, curr_gray: np.ndarray, flow_size: tuple | None) -> tuple[np.ndarray | None, float]:
    """
    计算 prev -> curr 的光流。若 flow_size 不为 None，在 (w,h)=flow_size 上计算（CPU 降分辨率）；
    否则先尝试 CUDA 全图，失败则在降分辨率上做 CPU。返回 (flow, scale)，scale 将 flow 换算到原图。
    """
    H, W = curr_gray.shape[:2]
    if flow_size is not None:
        w_small, h_small = flow_size
        scale = W / w_small
        prev = cv2.resize(prev_gray, (w_small, h_small), interpolation=cv2.INTER_LINEAR)
        curr = cv2.resize(curr_gray, (w_small, h_small), interpolation=cv2.INTER_LINEAR)
    else:
        scale = 1.0
        prev, curr = prev_gray, curr_gray

    if _USE_CUDA_FLOW and flow_size is None:
        try:
            gpu_prev = cv2.cuda_GpuMat()
            gpu_curr = cv2.cuda_GpuMat()
            gpu_prev.upload(prev)
            gpu_curr.upload(curr)
            farn = cv2.cuda.FarnebackOpticalFlow.create()
            gpu_flow = farn.calc(gpu_prev, gpu_curr)
            flow = gpu_flow.download()
            return flow, scale
        except Exception:
            pass
        w_s = min(FLOW_MAX_WIDTH_CPU, W)
        h_s = int(H * w_s / W)
        prev = cv2.resize(prev_gray, (w_s, h_s), interpolation=cv2.INTER_LINEAR)
        curr = cv2.resize(curr_gray, (w_s, h_s), interpolation=cv2.INTER_LINEAR)
        scale = W / w_s

    flow = cv2.calcOpticalFlowFarneback(
        prev, curr, None,
        pyr_scale=FLOW_PYR_SCALE,
        levels=FLOW_LEVELS,
        winsize=FLOW_WINSIZE,
        iterations=FLOW_ITERATIONS,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    return flow, scale


def _direction_label(flow_x: float, conf: float) -> str:
    """根据 mask 内光流水平分量 flow_x 返回：train + move to the left/move to the right/stop + 置信度。"""
    if abs(flow_x) < FLOW_THRESHOLD_PX:
        move = "stop"
    elif flow_x > 0:
        move = "move to the right"
    else:
        move = "move to the left"
    return f"train {move} {conf:.2f}"


def _reuse_flow_x(current_centers: list, last_train_flow_x: dict) -> dict:
    """用上一段光流结果按中心最近邻匹配到当前帧的 train，返回 (cx,cy)->flow_x。"""
    out = {}
    if not last_train_flow_x:
        return out
    keys = list(last_train_flow_x.keys())
    for (cx, cy) in current_centers:
        i = min(range(len(keys)), key=lambda j: (cx - keys[j][0]) ** 2 + (cy - keys[j][1]) ** 2)
        out[(cx, cy)] = last_train_flow_x[keys[i]]
    return out


def _annotate_frame(result, prev_gray, names: dict, do_flow: bool, last_train_flow_x: dict):
    """
    用当前帧 result 与上一帧灰度图 prev_gray（当 do_flow 为 True 时）算光流，在 train 的 mask 内取 flow_x；
    否则复用 last_train_flow_x。返回 (annotated_bgr, 本帧灰度图, 本帧 train_flow_x)。
    """
    img = deepcopy(result.orig_img)
    if isinstance(img, torch.Tensor):
        img = (img[0].detach().permute(1, 2, 0).contiguous() * 255).byte().cpu().numpy()

    H, W = img.shape[:2]
    curr_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    flow = None
    flow_scale = 1.0
    if do_flow and prev_gray is not None and prev_gray.shape == curr_gray.shape:
        if _USE_CUDA_FLOW:
            flow_size = None
        else:
            w_s = min(FLOW_MAX_WIDTH_CPU, W)
            h_s = int(H * w_s / W)
            flow_size = (w_s, h_s)
        flow, flow_scale = _compute_flow(prev_gray, curr_gray, flow_size)

    annotator = Annotator(img, line_width=None, font_size=None, font="Arial.ttf", pil=False, example=names)
    pred_boxes = result.boxes
    pred_masks = result.masks

    all_boxes = []
    train_flow_x = {}
    train_centers = []

    if pred_boxes is not None:
        for i, d in enumerate(pred_boxes):
            c = int(d.cls)
            conf = float(d.conf)
            box = d.xyxy.squeeze()
            cx = (box[0].item() + box[2].item()) / 2.0 if isinstance(box, torch.Tensor) else (float(box[0]) + float(box[2])) / 2
            cy = (box[1].item() + box[3].item()) / 2.0 if isinstance(box, torch.Tensor) else (float(box[1]) + float(box[3])) / 2
            all_boxes.append((c, conf, box, cx, cy))
            if c == TRAIN_CLS_ID:
                train_centers.append((cx, cy))

            if c == TRAIN_CLS_ID and pred_masks is not None and i < pred_masks.data.shape[0] and flow is not None:
                flow_h, flow_w = flow.shape[:2]
                mask_np = _get_mask_at_orig_shape(pred_masks.data[i], H, W)
                mask_flow = cv2.resize(mask_np.astype(np.float32), (flow_w, flow_h), interpolation=cv2.INTER_LINEAR)
                flow_x = _flow_x_in_mask(flow, mask_flow, scale=flow_scale, use_median=True)
                train_flow_x[(cx, cy)] = flow_x

        if not train_flow_x and train_centers and last_train_flow_x:
            train_flow_x = _reuse_flow_x(train_centers, last_train_flow_x)

    # 1) 画分割 mask
    if pred_masks is not None and pred_masks.data.shape[0] > 0:
        img_letterbox = LetterBox(pred_masks.shape[1:])(image=annotator.result())
        im_gpu = (
            torch.as_tensor(img_letterbox, dtype=torch.float16, device=pred_masks.data.device)
            .permute(2, 0, 1)
            .flip(0)
            .contiguous()
            / 255
        )
        idx = (
            pred_boxes.cls
            if pred_boxes is not None
            else reversed(range(len(pred_masks)))
        )
        if isinstance(idx, torch.Tensor):
            idx = idx.cpu().tolist()
        annotator.masks(pred_masks.data, colors=[colors(x, True) for x in idx], im_gpu=im_gpu)

    # 2) 画框与标签（train 用 mask 内光流，其余用 类别+置信度）
    for c, conf, box, cx, cy in reversed(all_boxes):
        if c == TRAIN_CLS_ID:
            flow_x = train_flow_x.get((cx, cy), 0.0)
            label = _direction_label(flow_x, conf)
        else:
            label = f"{names.get(c, c)} {conf:.2f}"

        color = colors(c, True)
        box_list = box.tolist() if isinstance(box, torch.Tensor) else box
        annotator.box_label(box_list, label, color=color)

    return annotator.result(), curr_gray, train_flow_x


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(SOURCE))
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {SOURCE}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    model = YOLO(str(WEIGHTS))
    # 加载后模型默认在 CPU，需显式移到 GPU；predict(device=...) 只影响 predictor 的 device，不会把已加载的 model 移过去
    if torch.cuda.is_available():
        model.to("cuda:0")
        if hasattr(model, "model") and isinstance(model.model, torch.nn.Module):
            model.model.to("cuda:0")
    try:
        dev = next(model.model.parameters()).device
        print(f"[设备] YOLO 推理: {dev} (cuda 表示在 GPU 上)")
    except Exception:
        print("[设备] YOLO 推理: 已设置 device，推理应在 GPU 上")
    print(f"[光流] 后端: {'OpenCV CUDA (GPU)' if _USE_CUDA_FLOW else 'OpenCV CPU (降分辨率)'} | 每 {FLOW_EVERY_N_FRAMES} 帧计算一次")

    out_path = SAVE_DIR / SOURCE.name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None
    prev_gray = None
    last_train_flow_x = {}
    frame_index = 0

    for result in model.predict(
        source=str(SOURCE),
        stream=True,
        save=False,
        show=False,
        conf=0.60,
        # device="cuda:0",
        classes=[2],
    ):
        result = result[0] if isinstance(result, (list, tuple)) else result
        names = result.names or {}
        do_flow = (frame_index % FLOW_EVERY_N_FRAMES) == 0
        frame, prev_gray, last_train_flow_x = _annotate_frame(result, prev_gray, names, do_flow, last_train_flow_x)
        frame_index += 1

        if writer is None:
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (frame.shape[1], frame.shape[0]))

        writer.write(frame)
        cv2.imshow("segment + train motion (optical flow)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print(f"结果已保存到: {out_path}")
    return out_path


if __name__ == "__main__":
    main()
