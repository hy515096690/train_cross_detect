from pathlib import Path

from ultralytics import YOLO

# 当前项目根目录，训练结果将保存到 项目根/runs/segment/
PROJECT_RUNS = Path(__file__).resolve().parent.parent / "runs" / "segment"


def main():
    # Load a model（yolo26x-seg.pt 为 x 尺度分割模型）
    model = YOLO("yolo26-seg.yaml").load("./pre_model/yolo26x-seg.pt")

    # Train the model
    results = model.train(
        data="train-seg.yaml",
        epochs=100,
        imgsz=640,
        batch=-1,
        lr0=0.01,
        warmup_epochs=3,
        pretrained=True,
        patience=50,
        save=True,
        project=str(PROJECT_RUNS),  # 结果保存到当前项目 runs/segment/，不依赖全局 settings
    )
    return results


if __name__ == "__main__":
    main()