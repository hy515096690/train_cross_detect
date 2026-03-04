from ultralytics import YOLO

# Load a model
model = YOLO("yolo26x-seg.yaml").load("./pre_model/yolo26x.pt")  # 使用预训练模型

# Train the model
# 针对自定义数据集的迁移学习训练参数（重点调整）
results = model.train(
    data="数据集配置文件.yaml",  # 替换为你自己的数据集yaml
    epochs=150,                      # 迁移学习150轮足够（比从头训练少一半）
    imgsz=640,                       # 匹配你数据集的图片尺寸
    batch=-1,                        # 自动适配GPU显存的批次大小
    lr0=0.01,                        # 初始学习率（默认即可，迁移学习不用调）
    warmup_epochs=3,                 # 短热身即可（预训练权重已稳定）
    pretrained=True,                 # 显式指定迁移学习（默认开启）
    # freeze=10,                       # 可选：冻结前10层骨干网络，先微调头部（小数据集推荐）
    patience=50,                     # 早停策略：50轮没提升就停止，避免过拟合
    save=True                        # 保存最优模型
)