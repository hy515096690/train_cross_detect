> [!Note]
>
> ***注意！！！该版本是安装预编译好的gpu版opencv***

# 1、下载gpu版本的opencv

```bash
# 注意cuda与cudnn的版本
https://www.jamesbowley.co.uk/qmd/downloads.html
# 选择OpenCV Python wheels
# https://github.com/cudawarped/opencv-python-cuda-wheels/releases/download/4.13.0.20250811/opencv_contrib_python_rolling-4.13.0.20250812-cp37-abi3-win_amd64.whl
```

# 2、conda安装gpu版本opencv

```bash
# 1. 先卸载已安装的opencv
pip uninstall opencv-python opencv-contrib-python -y

# 2. 安装gpu版opencv
pip install whl的下载路径

# 3. 测试是否安装成功
python -c "import cv2; print('版本:', cv2.__version__)"
python -c "import cv2; print('CUDA设备:', cv2.cuda.getCudaEnabledDeviceCount())"
```

> [!warning]
>
> ***若报错则执行以下步骤***

# 3 、下载检测dll软件

```bash
https://github.com/lucasg/Dependencies/releases
```

# 4、检测缺少依赖

```bash
1. 打开 Dependencies.exe
2. 拖入以下文件：
   E:\Anaconda3\envs\opencv_gpu\Lib\site-packages\cv2\cv2.pyd
   （或同目录下的 .pyd 文件）
3. 查看标红的缺失 DLL
```

# 5、创建 sitecustomize.py

```bash
# python3.8以后不再从系统 PATH 搜索 DLL，必须通过 os.add_dll_directory() 显式指定！
# 创建默认加载环境变量文件，每次启动 Python 时自动执行。
notepad E:\Anaconda3\envs\train_detect\Lib\sitecustomize.py

# 将以下内容复制到文件中
import os
os.add_dll_directory(r"C:\Program Files\NVIDIA\CUDNN\v9.18\bin\13.1\x64")
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin")
```

> [!Tip]
>
> ***我的是cudnn的dll依赖报错，原因是cuda较新，安装路径发生改变***

# 6、较新的cuda的cudnn路径改变

```bash
1、新增cudnn环境变量
2、在系统环境变量-path中添加：
C:\Program Files\NVIDIA\CUDNN\v9.18\bin\13.1\x64
```

