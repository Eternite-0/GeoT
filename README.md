# GeoTrack-GS


本项目利用 3D 高斯溅射（3D Gaussian Splatting）技术，实现先进的 3D 场景重建与渲染。该项目基于论文《3D Gaussian Splatting for Real-Time Radiance Field Rendering》。此实现支持场景的训练和渲染，并利用 PyTorch 和自定义 CUDA 核心以实现高性能。

## 主要功能 (Features)

*   **高质量实时渲染**: 基于 3D Gaussian Splatting，实现照片级的实时渲染效果。
*   **端到端工作流**: 支持从 COLMAP 数据集直接进行训练、渲染和评估。
*   **置信度评估**: 集成了置信度信息，提升渲染的稳定性和准确性。
*   **PyTorch 核心**: 完全使用 PyTorch 构建，易于理解、修改和扩展。
*   **高性能 CUDA 加速**: 关键的渲染管线使用自定义 CUDA 核心实现，保证了渲染效率。

## 先决条件 (Prerequisites)

在开始之前，请确保你的系统满足以下要求：
*   **操作系统**: Linux (推荐) 或 Windows
*   **GPU**: 支持 CUDA 11.8 或更高版本的 NVIDIA GPU
*   **软件**:
    *   Anaconda 或 Miniconda
    *   Git

## 安装 (Installation)

1.  **克隆仓库：**

    ```bash
    # 注意：--recursive 参数是必需的，用于克隆所有子模块
    git clone --recursive https://github.com/CPy255/GeoTrack-GS.git
    cd GeoTrack-GS
    ```

2.  **创建并激活 Conda 环境：**

    ```bash
    conda env create -f environment.yml
    conda activate geotrack
    ```

3.  **构建自定义 CUDA 子模块：**

    本项目需要自定义的 CUDA 核心。请运行以下命令来构建和安装它们：

    ```bash
    cd submodules/diff-gaussian-rasterization-confidence
    python setup.py install
    cd ../../submodules/simple-knn
    python setup.py install
    cd ../..
    ```

## 使用方法 (Usage)

### 1. 准备数据

你需要一个由 COLMAP 处理过的数据集。目录结构通常应包含一个 `images` 文件夹和一个带有 COLMAP 重建结果的 `sparse` 文件夹。

为了方便快速开始，你可以下载官方提供的示例数据集：
```bash
# (可选) 下载示例数据到 data/ 目录
git clone https://github.com/graphdeco-inria/gaussian-splatting-data.git data
```
*注意：该数据集较大，下载可能需要一些时间。*

### 2. 训练

使用 `train.py` 脚本来训练一个新模型。

```bash
# 示例：使用下载的数据集中的 "tandt" 场景进行训练
python train.py -s data/tandt/train -m output/tandt
```

*   `-s, --source_path`: 输入数据集所在的目录路径。
*   `-m, --model_path`: 用于保存训练好的模型和检查点的目录路径。

要查看所有可用的训练选项，请运行：
```bash
python train.py --help
```

### 3. 渲染

当您拥有一个训练好的模型后，就可以从新的摄像机视角渲染图像。

```bash
python render.py -m output/tandt
```

### 4. 评估

要评估一个训练好的模型，请使用 `full_eval.py` 脚本：

```bash
python full_eval.py -m output/tandt
```

## 如何贡献 (Contributing)

欢迎任何形式的贡献！如果你发现了 bug 或有功能建议，请随时通过 [GitHub Issues](https://github.com/CPy255/GeoTrack-GS/issues) 提出。

## 致谢 (Acknowledgements)

本项目基于以下出色的工作：
*   [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
*   [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)

感谢原作者的杰出贡献。

## 许可证 (License)

本项目根据 [LICENSE.md](LICENSE.md) 文件中的条款进行许可。
