# HybrIK - 3D Human Pose and Shape Estimation

![HybrIK Logo](assets/hybrik.png)

**Modified by:** [xsiadron](https://xsiadron.com)

A powerful 3D human pose and shape estimation framework with enhanced Blender integration support.

## 📋 Overview

This is a modified version of HybrIK with enhanced features and updated Blender addon compatibility. The project provides accurate 3D human pose estimation and SMPL/SMPL-X body model fitting from single images or videos.

### ✨ Key Features

-   **3D Human Pose Estimation**: High-quality pose estimation from monocular images/videos
-   **SMPL/SMPL-X Integration**: Full body shape and pose reconstruction
-   **Blender Addon**: Updated addon with Blender 4.4+ compatibility
-   **Batch Processing**: Process multiple videos efficiently
-   **Real-time Inference**: Fast processing for practical applications

## 🔗 Original Project

For comprehensive documentation and research details, visit the original repository:
**[HybrIK Official Repository](https://github.com/jeffffffli/HybrIK)**

## 🛠️ Installation

### Prerequisites

-   Python 3.8+
-   CUDA-compatible GPU (recommended)
-   Conda package manager

### Step-by-Step Installation

#### 1. Create Virtual Environment

```bash
# Create new conda environment (if not already created)
conda create -n hybrik python=3.8 -y
conda activate hybrik
```

#### 2. Install PyTorch

```bash
conda install pytorch==1.9.1 torchvision==0.10.1 -c pytorch
```

#### 3. Install PyTorch3D (Optional - for visualization)

```bash
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
pip install git+ssh://git@github.com/facebookresearch/pytorch3d.git@stable
```

#### 4. Install Dependencies

```bash
pip install pycocotools
python setup.py develop  # or "pip install -e ."
```

#### 5. Initialize Setup

```bash
python init.py
```

## 🚀 Quick Start

### Batch Video Processing

1. **Place your input videos** in the `input/` folder
2. **Run batch processing**:
    ```bash
    python batch_process.py
    ```
3. **Find results** in the `output/` folder with generated `.pkl` files

### Single Video Processing

```bash
python process_new_video.py --input path/to/your/video.mp4
```

## 📁 Project Structure

```
HybrIK/
├── input/              # Place input videos here
├── output/             # Processed results
├── hybrik_blender_addon/  # Blender 4.4+ compatible addon
├── pretrained_models/  # Pre-trained model weights
├── scripts/           # Demo and training scripts
└── hybrik/           # Core framework code
```

## 🎨 Blender Integration

The updated Blender addon supports Blender 4.4+ and provides seamless integration for 3D pose visualization and animation workflows.

### Installation

1. Extract `hybrik_blender_addon.zip`
2. Install in Blender via Preferences → Add-ons → Install

## 📊 Model Files

Required model files are automatically downloaded during initialization. Manual download may be needed for:

-   SMPL/SMPL-X base models
-   Pre-trained HybrIK weights
-   Additional pose regressors

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project maintains the same license as the original HybrIK repository. See `LICENSE` file for details.

## 🙏 Acknowledgments

-   Original HybrIK team for the excellent framework
-   SMPL/SMPL-X creators for body modeling
-   PyTorch and PyTorch3D teams

---

**Note**: This is a community-maintained fork with enhanced Blender compatibility and batch processing features.
