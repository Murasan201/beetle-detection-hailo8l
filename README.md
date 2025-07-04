# Insect Detection Training Project

A YOLOv8-based insect detection training project for developing custom beetle detection models and CPU-optimized inference applications.

## 📋 Project Overview

This project evaluates the capability of a YOLO model to detect insects in still images and visualize the results. It's designed to run efficiently on CPU environments, specifically targeting WSL2 on Windows systems as a test environment before potential deployment on Raspberry Pi devices.

## 🎯 Features

- **Batch Image Processing**: Process multiple images in a specified directory
- **CPU-Optimized Inference**: Runs efficiently on CPU without GPU requirements
- **Result Visualization**: Draws bounding boxes around detected insects
- **Comprehensive Logging**: CSV format logging with processing time metrics
- **Command-Line Interface**: Simple CLI for easy operation
- **Multiple Format Support**: Handles JPEG and PNG input images

## 🛠️ Technical Specifications

### Runtime Environment
- **Test Environment**: WSL2 on Windows 10/11 (Ubuntu 22.04 recommended)
- **Hardware**: Host PC CPU (minimum quad-core recommended)
- **Accelerator**: CPU-only inference (no GPU required)

### Software Requirements
- **Python**: 3.9+ (tested with 3.10.12)
- **Deep Learning Framework**: Ultralytics YOLOv8 (CPU mode)
- **Key Libraries**: OpenCV, NumPy, PyTorch (CPU build)

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Murasan201/insect-detection-training.git
cd insect-detection-training
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Quick Start with Pre-trained Model

For readers who want to immediately test the beetle detection:

1. **Download the pre-trained model** from Hugging Face:
   - Visit: `https://huggingface.co/[your-username]/beetle-detection-yolov8`
   - Download `best.pt` to the `weights/` directory

2. **Run detection**:
```bash
python detect_insect.py --input input_images/ --output output_images/
```

### Complete Training Pipeline

For readers who want to understand the full training process:

```bash
# Open the training notebook
jupyter notebook yolov8_training_colab.ipynb
```

### Command Line Arguments
- `--input`: Input directory containing images to process
- `--output`: Output directory for processed images with bounding boxes
- `--help`: Display usage information
- `--model`: (Optional) Specify custom model weights path

### Directory Structure
```
insect-detection-training/
├── detect_insect.py          # Main detection script
├── requirements.txt          # Python dependencies
├── datasets/                 # Training datasets (excluded from Git)
├── input_images/            # Input directory (create manually)
├── output_images/           # Output directory (auto-created)
├── logs/                    # Log files (auto-created)
├── weights/                 # Model weights (auto-downloaded)
└── training_results/        # Training logs and metrics
```

## 📊 Performance Metrics

### Target Performance
- **Processing Time**: ≤ 1,000ms per image (CPU environment)
- **Memory Usage**: Efficient handling of large image batches
- **Stability**: Process 50+ consecutive images without crashes

### Expected Accuracy
- **True Positive Rate**: ≥ 80%
- **False Positive Rate**: ≤ 10%
- **Test Coverage**: Validated on ≥ 20 sample images

## 📝 Output Format

### Processed Images
- **Format**: PNG (regardless of input format)
- **Resolution**: Maintains original image resolution
- **Visualization**: Bounding boxes with confidence scores

### Log Files
- **Format**: CSV with columns: `filename, detected, count, time_ms`
- **Location**: `logs/` directory with timestamp in filename
- **Console Output**: Real-time processing information

## 🔧 Development

### Code Style
- Follows PEP 8 guidelines
- Maximum line length: 88 characters (Black formatter)
- Comprehensive docstrings for all functions

### Testing
```bash
# Run tests (when available)
pytest

# Code formatting
black .

# Linting
flake8 .
```

## 🏗️ Project Structure

The project follows a clean, modular structure with clear separation of concerns:

- **Main Script**: `detect_insect.py` - Core detection logic
- **Configuration**: Environment variables and model parameters
- **Logging**: Structured logging with CSV output
- **Error Handling**: Graceful handling of individual file failures

## 📚 For Book Readers

This project serves as a companion to **[Your Book Title]** - Chapter X: Insect Detection with YOLOv8.

### Learning Paths

#### 🚀 **Quick Start** (5 minutes)
Perfect for readers who want to see results immediately:
1. Clone this repository
2. Download pre-trained model from Hugging Face
3. Run detection on sample images
4. See the results!

#### 📖 **Full Learning** (2-3 hours)  
For readers who want to understand the complete process:
1. Read the theory in the book chapter
2. Follow the Jupyter notebook step-by-step
3. Train your own model with the dataset
4. Compare results with the pre-trained model

#### 🔬 **Advanced Exploration** (1-2 days)
For readers ready to experiment:
1. Try different YOLOv8 model sizes (nano → small → medium)
2. Experiment with your own insect images
3. Fine-tune hyperparameters
4. Deploy on Raspberry Pi (Chapter X+1)

### Model Access

- **Pre-trained Model**: [Hugging Face Repository](https://huggingface.co/[your-username]/beetle-detection-yolov8)
- **Training Code**: This GitHub repository
- **Dataset**: Instructions provided in the book

## 🔒 Security

This project implements security best practices:
- No sensitive information committed to version control
- Comprehensive `.gitignore` for security-sensitive files
- Environment variable usage for configuration
- Regular security auditing guidelines

## 📊 Dataset Information

### Training Dataset
This project uses the beetle detection dataset from Roboflow for custom YOLOv8 model training:

- **Source**: [Roboflow Beetle Dataset](https://universe.roboflow.com/z-algae-bilby/beetle)
- **Total Images**: 500 annotated images
- **Classes**: 1 class ('beetle' - general beetle detection)
- **Format**: YOLO format with bounding box annotations
- **License**: CC BY 4.0
- **Split**: 
  - Training: 400 images (80%)
  - Validation: 50 images (10%)
  - Test: 50 images (10%)

### Dataset Setup
**Note**: Dataset files are excluded from Git due to large size and license considerations.

1. **Download Dataset**: Access the Roboflow dataset link above
2. **Extract to datasets/**: Place the downloaded files in the `datasets/` directory
3. **Verify Structure**: Ensure the following structure:
   ```
   datasets/
   ├── train/
   │   ├── images/    # 400 training images
   │   └── labels/    # 400 training labels
   ├── valid/
   │   ├── images/    # 50 validation images
   │   └── labels/    # 50 validation labels
   ├── test/
   │   ├── images/    # 50 test images
   │   └── labels/    # 50 test labels
   └── data.yaml      # YOLOv8 configuration
   ```

## 🚧 Current Status

This is a training project designed to develop custom YOLOv8 models for insect detection and evaluate their capabilities on CPU hardware. The project is currently in development and testing phase.

### Future Enhancements
- Complete custom model training pipeline
- Model performance evaluation and optimization
- Raspberry Pi 5 deployment optimization
- Real-time video processing support
- Web interface for easier operation

## 📄 License

### Code License
This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) - see the [LICENSE](LICENSE) file for details.

### Dataset Attribution
This model was trained using the Beetle Dataset by z Algae Bilby, licensed under CC BY 4.0.
- **Dataset**: [Roboflow Beetle Dataset](https://universe.roboflow.com/z-algae-bilby/beetle/dataset/1)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Creator**: z Algae Bilby
- **Modifications**: Used for training YOLOv8 object detection model

### Model License
The trained model (`best.pt`) is derived from YOLOv8 (Ultralytics) and inherits the AGPL-3.0 license.

### Third-Party Components
- **YOLOv8**: Ultralytics - AGPL-3.0 License
- **Training Dataset**: z Algae Bilby - CC BY 4.0 License

## 👨‍💻 Developer

**Developed by**: Murasan201

For more information about the developer and other projects, please visit: **https://murasan-net.com/**

## 🤝 Contributing

Contributions are welcome! Please read the project guidelines in `CLAUDE.md` before contributing.

## 📞 Support

For issues, questions, or contributions, please open an issue on GitHub.

---

**Note**: This project is optimized for CPU-only inference and is specifically designed for testing on WSL2 environments before potential Raspberry Pi deployment.