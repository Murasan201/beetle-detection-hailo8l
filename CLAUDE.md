# Project Rules and Guidelines

## Project Requirements

### Requirements Specification
- **Project requirements are defined in**: `insect_detection_training_project_requirements_spec.md`
- **MUST review this document before starting any work**
- Contains detailed functional and non-functional requirements
- Provides context for all development decisions

## Project Structure

```
insect-detection-cpu-test/
├── detect_insect.py          # Main detection script (CPU inference)
├── hailo_beetle_detection.py # Production NPU inference script (Raspberry Pi 5)
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
├── README.md                # Project documentation
├── RULES.md                 # This file
├── input_images/            # Input directory (not tracked)
├── output_images/           # Output directory (not tracked)
├── logs/                    # Log files (not tracked)
└── weights/                 # Model weights (not tracked)
```

## Coding Standards

### Python Code Style
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Include docstrings for all functions and classes
- Maximum line length: 88 characters (Black formatter)

### File Naming
- Use snake_case for Python files
- Use descriptive names that indicate purpose
- Avoid abbreviations unless commonly understood

## Performance Requirements

- Processing time per image: ≤ 1,000ms (CPU environment)
- Memory usage: Efficient handling of large image batches
- Error handling: Continue processing on individual file failures

## Logging Standards

### Log Format
- CSV format: `filename, detected, count, time_ms`
- Include timestamp in log filename
- Log both to console and file

### Log Levels
- INFO: Normal processing information
- WARNING: Non-critical issues
- ERROR: Processing failures that don't stop execution
- CRITICAL: Fatal errors that stop execution

## Testing Requirements

### Accuracy Metrics
- True positive rate: ≥ 80%
- False positive rate: ≤ 10%
- Test with ≥ 20 sample images

### Stability Testing
- Must process 50 consecutive images without crashes
- Handle various image formats (JPEG, PNG)
- Handle various image resolutions

## Dependencies

### Required Libraries
- Python 3.9+
- PyTorch (CPU version)
- Ultralytics YOLOv8
- OpenCV
- NumPy

### Installation
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### System Environment Information
- **Python Version**: 3.10.12 (System Level)
- **Pip Version**: 22.0.2
- **Installation Type**: User-level packages (pip install --user)
- **Package Location**: `/home/win/.local/lib/python3.10/site-packages/`
- **System**: Linux WSL2 (Ubuntu)
- **Architecture**: x86_64
- **Last Updated**: 2025-07-04

### Installed Key Packages
- **torch**: 2.7.1 (Deep Learning Framework)
- **torchvision**: 0.22.1 (Computer Vision)
- **ultralytics**: 8.3.162 (YOLOv8 Implementation)
- **opencv-python**: 4.11.0.86 (Computer Vision)
- **numpy**: 2.2.6 (Numerical Computing)
- **pandas**: 2.3.0 (Data Analysis)

## Usage Guidelines

### Command Line Interface
```bash
python detect_insect.py --input input_images/ --output output_images/
```

### Required Arguments
- `--input`: Input directory containing images
- `--output`: Output directory for processed images

### Optional Arguments
- `--help`: Display usage information
- `--model`: Specify custom model weights path

## File Handling Rules

### Input Files
- Support JPEG and PNG formats
- Process all valid images in input directory
- Skip invalid or corrupted files with warning

### Output Files
- Save as PNG format regardless of input format
- Maintain original resolution
- Use same filename as input with .png extension

## Error Handling

### Exception Management
- Catch and log exceptions for individual files
- Continue processing remaining files
- Provide meaningful error messages
- Exit gracefully on critical errors

### Resource Management
- Close file handles properly
- Clean up temporary resources
- Handle memory efficiently for large batches

## Version Control

### Git Workflow
- Use meaningful commit messages
- Don't commit large files (images, models, datasets)
- Keep repository clean and organized

### Dataset Management
- **NEVER commit dataset files to GitHub**
- Datasets are excluded via .gitignore due to:
  - Large file sizes (500+ images)
  - License considerations (CC BY 4.0 attribution requirements)
  - Repository efficiency (focus on code, not data)
- Use external storage or download scripts for dataset distribution

### Ignored Files
- **Model weights (*.pt, *.pth, *.onnx)** - Store in Hugging Face instead
- Input/output directories
- Log files
- Temporary files
- Python cache files
- **Dataset files (datasets/, *.jpg, *.png, *.txt, data.yaml)**

### Model File Distribution Policy

**IMPORTANT: Model files must NOT be uploaded to GitHub**

#### Rationale
- **License Compliance**: Trained models inherit AGPL-3.0 from YOLOv8
- **File Size**: Model files (6.3MB+) approach GitHub's recommended limits
- **Distribution Strategy**: Hugging Face Model Hub is optimized for ML models
- **Commercial Safety**: Separation maintains MIT license for codebase

#### Approved Distribution Method
- **GitHub Repository**: Source code, training scripts, documentation (MIT License)
- **Hugging Face Model Hub**: Trained model weights with proper AGPL-3.0 attribution
- **Book Integration**: Programmatic download via `huggingface_hub` library

#### Fine-tuned Model Repository
- **Model Location**: https://huggingface.co/Murasan/beetle-detection-yolov8
- **License**: AGPL-3.0 (inherited from YOLOv8)
- **Available Formats**: PyTorch (.pt), ONNX (.onnx)
- **Performance**: mAP@0.5: 97.63%, mAP@0.5:0.95: 89.56%

#### Prohibited Actions
- ❌ Committing model files (*.pt, *.pth, *.onnx) to GitHub
- ❌ Using Git LFS for model storage
- ❌ Distributing models without proper AGPL-3.0 compliance
- ❌ Mixing model files with MIT-licensed codebase

#### Implementation
```python
# Correct approach - Reference external models
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="Murasan/beetle-detection-yolov8",
    filename="best.pt",
    local_dir="./weights"
)
```

## Documentation

### Code Documentation
- Include module-level docstrings
- Document all public functions
- Explain complex algorithms
- Provide usage examples

### Project Documentation
- Keep README.md updated
- **README.md must be written in English**
- Document installation steps
- Provide usage examples
- Include troubleshooting guide

## Information Search Guidelines

### Web Search Usage
- Use `mcp__gemini-google-search__google_search` when latest information is needed
- Search for current library versions, API changes, or recent documentation
- Use web search when local information is insufficient or outdated
- Verify information from multiple sources when possible

## Hailo 8L NPU Performance Guidelines

### Performance Disclosure Restrictions
- **PROHIBITED**: Publishing specific performance metrics (FPS, processing time, throughput) of Hailo 8L NPU execution
- **PROHIBITED**: Comparative benchmarks between Hailo 8L and other hardware accelerators
- **PROHIBITED**: Detailed power consumption measurements on Hailo 8L hardware

### Allowed Content for Public Documentation
- ✅ General acceleration concepts and benefits
- ✅ Code examples and implementation guides  
- ✅ Configuration files and setup procedures
- ✅ Theoretical performance improvements (without specific numbers)
- ✅ CPU baseline performance (non-Hailo hardware)

### Documentation Guidelines
- Use placeholder values like "XX% faster" or "significant improvement" instead of actual metrics
- Focus on implementation methodology rather than performance results
- Include disclaimer about commercial licensing requirements
- Reference official Hailo documentation for performance specifications

## Hailo SDK Management

### SDK File Location
- **SDK Installation Path**: `/home/win/dev/hailo-sdk/`
- **Required Components**:
  - `hailo_ai_sw_suite_2025-04_docker/` (Docker environment)
  - `hailort-pcie-driver_4.21.0_all.deb` (PCIe driver)
- **Status**: SDK files prepared for installation

### SDK File Restrictions
- **NEVER commit Hailo SDK files to GitHub**
- SDK files are excluded via `.gitignore` for:
  - Commercial license compliance (EULA restrictions)
  - Large file sizes (Docker archives, drivers)
  - Redistribution restrictions
  - Repository efficiency

## Hailo 8L NPU Compilation Project

### Project Status
- **Compilation Plan**: Detailed plan documented in `HAILO_COMPILATION_PLAN.md`
- **Final Status**: **95% Complete** - HEF compilation successfully achieved
- **Key Achievement**: Custom 1-class YOLOv8 model compiled to HEF format for Hailo 8L NPU

### Implementation Results
- **✅ Environment Setup**: Docker-based Hailo AI Software Suite fully configured
- **✅ SDK Integration**: hailomz, hailo, hailortcli commands operational
- **✅ Model Conversion**: PyTorch → ONNX → HEF conversion pipeline established
- **✅ HEF Generation**: `best.hef` (9.3MB) successfully generated
- **✅ Quantization**: INT8 quantization completed with custom NMS configuration
- **✅ Optimization**: 3-context partitioning achieved 38% performance improvement

### Key Technical Achievements
1. **Custom Model Support**: Solved class count mismatch (1 vs 80 classes) using low-level API
2. **Layer Name Mapping**: Identified internal layer names (best/conv41-63) for proper configuration
3. **NMS Configuration**: Created custom 1-class NMS settings for bbox detection
4. **Troubleshooting**: Documented 6 failed approaches and successful resolution path

### Troubleshooting Knowledge Base
- **Problem**: `Cannot infer bbox conv layers automatically` error
- **Root Cause**: Custom 1-class model vs standard 80-class YOLO assumptions
- **Solution**: Manual NMS configuration with correct internal layer names
- **Critical Files**: Custom JSON config + .alls script with proper syntax

### Files Generated
- `best.hef` (9.3MB) - Hailo 8L NPU executable
- `best_optimized.har` (60MB) - INT8 quantized model
- Custom NMS configuration files
- Calibration dataset (64 images, 640x640)

### Configuration Files
- **Successful Configuration**: Working configuration files documented in `SUCCESSFUL_CONFIGURATION_FILES.md`
- **Purpose**: Contains all validated configuration files that successfully compiled the HEF model

### Hailo 8L NPU Inference Scripts

#### Production Deployment Scripts
- **Main Inference Script**: `hailo_beetle_detection.py`
  - **Purpose**: Production-ready real-time beetle detection for Raspberry Pi 5 environment
  - **Target Hardware**: Raspberry Pi 5 + Hailo 8L NPU (AI Kit)
  - **Implementation Role**: Actual NPU inference execution in production environment
  - **Input Support**: USB camera, RPi camera, video files
  - **Features**: 1-class beetle detection, monitoring capabilities, configurable parameters

#### Supporting Libraries
- **GStreamer Utilities**: `hailo_rpi_common.py`
  - Buffer processing, format conversion, FPS calculation
- **Pipeline Management**: `detection_pipeline.py`
  - GStreamer pipeline base class, error handling, callback management

#### Documentation
- **Deployment Guide**: `HAILO_INFERENCE_README.md`
  - Complete installation instructions for Raspberry Pi 5
  - Hardware requirements, software dependencies
  - Usage examples, troubleshooting, performance optimization

#### System Integration
- **HEF File Location**: Default `best.hef` in same directory as scripts
- **Custom Path Support**: `--hef-path` argument for flexible model location
- **Multi-Input Support**: Automatic detection source switching (USB/RPi/file)
- **Hardware Acceleration**: Optimized for NPU acceleration on supported hardware

## Security Guidelines

### Sensitive Information Protection
- **NEVER commit API keys, passwords, or secrets** to version control
- Use environment variables for all sensitive configuration
- Store API keys in `.env` files (which must be in `.gitignore`)
- Use configuration files in `.gitignore` for local settings
- Regularly audit code for accidentally committed secrets

### Files to Never Commit
- API keys (Google, OpenAI, AWS, etc.)
- Database credentials
- Private keys and certificates
- Local configuration files with sensitive data
- `.mcp.json` and similar MCP configuration files
- Any file containing `password`, `secret`, `key`, `token`
- GitHub personal access tokens and authentication credentials
- Email addresses used for GitHub authentication
- Git configuration files containing personal information
- **Hailo調査報告書**: `YOLOv8モデルをHailo-8Lで実行するためのコンパイル手順.pdf`
- 商用ライセンス関連の調査資料・内部文書
- **Hailo SDK files**: All files in `/home/win/dev/hailo-sdk/`

### Security Best Practices
- Review all files before committing with `git status` and `git diff`
- Use `.gitignore` to prevent accidental commits of sensitive files
- Revoke and regenerate any accidentally committed secrets immediately
- Implement pre-commit hooks for sensitive data detection
- Store production secrets in secure secret management systems