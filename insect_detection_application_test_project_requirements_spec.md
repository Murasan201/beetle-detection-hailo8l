# Requirements Specification Document

## 1. Project Overview

**Name**: Insect Detection Application Test Project (CPU Version)  
**Purpose**: Train a custom YOLOv8 model using insect datasets to detect beetles (specifically Japanese rhinoceros beetles) in still images through fine-tuning and visualize the results.

---

## 2. Scope

- **Input Images**: JPEG/PNG still images in a user-specified directory  
- **Detection Target**: Custom trained classes focusing on beetles (Japanese rhinoceros beetles)  
- **Training Data**: Insect datasets for YOLOv8 fine-tuning  
- **Output**: Images overlaid with bounding boxes where detections occur, saved to an output directory

---

## 3. Runtime Environment

### 3.1 Test Environment (Current Phase)
- **OS**: WSL2 on Windows 10/11 (Ubuntu 22.04 recommended)  
- **Hardware**: Host PC CPU (minimum quad-core recommended)  
- **Accelerator**: None (CPU-only inference)

### 3.2 Training Environment (Cloud-based)
- **Platform**: Google Colaboratory  
- **GPU**: T4 or V100 (free tier available)  
- **Runtime**: Python 3.10+ with CUDA support  
- **Storage**: Google Drive integration for dataset management  

### 3.3 Future Environment (Reference)
- **OS**: Raspberry Pi OS (64bit)  
- **Hardware**: Raspberry Pi 5 (8GB RAM)  
> *Not used in this phase*

---

## 4. Software Components

- **Language**: Python 3.9+  
- **Deep Learning Framework**: Ultralytics YOLOv8 (CPU mode)  
- **Key Libraries**:
  - OpenCV (image I/O and drawing)  
  - NumPy  
  - torch, torchvision (CPU builds)

---

## 5. Functional Requirements

### 5.1 Model Training Requirements
1. **Dataset Preparation**  
   - Collect and prepare insect datasets (focusing on beetles)  
   - Convert datasets to YOLO format with proper annotations  
2. **Model Fine-tuning**  
   - Fine-tune YOLOv8 model using custom insect datasets  
   - Train specifically for Japanese rhinoceros beetle detection  
3. **Model Validation**  
   - Validate trained model performance on test datasets  
   - Generate training metrics and loss curves  

### 5.2 Detection Application Requirements
1. **Standalone Execution**  
   - Users run the script via `python detect_insect.py`  
2. **Directory Specification**  
   - Input and output directories specified via command-line arguments  
3. **Batch Inference**  
   - Perform sequential YOLO inference using custom trained model on CPU  
4. **Result Visualization & Saving**  
   - Draw bounding boxes for detected beetles and save images to the output directory with the same filename as PNG  
5. **Logging**  
   - Record filename, detection status, count, and processing time (ms) to both terminal and a log file  
6. **Help Display**  
   - Provide usage instructions via `-h, --help` options

---

## 6. Non-Functional Requirements

- **Performance**: Combined inference and drawing time ≤ 1,000 ms per image (CPU environment)  
- **Reliability**: Continue processing remaining files on exceptions  
- **Portability**: Run within a Python virtual environment (venv/conda)  
- **Reproducibility**: Log model version and weight filename

---

## 7. Data Requirements

- **Input**: JPEG/PNG still images (any resolution)  
- **Output**: PNG images of the same resolution with bounding boxes  
- **Log**: CSV format (`filename, detected, count, time_ms`)

---

## 8. Testing & Evaluation Criteria

### 8.1 Training Performance
1. **Model Convergence**  
   - Training loss should decrease consistently  
   - Validation mAP@0.5 ≥ 0.7 for beetle detection  
2. **Training Stability**  
   - Complete training without crashes  
   - Generate reproducible results  

### 8.2 Detection Accuracy
1. **Beetle Detection Performance**  
   - True positive rate ≥ 85% for Japanese rhinoceros beetles  
   - False positive rate ≤ 5%  
   - Validation on ≥ 50 sample images including beetles  
2. **Processing Time**  
   - Average processing time ≤ 1 second (CPU on WSL)  
3. **Stability**  
   - No crashes over 50 consecutive image processes

---

## 9. Constraints & Assumptions

### 9.1 Training Phase
- Access to insect datasets (publicly available or collected)  
- Sufficient training time and computational resources  
- Proper dataset annotation and formatting  

### 9.2 Inference Phase
- Custom trained YOLOv8 weights available locally  
- Python environment set up in a virtual environment with required dependencies  
- No network required; fully on-device inference  

---

## 10. Training Data Requirements

### 10.1 Dataset Specifications
- **Dataset Source**: https://universe.roboflow.com/z-algae-bilby/beetle
- **Total Dataset Size**: 500 images with YOLO format annotations
- **Classes**: 1 class ('beetle' - general beetle detection)
- **Primary Target**: Japanese rhinoceros beetles (Trypoxylus dichotomus)  
- **Secondary Targets**: Other beetle species for robust training  
- **Format**: YOLO format with bounding box annotations  
- **Image Quality**: High resolution (≥ 640x640 pixels recommended)
- **License**: CC BY 4.0  

### 10.2 Data Split (Actual Implementation)
- **Training Set**: 400 images (80% of total dataset)
- **Validation Set**: 50 images (10% of total dataset)  
- **Test Set**: 50 images (10% of total dataset)  

### 10.3 Directory Structure (Actual Implementation)
```
insect-detection-training/
├── datasets/
│   ├── train/
│   │   ├── images/            # 400 training images
│   │   └── labels/            # 400 training labels (.txt)
│   ├── valid/
│   │   ├── images/            # 50 validation images
│   │   └── labels/            # 50 validation labels (.txt)
│   ├── test/
│   │   ├── images/            # 50 test images
│   │   └── labels/            # 50 test labels (.txt)
│   └── data.yaml              # YOLOv8 configuration file
├── weights/                   # Local model weights (not committed to Git)
└── training_results/          # Training logs and metrics
```

### 10.4 File Format Requirements (Actual Implementation)
- **Images**: JPEG format with Roboflow naming convention
- **Labels**: YOLO format (.txt files) with class_id=0 for 'beetle', x_center, y_center, width, height (normalized coordinates)
- **Configuration**: data.yaml file with relative paths and single class definition
- **Naming**: Image and label files have matching base names with Roboflow hash identifiers
- **Class Definition**: Single class 'beetle' with class_id=0  

---

## 11. Model Training Workflow

### 11.1 Local Environment (Data Preparation)
1. **Data Collection & Preparation**  
   - Collect insect images from public datasets  
   - Annotate images with bounding boxes  
   - Convert to YOLO format  
   - Organize in specified directory structure  

### 11.2 Cloud Environment (Training Phase)
2. **Google Colab Setup**  
   - Upload dataset to Google Drive  
   - Set up Colab notebook with GPU runtime  
   - Install required dependencies (ultralytics, etc.)  
3. **Model Configuration**  
   - Configure YOLOv8 for custom classes  
   - Set up training parameters (epochs, batch size, etc.)  
4. **Training Execution**  
   - Fine-tune YOLOv8 on insect datasets using GPU  
   - Monitor training progress and loss curves  
   - Save training checkpoints  
5. **Model Evaluation**  
   - Validate model performance on test set  
   - Generate metrics and visualizations  
   - Export confusion matrix and performance reports  

### 11.3 Local Environment (Deployment Phase)
6. **Model Deployment**  
   - Download trained weights from Hugging Face Model Hub
   - Integrate with local detection application  
   - Test inference performance on local CPU

### 11.3.1 Fine-tuned Model Distribution
- **Repository**: https://huggingface.co/Murasan/beetle-detection-yolov8
- **License**: AGPL-3.0 (inherited from YOLOv8)
- **Available Formats**: PyTorch (.pt), ONNX (.onnx)
- **Performance Metrics**: mAP@0.5: 97.63%, mAP@0.5:0.95: 89.56%  

### 11.4 Development Workflow Benefits
- **GPU Acceleration**: Faster training (hours vs days)  
- **Cost Efficiency**: Free GPU access with Google Colab  
- **Scalability**: Easy to scale up with Colab Pro if needed  
- **Reproducibility**: Notebook-based training for consistent results  

---

## 12. License and Distribution Strategy

### 12.1 License Analysis Results (2025-07-04)

#### 12.1.1 Component Licensing
| Component | License | Impact | Commercial Use |
|-----------|---------|--------|---------|
| **Project Code** | MIT | Permissive | ✅ Unrestricted |
| **YOLOv8 Framework** | AGPL-3.0 | Strong Copyleft | ⚠️ Restricted |
| **Training Dataset** | CC BY 4.0 | Attribution Required | ✅ Allowed |
| **Trained Model** | AGPL-3.0 (inherited) | Strong Copyleft | ⚠️ Restricted |

#### 12.1.2 Key License Implications

**YOLOv8 AGPL-3.0 Requirements:**
- **Network Use Clause**: Service provision = distribution requirement
- **Copyleft Effect**: Derived works must be AGPL-3.0
- **Source Code Disclosure**: Required for network services
- **Commercial Alternative**: Ultralytics Enterprise License available

**Dataset CC BY 4.0 Requirements:**
- **Attribution**: Credit to "z Algae Bilby"
- **Source Reference**: https://universe.roboflow.com/z-algae-bilby/beetle/dataset/1
- **Modification Notice**: Training constitutes derivative work
- **License Compatibility**: Compatible with AGPL-3.0

#### 12.1.3 Commercial Book Publication Analysis

**Legal Assessment: ✅ PERMITTED**

Rationale:
- **No Direct Distribution**: Book references models via external links
- **Educational Commentary**: Book constitutes separate literary work
- **Reader Independence**: Downloads performed by readers independently
- **Clear Attribution**: Proper licensing disclosure provided

### 12.2 Distribution Strategy

#### 12.2.1 Separation Architecture (Recommended)

```
┌──────────────────────────────────────────────────────────────┐
│                    GitHub Repository                        │
│                      (MIT License)                         │
├─────────────────────────────────────────────────────────────┤
│  • Source Code                                             │
│  • Training Scripts                                        │
│  • Documentation                                           │
│  • Book Integration Code                                   │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          │ References
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Hugging Face Model Hub                   │
│                    (AGPL-3.0 License)                      │
├─────────────────────────────────────────────────────────────┤
│  • Trained Model Weights (best.pt)                        │
│  • Model Card Documentation                                │
│  • Performance Metrics                                     │
│  • Usage Examples                                          │
└─────────────────────────────────────────────────────────────┘
```

#### 12.2.2 Benefits of Separation

**Legal Benefits:**
- **License Isolation**: MIT code separate from AGPL-3.0 models
- **Commercial Safety**: Book publication legally protected
- **Clear Boundaries**: Distinct licensing responsibilities

**Technical Benefits:**
- **Programmatic Access**: `huggingface_hub` integration
- **Version Control**: Independent model and code versioning
- **Distribution**: Optimized model hosting and bandwidth

**Educational Benefits:**
- **Easy Access**: One-click model download for book readers
- **Learning Paths**: Multiple complexity levels supported
- **Reproducibility**: Consistent model access across readers

#### 12.2.3 Implementation Guidelines

**GitHub Repository (MIT):**
- Training scripts and detection code
- Book integration utilities
- Documentation and tutorials
- Clear license attribution

**Hugging Face Repository (AGPL-3.0):**
- Trained model weights (best.pt, best.onnx)
- Comprehensive model card
- Performance benchmarks
- Usage examples and attribution

**Book Integration:**
```python
# Recommended pattern for book readers
from huggingface_hub import hf_hub_download

# Automatic model download
model_path = hf_hub_download(
    repo_id="[author]/beetle-detection-yolov8",
    filename="best.pt",
    local_dir="./weights"
)
```

### 12.3 Compliance Requirements

#### 12.3.1 Attribution Requirements

**Dataset Attribution (CC BY 4.0):**
```markdown
Dataset: Beetle Detection Dataset by z Algae Bilby
Source: https://universe.roboflow.com/z-algae-bilby/beetle/dataset/1
License: CC BY 4.0
Modifications: Used for YOLOv8 model training
```

**Model Attribution (AGPL-3.0):**
```markdown
Model: YOLOv8-based Beetle Detection
Framework: Ultralytics YOLOv8 (AGPL-3.0)
License: GNU Affero General Public License v3.0
Commercial License: Available from Ultralytics
```

#### 12.3.2 Recommended Disclaimers

**For Book Publication:**
```markdown
Legal Notice: This book references open-source models subject to 
various licenses including AGPL-3.0. Readers are responsible for 
understanding and complying with applicable licenses when using 
these models. For commercial deployments, readers should consult 
legal counsel and consider enterprise licensing options.
```

**For Repository:**
```markdown
Important: The trained models referenced in this repository are 
licensed under AGPL-3.0. Commercial use may require enterprise 
licensing from Ultralytics. Educational and research use is 
encouraged under the open-source license terms.
```

### 12.4 Risk Assessment

#### 12.4.1 Risk Level: **LOW** ✅

**Mitigating Factors:**
- No direct AGPL-3.0 software distribution
- Clear separation of licensed components
- Educational focus with proper attribution
- Transparent licensing documentation

#### 12.4.2 Ongoing Compliance

**Monitoring Requirements:**
- Regular license compliance review
- Updated attribution as needed
- Clear communication of licensing terms
- Consultation with legal counsel for commercial use
