# XAI-MedVision


# Hybrid Architecture with Explainable AI for Medical Image Segmentation: Research Summary

## Project Overview

This report summarizes research on a novel hybrid architecture for medical image segmentation that combines UNet++, ResNet50, and TinyBERT with Explainable AI (XAI) techniques. The model was developed to address challenges in brain tumor segmentation including accuracy, computational efficiency, and clinical interpretability.

## Methodology

### Architecture Design

The proposed hybrid architecture integrates several key components:

1. **Modified UNet++ with ResNet50 Backbone**:
    - Leverages nested skip connections for improved feature fusion
    - Uses pre-trained ResNet50 encoder for hierarchical feature extraction
    - Enhanced decoder structure for precise segmentation of brain tumor regions
2. **Lightweight TinyBERT Transformer Integration**:
    - Incorporated at the bottleneck stage of UNet++
    - Captures global attention and long-range dependencies
    - Enhances contextual understanding while maintaining computational efficiency
    - Mathematical representation:
        
        ```
        E_ResNet50(x) → w,Dense_1(w) → T_TinyBERT(w) → w′,Dense_2(w′) → z′,D_U-Net++(z′) → y
        
        ```
        
3. **Explainable AI (XAI) Integration**:
    - Multiple complementary techniques:
        - SLICE (Synthetic Labeled Input Counterfactual Explanation)
        - Grad-CAM for visual heatmap generation
        - Integrated Gradients for pixel-level attribution
        - Transformer Attention Maps
    - Agentic framework (MedBrainInsight) to generate clinical interpretations of XAI outputs
    - Bridges gap between technical outputs and clinical usability

### Dataset & Preprocessing

- **BRaTS 2021 Dataset**:
    - 1251 cases with 400 expert-annotated cases
    - Multi-modal MRI scans (T1, T2, FLAIR, T1c)
    - Three target regions: enhancing tumor (ET), tumor core (TC), whole tumor (WT)
- **Preprocessing Pipeline**:
    - Data normalization and standardization
    - Resampling to 1×1×1 mm resolution
    - Skull stripping to focus on brain regions
    - Data augmentation (rotation, flipping, scaling)
    - 128×128 patch extraction
    - Tumor region extraction and label encoding

### Training Protocol

- **Hardware**: NVIDIA Tesla P100 GPU with 16GB memory
- **Software Stack**: PyTorch 1.9.0, CUDA 11.1, cuDNN 8.0.5
- **Optimization**:
    - AdamW optimizer with 1e-4 weight decay
    - OneCycleLR scheduling with max learning rate of 2e-3
    - Composite loss function: L_total = 0.4 × L_BCE + 0.6 × L_Dice
- **Training Techniques**:
    - Mixed precision training
    - Gradient accumulation and clipping
    - Early stopping after 30 epochs without improvement
    - Batch size of 32

## Results & Findings

### Quantitative Performance

- **Overall Performance**:
    - Training Dice Score: 0.86371
    - Validation Dice Score: 0.91277
    - High precision (0.99505), sensitivity (0.99522), and specificity (0.99835)
- **Region-Specific Performance**:
    - Necrotic Core (NC): 0.86187
    - Edema (ED): 0.84433
    - Enhancing Tumor (ET): 0.82962
- **Comparison with State-of-the-Art**:
    
    
    | Model | Parameters (M) | WT | TC | ET |
    | --- | --- | --- | --- | --- |
    | Our Model | 19 | 0.9128 | 0.8619 | 0.8296 |
    | Swin UNETR | 62 | 0.9294 | - | - |
    | MedVisionLlama | 218 | 0.8400 | - | - |
    | E1D3 U-Net | - | 0.9256 | 0.8774 | 0.8576 |

### Qualitative Analysis

- Strong boundary delineation between different tumor regions
- Consistent identification of necrotic core areas
- Minor undersegmentation in cases with diffuse infiltrative patterns
- Most variability in enhancing tumor regions, particularly with small enhancing components
- Visual quality of segmentations closely matched ground truth annotations

### Interpretability Insights

- **Variable XAI Performance by Region**:
    - Necrotic regions showed high interpretability with strong concordance between XAI methods
    - Enhancing tumor regions exhibited greater variability between methods
    - GradCAM typically provided clearer visualization for enhancing tumors compared to other methods
- **Agentic XAI Framework Benefits**:
    - Generated structured clinical assessments from XAI visualizations
    - Provided comprehensive analysis of tumor characteristics
    - Compared strengths of different XAI techniques
    - Offered confidence analysis of model predictions
    - Suggested regions requiring further clinical evaluation

### Computational Efficiency

- Average processing time per case: 0.85s (0.52s for inference)
- Memory consumption: 287.64MB per case
- Balanced trade-off between accuracy and computational requirements
- Model size: 19 million parameters (compared to 62M for Swin UNETR and 218M for MedVisionLlama)

## Ablation Studies

- **ResNet Backbone**: Removal reduced performance, confirming importance of deep pre-trained features
- **TinyBERT Layers**: Performance drop when removed highlighted their role in modeling long-range dependencies
- **XAI Integration**: Improved interpretability without affecting accuracy

## Conclusions & Future Work

### Key Contributions

1. Successfully integrated TinyBERT into UNet++ for enhanced segmentation accuracy
2. Developed ResNet50-based encoder-decoder architecture with improved feature representation
3. Integrated complementary XAI techniques to enhance model interpretability
4. Created an agentic framework that generates human-readable explanations
5. Optimized the model for real-world deployment while maintaining accuracy
6. Comprehensive evaluation showing competitive performance with lower computational requirements

### Future Directions

- Real-time clinical integration of the XAI framework
- Extension to multi-modal disease segmentation
- Further optimization for edge devices and resource-constrained environments
- Enhanced XAI frameworks with improved region-specific interpretation
- Development of LLM-driven clinical dialogue systems

## Code Repository

All code related to the proposed model is available at: https://github.com/Vayuputra2401/XAI-MedVision
