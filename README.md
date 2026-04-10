# AllMetrics: A Unified Python Library for Standardized Metric Evaluation in Machine Learning

**Paper Title:** AllMetrics: A Unified Python Library for Standardized Metric Evaluation and Robust Data Validation in Machine Learning.

**Paper:** coming Soon.

**PyPI:** https://pypi.org/project/allmetrics/ 

**GitHub:** https://github.com/MohammadRSalmanpour/AllMetrics  

**Python Version:** 3.11+

**AllMetrics** is a comprehensive Python library designed to **standardize performance metric evaluation across diverse machine learning tasks**. It provides a unified API for computing metrics while ensuring **robust data validation, consistent implementations, and standardized reporting formats**.

Many existing libraries compute evaluation metrics differently, leading to **inconsistent results across research papers, tools, and frameworks**. AllMetrics addresses these issues by providing **consistent implementations, validation mechanisms, and unified outputs** across multiple machine learning domains.

AllMetrics supports evaluation for:

- **Regression**
- **Classification**
- **Clustering**
- **Segmentation**
- **Image-to-Image Translation**

## 🔍 Table of Contents

- [📌 Motivation](#-motivation)
- [✨ Key Features](#-key-features)
- [📥 Installation](#-installation)
- [🚀 Quick Start](#-quick-start)
- [📈 Task Examples](#-task-examples)
- [📚 Supported Metrics](#-supported-metrics)
  - [Regression Metrics](#-regression)
  - [Classification Metrics](#-classification)
  - [Clustering Metrics](#-clustering)
  - [Segmentation Metrics](#-segmentation-2d3d)
  - [Image-to-Image Translation Metrics](#-image-to-image-translation)
- [🧩 Library Design Principles](#-library-design-principles)
- [📊 Output Format](#-output-format)
- [⚠️ Data Validation](#-data-validation)
- [📚 API Structure](#-api-structure)
- [❓ Troubleshooting](#-troubleshooting)
- [🕒 Version History](#-version-history)
- [📬 Maintenance](#maintenance)
- [📚 Citation](#citation)
- [📜 License](#license)
- [📬 Contact](#contact)


# 📌 Motivation
Evaluation metrics play a central role in machine learning research and practice. They are used to compare models, report experimental results, and guide model selection. However, despite their importance, metric implementations across existing libraries are often inconsistent. Differences in mathematical definitions, preprocessing assumptions, aggregation strategies, and reporting formats can lead to substantially different results—even when the same metric name is used.

These inconsistencies arise mainly from two sources:

### 1️⃣ Implementation Differences (ID)
variations in how metrics are mathematically defined or computed across tools.
### 2️⃣ Reporting Differences (RD)
variations in how results are aggregated or summarized (e.g., micro, macro, weighted, class-wise reporting).

As a consequence, identical models evaluated on identical datasets may produce different results depending on the library, framework, or configuration used. This makes cross-study comparisons difficult and may undermine reproducibility in machine learning research.

AllMetrics was developed to address this challenge. It provides a unified and transparent framework for computing evaluation metrics across a wide range of machine learning tasks. The library standardizes metric implementations, explicitly exposes evaluation assumptions, and integrates robust data validation mechanisms to ensure reliable and reproducible results.

By unifying metric evaluation for classification, regression, clustering, segmentation, and image-to-image analysis, AllMetrics enables consistent benchmarking and facilitates reproducible experimentation across diverse ML workflows.


# ✨ Key Features
### Unified Metric Evaluation
AllMetrics provides standardized implementations of evaluation metrics across multiple machine learning tasks, including classification, regression, clustering, segmentation, and image-to-image translation.

### Explicit Control of Evaluation Assumptions
The library explicitly distinguishes between Implementation Differences (ID) and Reporting Differences (RD), enabling transparent and reproducible metric evaluation.

### Robust Data Validation
Automatic validation checks help detect common data issues before metrics are computed, including:

- shape mismatches
- invalid value ranges
- class imbalance
- missing or empty labels
- outliers and abnormal distributions

### Task-Agnostic API
A unified API design allows users to evaluate models across different ML tasks using consistent function interfaces and parameter conventions.

### Extensible Architecture
Users can extend the library by adding custom metrics or integrating new validation rules while preserving standardized reporting and evaluation workflows.

### Broad Metric Coverage
The library includes more than 50 evaluation metrics spanning multiple ML domains.

### Support for Advanced Applications
AllMetrics supports specialized scenarios such as:

- multi-class and multi-label classification
- 2D and 3D medical image segmentation
- clustering quality evaluation
- image-to-image translation assessment (SSIM, PSNR)


# 📥 Installation
AllMetrics can be installed directly from PyPI using pip:
```bash
pip install allmetrics
```
After installation, the library can be imported in Python:
```python
import allmetrics
```
The package is designed to integrate easily with common scientific computing and machine learning libraries such as NumPy, PyTorch, and standard Python data pipelines.


# 🚀 Quick Start
**The following example demonstrates how to compute a simple classification metric using AllMetrics.**
```python
from allmetrics.classification import accuracy_score

y_true = [1, 0, 1, 1, 0]
y_pred = [1, 0, 0, 1, 0]

acc = accuracy_score(y_true, y_pred)
print("Accuracy:", acc)
```
AllMetrics provides additional configuration options that allow users to control validation behavior and evaluation assumptions.

**Example with validation options:**
```python
from allmetrics.classification import accuracy_score

acc = accuracy_score(
    y_true,
    y_pred,
    normalize=True,
    check_outliers=False,
    check_distribution=False,
    check_correlation=False,
    check_missing_large=False,
    check_class_balance=False
)
```
**Users can also explore available metrics programmatically:**
```python
import allmetrics

allmetrics.classification.list_of_metrics()
```
**To retrieve detailed information about a specific metric:**
```python
allmetrics.classification.get_metric_details("accuracy_score")
```
This discovery mechanism allows users to easily inspect available metrics, understand parameter configurations, and select appropriate evaluation measures for their experiments.

# 📈 Task Examples
The following examples demonstrate how AllMetrics can be used across different machine learning tasks. The library provides a consistent API for computing evaluation metrics in classification, regression, segmentation, clustering, and image-to-image analysis.
## 📊 Classification Example
```python
from allmetrics.classification import f1_score

y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]

f1 = f1_score(
    y_true,
    y_pred,
    average="macro",   # options: micro, macro, weighted, none
    zero_division=0,
    check_class_balance=True
)

print("F1 Score:", f1)
```
Key point:

All classification metrics follow a unified API. The average parameter explicitly controls Reporting Differences (RD) such as micro, macro, or weighted aggregation.
## 📈 Regression Example
```python
from allmetrics.regression import mean_absolute_error

y_true = [3.2, 2.8, 4.1, 5.0]
y_pred = [2.9, 3.0, 3.8, 5.3]

mae = mean_absolute_error(
    y_true,
    y_pred,
    check_outliers=True,
    check_distribution=True
)

print("MAE:", mae)
```
Key point:

Built-in validation can automatically detect issues such as abnormal distributions, outliers, or invalid numeric values before computing the metric.
## 🧠 Segmentation Example (2D/3D)
```python
import numpy as np
from allmetrics.segmentation import dice_score

# Example 2D masks
y_true = np.array([[1, 1, 0], [0, 1, 0]])
y_pred = np.array([[1, 0, 0], [0, 1, 1]])

dice = dice_score(
    y_true,
    y_pred,
    mode="binary",         # multi-class also supported
    ignore_background=True,
    check_empty_masks=True
)

print("Dice Score:", dice)
```
Key point:

AllMetrics supports 2D and 3D segmentation evaluation, including metrics such as Dice, IoU, Hausdorff Distance, and ASSD, which are widely used in medical imaging.
# 📚 Supported Metrics
AllMetrics includes 50+ standardized evaluation metrics covering multiple machine learning tasks. The implementations follow consistent definitions and transparent evaluation assumptions.

### 📈 Regression
- mean_absolute_error
- mean_squared_error
- mean_bias_deviation
- r_squared
- r_squared(adjusted)
- mean_absolute_percentage_error
- symmetric_mean_absolute_percentage_error
- huber_loss
- relative_squared_error
- mean_squared_log_error
- log_cosh_loss
- explained_variance
- median_absolute_error
- max_error
- mean_tweedie_deviance
- mean_pinball_loss

### 📊 Classification
- accuracy_score
- precision_score
- recall_score
- balanced_accuracy
- matthews_correlation_coefficient
- cohens_kappa
- f1_score
- confusion_matrix
- fbeta_score
- jaccard_score
- log_loss
- hamming_loss
- top_k_accuracy

### 🌀 Clustering
- adjusted_rand_index
- normalized_mutual_info_score
- silhouette_score
- calinski_harabasz_index
- homogeneity_score
- completeness_score
- davies_bouldin_index
- mutual_information
- v_measure_score
- rand_score
- adjusted_mutual_info_score
- fowlkes_mallows_score

### 🧠 Segmentation (2D/3D)
- dice_score
- iou_score
- sensitivity
- specificity
- precision
- hausdorff_distance

### 🖼️ Image-to-Image Translation
- ssim
- psnr

# 🧩 Library Design Principles
AllMetrics is designed around a set of principles aimed at ensuring reproducibility, transparency, and consistency in machine learning metric evaluation.

1. Standardized Metric Implementations
All metrics are implemented using consistent mathematical definitions and verified implementations. This minimizes Implementation Differences (ID) that often arise across different libraries.
2. Explicit ID/RD Control
A core design concept of AllMetrics is the explicit distinction between:

- ### **Implementation Differences (ID)**
- ### **Reporting Differences (RD)**
Examples include:

- averaging strategies in classification
- handling of missing classes
- background handling in segmentation
- surface distance computation methods
Making these assumptions explicit improves reproducibility and transparency in reported results.
3. Layered Architecture
The library follows a layered architecture that separates different responsibilities:

- Preprocessing Layer

Handles input validation, shape checking, and normalization rules.

- Metrics Core Layer

Provides standardized implementations of evaluation metrics.

- ID/RD Control Layer

Manages configuration related to evaluation assumptions.

- Reporting Layer

Generates interpretable and structured evaluation results.

- Extensions Layer

Allows users to add custom metrics or extend the library.

4. Task-Agnostic API
AllMetrics provides a task-agnostic API design. Metrics across different tasks follow similar function signatures and parameter conventions, making the library easy to learn and use.

5. Robust Validation by Default
To prevent misleading results, AllMetrics performs automatic validation checks before computing metrics. These checks may include:

- missing classes
- abnormal correlations
- outliers
- empty segmentation masks
- invalid value ranges
Users can customize or disable these checks depending on their workflow.
6. Extensible and Research-Friendly
The library is designed to support research workflows. Users can easily extend AllMetrics by implementing new metrics while leveraging the existing validation and reporting infrastructure.

This extensibility makes AllMetrics suitable for both applied machine learning projects and academic research.

# 📊 Output Format
AllMetrics is designed to produce clear, interpretable, and reproducible outputs. Metric functions typically return a numerical value, but they can also provide structured summaries when detailed reporting is enabled.

### Standard Output
Most metrics return a single numeric value:

```python
from allmetrics.classification import accuracy_score

acc = accuracy_score(y_true, y_pred)

print(acc)
```
Output example:
```python
0.84
```
### Multi-Class Reporting
For metrics that involve multiple classes, users can control the aggregation strategy using the average parameter.

Example:
```python
from allmetrics.classification import precision_score

precision = precision_score(
    y_true,
    y_pred,
    average="macro"
)

print(precision)
```
Available aggregation modes:

- micro – global aggregation over all samples
- macro – unweighted mean across classes
- weighted – class-weighted mean
- none – class-wise results
Example class-wise output (average="none"):
```python
[0.82, 0.76, 0.91]
```
# ⚠️ Data Validation
Reliable evaluation requires reliable inputs. AllMetrics includes an integrated data validation layer that automatically checks input data before computing metrics.

These checks help detect common issues that can silently distort evaluation results.
### Shape Consistency
Ensures that predicted and ground-truth arrays have compatible shapes.

Example problem detected:

- mismatched lengths
- incompatible tensor shapes
### Value Range Checks
Validates that predictions and labels fall within acceptable ranges.

Examples:

- classification labels outside expected class indices
- segmentation masks containing invalid values
- regression predictions with NaN or infinite values
### Class Presence & Balance
For classification tasks, AllMetrics can check:

- missing classes in predictions
- severe class imbalance
- degenerate predictions (predicting a single class only)
Example option:
```python
check_class_balance=True
```
### Outlier Detection
For regression tasks, optional outlier checks can identify abnormal values that may distort evaluation.

Example option:
```python
check_outliers=True
```
### Segmentation-Specific Checks
For image segmentation tasks, additional checks are available:

- empty masks
- background-only predictions
- mismatched mask dimensions
Example:
```python
check_empty_masks=True
```
### Configurable Validation
All validation checks are fully configurable and can be enabled or disabled depending on the application.

Example:
```python
accuracy_score(
    y_true,
    y_pred,
    check_outliers=False,
    check_distribution=False,
    check_class_balance=False
)
```
# 📚 API Structure
AllMetrics follows a task-oriented modular API design, where metrics are organized by machine learning task. This structure keeps the library intuitive and easy to navigate.

## Main Modules
```bash
allmetrics
 ├── classification
 ├── regression
 ├── clustering
 ├── segmentation
 └── image_translation
```
Each module contains task-specific evaluation metrics.

Example imports:
```python
from allmetrics.classification import accuracy_score
from allmetrics.regression import mean_squared_error
from allmetrics.clustering import rand_score
from allmetrics.segmentation import dice_score
from allmetrics.imagetoimage import psnr
```
## Metric Discovery Utilities
AllMetrics provides built-in utilities to explore available metrics.

List metrics within a module:
```python
import allmetrics

allmetrics.classification.list_of_metrics()
```
Get details about a specific metric:
```python
allmetrics.classification.get_metric_details("f1_score")
```
These utilities help users quickly discover supported metrics and understand their parameters.

## Consistent Function Signatures
Most metric functions follow a consistent structure:
```python
metric_function(
    y_true,
    y_pred,
    **options
)
```
Where:

- y_true – ground truth values
- y_pred – predicted values
- options – configuration parameters controlling validation, averaging, and evaluation behavior
This consistent API allows users to switch between metrics without changing their workflow.

# ❓ Troubleshooting
This section addresses common issues users may encounter when using AllMetrics.

## Shape Mismatch Errors
Problem
```python
ValueError: y_true and y_pred must have the same shape
```
Solution

Ensure that both arrays contain the same number of samples and compatible dimensions.

Example:
```python
len(y_true) == len(y_pred)
```
## Invalid Label Values
Problem

Labels contain values outside the expected range.

Solution

Verify that classification labels correspond to valid class indices and do not contain unexpected values.
## Empty Segmentation Masks
Problem

Segmentation metrics fail when masks contain no foreground pixels.

Solution

Enable the built-in validation checks:
```python
check_empty_masks=True
```
or ensure masks contain valid foreground regions.

## NaN or Infinite Values
Problem

Metrics return NaN due to invalid numeric values.

Solution

Check the dataset for:

- NaN values
- infinite values
- invalid predictions
## Unexpected Metric Results
If results differ from those produced by another library, possible reasons include:

- different aggregation strategies
- different implementation assumptions
- different handling of edge cases
AllMetrics makes these assumptions explicit through configuration parameters.



# 🕒 Version History

### v0.0.0 — 2025-03-05
Initial public release of AllMetrics.
Key features:

- unified evaluation framework for machine learning metrics
- support for classification, regression, clustering, segmentation, and image-to-image evaluation
- implementation of 50+ standardized metrics
- explicit control of Implementation Differences (ID) and Reporting Differences (RD)
- integrated data validation layer
- modular task-based API
- support for 2D and 3D segmentation evaluation
Future versions will expand metric coverage, improve reporting capabilities, and introduce additional validation tools for advanced machine learning workflows.



# 📬 Maintenance

For technical support and maintenance inquiries, please contact:

**Dr. Mohammad R. Salmanpour (Team Lead)**

msalman@bccrc.ca – m.salmanpoor66@gmail.com – m.salmanpour@ubc.ca

**Morteza Alizadeh (Assistant Team Lead)**

alizadehmorteza2020@gmail.com

# 👥Authors

- **Morteza Alizadeh (Backend Development, Code Refactoring, Debugging, Library Management)**
- **Mehrdad Oveisi (Evaluator, Software Engineer, AI Expert, and Advisor)** 
- **Sonya Falahati (Testing and Data prepration)** 
- **Ghazal Mousavi (Backend Development, Testing, and Data prepration)**
- **Mohsen Alambardar Meybodi (Advisor and Evaluator)**
- **Somayeh Sadat Mehrnia (Coordinator and Evaluator)**
- **Ilker Hacihaliloglu (Medical Imaging Expert and Advisor)**
- **Arman Rahmim (Fund Provider, Medical Imaging Expert, Evaluator, and Advisor)** 
- **Mohammad R. Salmanpour (Team Lead, Conceptualization, Supervisor, Fund Provider, AI and Medical Imaging Expert, and Evaluator)** 

# 📚Citation

```bibtex
@misc{abcdefgh,
      title={AllMetrics: A Unified Python Library for Standardized Metric Evaluation and Robust Data Validation in Machine Learning}, 
      author={Morteza Alizadeh and Mehrdad Oveisi and Sonya Falahati and Ghazal Mousavi and Mohsen Alambardar Meybodi and Somayeh Sadat Mehrnia and Ilker Hacihaliloglu and Arman Rahmim and Mohammad R. Salmanpour.},
      year={2025},
      eprint={2511.15963},
      archivePrefix={arXiv},
      primaryClass={physics.med-ph},
      url={https://arxiv.org/abs/2505.15931}, 
}
```

# 📜License

This open-source software is released under the **MIT License**, which grants permission to use, modify, and distribute it for any purpose, including research or commercial use, without requiring modified versions to be shared as open source. See the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/radiuma-com/PySERA/issues)
- **Documentation**: This README and the included guides
- **Examples**: See `examples/basic_usage.py`

# Acknowledgment

This study was supported by:

- [💻 **Vir**tual **Collab**oration (VirCollab) Group, Vancouver, BC, Canada](https://vircollab.com/#/) 
- [🏭 **Tec**hnological **Vi**rtual **Co**llaboration **Corp**oration (TECVICO Corp.), Vancouver, BC, Canada](https://www.tecvico.com)
- [🔬 **Qu**antitative **R**adiomolecular **I**maging and **T**herapy (Qurit) Lab, University of British Columbia, Vancouver, BC, Canada](https://www.qurit.ca)  
- [🏥 BC Cancer Research Institute, Department of Basic and Translational Research, Vancouver, BC, Canada](https://www.bccrc.ca/)  

---
# 📬Contact
AllMetrics is available **free of charge**.
For access, questions, or feedback:

**Morteza Alizadeh (Backend Developer)**

📧[AlizadehMorteza2020@gmail.com](mailto:alizadehmorteza2020@gmail.com)

**Dr. Mohammad R. Salmanpour (Team Lead)**  

📧[msalman@bccrc.ca](mailto:msalman@bccrc.ca) | [m.salmanpoor66@gmail.com](mailto:m.salmanpoor66@gmail.com) | [m.salmanpour@ubc.ca](mailto:m.salmanpour@ubc.ca)
