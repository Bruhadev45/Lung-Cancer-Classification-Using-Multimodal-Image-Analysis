
# Multimodal Machine Learning for Lung Cancer Classification

A comprehensive machine learning project comparing Convolutional Neural Networks, Transfer Learning (VGG16, ResNet50), and Random Forest for lung cancer image classification. Achieves up to 98% accuracy, with reproducible experiments and detailed model evaluation in a single Jupyter notebook.

---

## Overview

This repository presents a comparative study of multiple machine learning models for the classification of lung cancer from medical images. The project evaluates:

* **Convolutional Neural Network (CNN)**
* **Transfer Learning with VGG16**
* **Transfer Learning with ResNet50**
* **Random Forest**

All implementations, experiments, and results are contained within a single, well-documented Jupyter notebook.

---

## Repository Structure

* `multimodal-machine-learning-model-to-lung-cancer-classification.ipynb`: Main notebook with complete workflow.
* `Dataset/`: Contains categorized lung cancer images for model training and evaluation.
* `Models/`: Stores trained model files for reuse or further analysis.

---

## Getting Started

**Prerequisites:**

* Python 3.7+
* Jupyter Notebook

**Required Packages:**

```sh
pip install numpy pandas matplotlib scikit-learn tensorflow keras
```

**Clone the Repository:**

```sh
git clone https://github.com/algo-tushar/comparing-multimodal-machine-learning-to-lung-cancer-classification.git
cd comparing-multimodal-machine-learning-to-lung-cancer-classification
```

---

## Usage

1. Organize your dataset in the `Dataset/` folder, with appropriate subfolders for each class.
2. Launch Jupyter Notebook and open the main notebook:

   ```sh
   jupyter notebook multimodal-machine-learning-model-to-lung-cancer-classification.ipynb
   ```
3. Execute all cells sequentially. The notebook covers:

   * Data loading and preprocessing
   * Model building and configuration
   * Training and evaluation
   * Performance visualization and comparison

**Note:** Trained models are saved automatically in the `Models/` directory.

---

## Results

Models are evaluated using **accuracy, precision, recall, and F1-score**. Comprehensive performance comparisons are included in the notebook, with the best models achieving up to **98% accuracy**.

---

## Contributing

Contributions and suggestions are welcome. Please open issues or submit pull requests to help improve this project.

---

## License

This project is licensed under the [MIT License](LICENSE).

---
