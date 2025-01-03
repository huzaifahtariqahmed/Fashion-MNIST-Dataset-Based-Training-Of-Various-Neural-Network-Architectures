# Training Various Neural Network Architectures on Fashion MNIST Dataset

## Overview
This repository contains implementations of three different neural network architectures trained on the Fashion MNIST dataset. The goal is to compare the performance of these architectures in terms of accuracy and other evaluation metrics.

## Neural Network Architectures
1. **Feed Forward Neural Network (FFNN)**
   - A basic fully connected network.
   - Trained over 100 epochs.

2. **Convolutional Neural Network (CNN)**
   - A deep learning model leveraging convolutional layers for feature extraction.
   - Trained over 10 epochs.

3. **Transfer Learning with VGG-16**
   - Using a pre-trained VGG-16 model (trained on ImageNet) as a feature extractor.
   - Connected to a feed forward network and trained for 20 epochs.

## Dataset
The [Fashion MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist) dataset consists of 70,000 grayscale images of 10 fashion categories, split into 60,000 training and 10,000 test images. Each image is 28x28 pixels.

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/huzaifahtariqahmed/Fashion-MNIST-Dataset-Based-Training-Of-Various-Neural-Network-Architectures.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook:
   ```bash
   jupyter notebook training_on_fashion_mnist.ipynb
   ```

## Results and Comparisons
### Evaluation Metrics
The following metrics were used to evaluate the performance of the models:
- **Accuracy**: Percentage of correctly classified samples.
- **Loss**: Cross-entropy loss during training and validation.

### Performance Comparison
| Model                  | Training Epochs | Test Accuracy | Remarks                             |
|------------------------|-----------------|---------------|-------------------------------------|
| Feed Forward NN        | 100             | 90.46%           | Basic model with limited capacity.  |
| Convolutional NN       | 10              | 92.88%           | Strong feature extraction ability.  |
| Transfer Learning (VGG)| 20              | 88.11%           | High accuracy due to pre-trained weights. |

## Analyses
- **Feed Forward Neural Network**:
  - Simpler architecture, slower convergence due to its lack of feature extraction capabilities.

- **Convolutional Neural Network**:
  - Achieves significantly better performance with fewer epochs due to convolutional layers.

- **Transfer Learning**:
  - Leverages pre-trained knowledge, resulting in faster convergence.
  
## Future Work
- Hyperparameter tuning for all models.
- Experimenting with other pre-trained architectures.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

--- 

**Contributions**: 

[Huzaifah Tariq Ahmed](https://github.com/huzaifahtariqahmed). 
