# HAM10000-MultiModal-ConvNeXt-Small
Multi-modal skin lesion classification on the HAM10000 dataset using ConvNeXt-Small. The model fuses dermoscopic images with clinical metadata (age, sex, localization) to improve performance on imbalanced classes.
Multi-Modal ConvNeXt-Small Model for HAM10000 Skin Lesion Classification

In this project, we implemented a multi-modal deep learning model for skin lesion classification using the HAM10000 dataset. Unlike previous models that only processed images, this fifth model incorporates both image data and patient metadata (age, sex, lesion localization) to improve predictive performance.

Data Preparation and Preprocessing

The HAM10000 dataset was loaded from Google Drive, unzipped, and metadata was preprocessed. Missing age values were imputed with the mean, categorical variables (sex, localization) were converted to one-hot encoding, and age was standardized using StandardScaler. Image paths were mapped to metadata, and lesion labels were converted into numeric classes (0–6). The dataset was split into training (80%) and validation (20%) subsets with stratification to maintain class distribution. Class weights were computed to mitigate class imbalance.

Dataset Pipeline

A tf.data.Dataset pipeline was created to handle two inputs per sample: the image and its corresponding metadata. Images were decoded, resized to 224×224, and batched. Metadata was processed as float32 tensors, allowing simultaneous feeding into the model.

Model Architecture

Image Branch: Utilizes ConvNeXt-Small pretrained on ImageNet. The output features are pooled using GlobalAveragePooling2D and regularized with Dropout.

Metadata Branch: A simple MLP with two dense layers (32 and 16 units) processes patient information.

Fusion: Features from both branches are concatenated and passed through additional dense layers with dropout.

Output Layer: A final dense layer with softmax activation predicts 7 lesion classes.

Training and Evaluation

The model was compiled with the Adam optimizer and sparse categorical cross-entropy loss, and trained for up to 40 epochs with EarlyStopping monitoring validation loss. Class weights were applied during training to balance rare classes.

Validation performance:

Accuracy: 69.25%

Loss: 0.8601

Detailed classification reports and confusion matrix show improved performance on underrepresented classes compared to image-only models.

Comparison with Previous Models

EfficientNetB1 (Image-only): Achieved 68.10% accuracy, relying solely on visual features.

ConvNeXt-Small (Image-only, previous trial): Slightly lower accuracy than the multi-modal version.

Multi-modal ConvNeXt-Small (Current): 69.25% accuracy, benefiting from combined image and metadata features.

This demonstrates that integrating patient metadata with image features improves classification, particularly for rare lesion types.

Code and Repository

All code for the current and previous models (EfficientNetB1, ConvNeXt-Small image-only) is available in previous repositories, enabling reproducibility and comparison.

Dataset: HAM10000 dataset
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
