# Neural Networks and CNN

**PART 1: Building a Basic NN**

Project Workflow

Step 1: Importing Libraries

The project uses the following libraries for data manipulation, visualization, and model building:
	•	Numpy and Pandas for data manipulation.
	•	Matplotlib and Seaborn for visualization.
	•	Scikit-learn for preprocessing, splitting data, and evaluating metrics.
	•	PyTorch for building, training, and evaluating the neural network.

Step 2: Loading the Dataset

The dataset is loaded using Pandas, and the initial steps include:
	•	Exploring the dataset size and structure (df.shape).
	•	Inspecting the unique values of the target variable (df['target'].unique()).

Step 3: Data Preprocessing
	•	Scaling: The features are scaled using StandardScaler to improve model performance.
	•	Resampling: The dataset is balanced using resampling techniques if required.
	•	Splitting: The dataset is split into training and testing sets using train_test_split.

Step 4: Model Building
	•	A neural network model is defined using PyTorch with appropriate layers and activation functions.
	•	The architecture is visualized using the torchinfo.summary() function.

Step 5: Training
	•	The model is trained using the Adam optimizer and appropriate loss functions (e.g., CrossEntropyLoss).
	•	Training and validation loops include tracking loss and accuracy.

Step 6: Evaluation
	•	Metrics like accuracy, precision, recall, F1-score, and AUC-ROC are calculated.
	•	Confusion matrices and ROC curves are plotted for a detailed analysis of the model’s performance.

**PART 2: Optimizing NN**

 This project focuses on building a robust pipeline for training and evaluating a neural network with advanced techniques, including:
	•	Handling missing values
	•	Balancing datasets
	•	Scaling features
	•	Implementing multiple optimization strategies, including early stopping and learning rate scheduling

The model is trained and validated on a structured dataset, with its performance assessed using various metrics and visualizations.

Project Workflow

1. Data Preparation
	1.	Loading and Cleaning the Dataset:
The dataset is loaded, and missing values are handled by replacing them with NaN and dropping rows with missing values to ensure clean data.
	2.	Balancing the Dataset:
The minority class is upsampled to match the size of the majority class, ensuring the dataset is balanced and reducing potential bias.
	3.	Data Splitting:
The dataset is split into training, validation, and testing sets for proper evaluation. This includes reserving a portion of the training data as a validation set to monitor the model’s performance during training.
	4.	Feature Scaling:
Features are normalized using standard scaling to improve the efficiency and stability of the training process.

2. Neural Network Training
	1.	Model Architecture:
The neural network is implemented using fully connected layers, activation functions (e.g., ReLU), and dropout layers to prevent overfitting.
	2.	Optimization Methods:
Multiple optimization strategies are implemented:
	•	Stochastic Gradient Descent (SGD): Includes momentum and weight decay for better convergence.
	•	Adam Optimizer: Efficient for sparse gradients and faster convergence.
	•	AdamW Optimizer: Incorporates decoupled weight decay, improving generalization performance.
	3.	Early Stopping:
Early stopping is used to halt training if the validation loss does not improve after a predefined number of epochs, preventing overfitting and saving training time.
	4.	Learning Rate Scheduling:
Learning rate adjustments are applied dynamically to improve convergence:
	•	ReduceLROnPlateau: Reduces the learning rate when the validation loss stops improving.
	•	StepLR: Decreases the learning rate at regular intervals during training.
	5.	Mini-Batch Training:
Data is processed in mini-batches for efficient computation and better utilization of hardware resources.

3. Evaluation
	1.	Metrics:
The model’s performance is assessed using:
	•	Accuracy
	•	Precision
	•	Recall
	•	F1-Score
	•	AUC-ROC (Area Under the Curve - Receiver Operating Characteristic)
	•	Confusion Matrix
	2.	Visualizations:
	•	Loss Curves: Display training and validation loss over epochs to track learning progress.
	•	ROC Curve: Highlights the trade-off between sensitivity and specificity.
	•	Confusion Matrix Heatmap: Provides a visual representation of true/false positives and negatives.

**PART 3: Building a CNN:**

Overview

This project focuses on building, training, and evaluating a Convolutional Neural Network (CNN) for image classification. The project utilizes PyTorch and other essential libraries to preprocess image datasets, split them into training, validation, and testing sets, and train a CNN model on grayscale images. Key aspects include dataset transformations, model evaluation using metrics like AUC-ROC and confusion matrix, and performance visualization.

1. Data Preparation

1.1 Dataset Loading
	•	Images are loaded from a specified directory using PyTorch’s ImageFolder.
	•	Images are converted to grayscale for computational efficiency and resized to a uniform shape (28x28 pixels).

1.2 Transformations
	•	The following transformations are applied to the dataset:
	•	Grayscale Conversion: Converts RGB images to grayscale.
	•	Resizing: Resizes images to 28x28 pixels.
	•	Normalization: Normalizes pixel values for better model training.

1.3 Splitting the Dataset
	•	The dataset is split into training, validation, and testing sets using random splitting techniques.


2. Neural Network Training

2.1 CNN Architecture
	•	A Convolutional Neural Network (CNN) is implemented using PyTorch, comprising:
	•	Convolutional Layers: Extract spatial features from the images.
	•	Pooling Layers: Reduce the spatial dimensions of feature maps.
	•	Fully Connected Layers: Perform classification based on the extracted features.
	•	Activation Functions: Non-linear transformations (e.g., ReLU) are applied to introduce non-linearity.

2.2 Optimization
	•	Optimizer: The training process is optimized using the Adam optimizer.
	•	Loss Function: Cross-Entropy Loss is used for multi-class classification.

2.3 Training Pipeline
	•	The model is trained on the training set, and its performance is monitored on the validation set across epochs.
	•	A data loader processes the dataset in mini-batches to optimize memory usage.

3. Evaluation

Metrics:
	•	Confusion Matrix: Displays the classification results for each class.
	•	Classification Report: Provides precision, recall, and F1-scores for each class.
	•	AUC-ROC: Evaluates the model’s ability to distinguish between classes using receiver operating characteristics.

Visualizations:
	•	Loss Curves: Training and validation loss are plotted over epochs to monitor learning progress.
	•	Confusion Matrix Heatmap: A heatmap visualization of the confusion matrix is generated for insights into classification performance.

**PART 4: VGG - 13 Implementation:**

 Overview

This notebook continues the development and evaluation of a Convolutional Neural Network (CNN) for image classification. The focus is on preprocessing, dataset splitting, and further refining the training process, including model evaluation using advanced metrics such as AUC-ROC and confusion matrix. The project utilizes PyTorch for model implementation and analysis.

Project Workflow

1. Data Preparation

1.1 Dataset Loading
	•	Images are loaded using PyTorch’s ImageFolder, which organizes datasets based on folder structure.
	•	Images are converted to grayscale to simplify the computational process, and they are resized to a uniform size of 28x28 pixels.

1.2 Transformations
	•	The dataset undergoes preprocessing with the following transformations:
	•	Grayscale Conversion: Converts RGB images to a single-channel grayscale.
	•	Resizing: Ensures all images are of size 28x28.
	•	Normalization: Scales pixel values for improved training stability.

1.3 Dataset Splitting
	•	The dataset is split into training, validation, and testing sets with an 80%-10%-10% split:
	•	Training set: Used for model training.
	•	Validation set: Used for hyperparameter tuning and monitoring overfitting.
	•	Testing set: Used for final evaluation of model performance.

2. Neural Network Training

2.1 Model Architecture
	•	The CNN architecture includes:
	•	Convolutional Layers: Extract spatial features from input images.
	•	Pooling Layers: Reduce spatial dimensions and prevent overfitting.
	•	Fully Connected Layers: Perform classification tasks based on the extracted features.
	•	ReLU Activation Functions: Introduce non-linearity to the model.

2.2 Optimization
	•	Adam Optimizer: Used to minimize loss and optimize the model parameters.
	•	Cross-Entropy Loss Function: Suitable for multi-class classification problems.

2.3 K-Fold Cross-Validation
	•	The dataset is further split using K-Fold Cross-Validation to:
	•	Train and validate the model on multiple folds of data.
	•	Ensure robust evaluation by averaging performance metrics across folds.

3. Evaluation

Metrics:
	•	AUC-ROC: Measures the model’s ability to differentiate between classes.
	•	Classification Report: Provides precision, recall, and F1-scores for all classes.
	•	Confusion Matrix: Visualizes true positives, false positives, true negatives, and false negatives.

Visualizations:
	1.	Loss and Accuracy Curves: Track the training and validation performance over epochs.
	2.	Confusion Matrix Heatmap: Provides insights into class-specific classification performance.
	3.	ROC Curve: Highlights the trade-off between sensitivity and specificity.

 
