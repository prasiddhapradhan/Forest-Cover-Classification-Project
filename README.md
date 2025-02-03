# Forest-Cover-Classification-Project

Overview

This project involves building a deep learning model using TensorFlow and Keras to classify forest cover types based on dataset features. The dataset is preprocessed, scaled, and used to train a neural network model, followed by performance evaluation using accuracy and loss metrics.

Dataset

The dataset used for this project is cover_data.csv, which contains multiple features representing forest cover characteristics. The target variable, class, represents different forest cover types.

Project Workflow

Data Preprocessing

Loaded the dataset using pandas.

Checked for missing values and dataset information.

Extracted features and target variables.

Scaled features using StandardScaler from sklearn.

Data Splitting

Split the dataset into training (80%) and testing (20%) sets using train_test_split.

Deep Learning Model

Built a Sequential Neural Network with the following layers:

Dense (128 units, ReLU activation)

Dropout (30%)

Dense (64 units, ReLU activation)

Dropout (20%)

Dense (32 units, ReLU activation)

Dense (7 units, Softmax activation for multi-class classification)

Compiled the model using:

Optimizer: Adam

Loss Function: Sparse Categorical Crossentropy

Metrics: Accuracy

Model Training

Trained the model for 50 epochs with a batch size of 32.

Used validation data (20%) for monitoring performance.

Stored training history for visualization.

Performance Evaluation

Plotted Model Accuracy (Training vs. Validation).

Plotted Model Loss (Training vs. Validation).

Observed good generalization with validation accuracy reaching ~87% and minimal overfitting.

Results Interpretation

Accuracy Analysis:

The model shows a steady increase in accuracy, with validation accuracy outperforming training accuracy.

Loss Analysis:

The loss consistently decreases, indicating effective learning and minimal overfitting.

Possible Improvements

Hyperparameter Tuning: Adjust learning rate, dropout rate, and layer configurations.

Feature Engineering: Introduce additional transformations to improve classification.

Data Augmentation: Apply techniques to improve generalization.

Dependencies

Ensure the following dependencies are installed before running the script:

pip install pandas scikit-learn tensorflow keras matplotlib

Running the Project

Execute the script using:

python forest_cover_classification.py

Author

Prasiddha Pradhan

