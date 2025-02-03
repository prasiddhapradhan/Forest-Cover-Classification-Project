import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


df = pd.read_csv("cover_data.csv")

print(df.head())

print(df.info())

print(df.isnull().sum())

#Define features and target
target_column = "class"

#Separate features and target variable
features = df.drop(columns=[target_column])
target = df[target_column]

# Ensure target labels start from 0 (if they are 1-7, shift to 0-6)
target = target - 1  # Shifts labels from 1-7 to 0-6

# Check unique target classes
print("Unique target values:", target.unique())

#Check unique target classes
print(target.unique())

#Initialize scaler
scaler = StandardScaler()

#Scale features
scaled_features = scaler.fit_transform(features)

#Convert back to DataFrame
features = pd.DataFrame(scaled_features, columns=features.columns)

#STEP 4: SPLIT DATA INTO TRAINING AND TEST SETS

#Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

#STEP 5: BUILD THE DEEP LEARNING MODEL

# Define model
model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(7, activation="softmax")  # 7 output classes
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Show model summary
model.summary()

#STEP 6: TRAIN THE MODEL
#Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

#STEP 7: EVALUATE MODEL PERFORMANCE

#Evaluate the model

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

#Plot accuracy & Loss
plt.figure(figsize=(12,5))

# Accuracy plot
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Model Accuracy")

# Loss plot
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Model Loss")

plt.show()

# Save model
model.save("forest_cover_model.h5")
print("Model saved successfully.")


# Model Performance Analysis
# Accuracy Analysis
# The accuracy plot indicates that both the training and validation accuracy improved steadily over the 50 training epochs. 
# The validation accuracy surpasses the training accuracy throughout the training process, reaching a final value of approximately 0.87, while the training accuracy stabilizes around 0.83. 
# This suggests that the model is performing well on the validation set and generalizing effectively.

# Loss Analysis
# The loss plot shows a consistent decline in both training and validation loss over the epochs. 
# The validation loss decreases more rapidly compared to the training loss and stabilizes at a lower value. 
# This indicates that the model is learning effectively without significant overfitting. 
# The lower validation loss compared to training loss suggests that regularization techniques such as dropout are helping improve generalization.

# Overall Assessment
# The model demonstrates strong learning capability, achieving high validation accuracy with continuously decreasing loss. 
# The absence of divergence between training and validation curves indicates that overfitting is minimal. 
# However, the consistently higher validation accuracy compared to training accuracy suggests the possibility of data augmentation effects, batch normalization benefits, or an imbalance in the dataset.

