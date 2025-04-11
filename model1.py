import tensorflow as tf
from tensorflow import keras
import numpy as np

# 1. Prepare Your Dataset (Example using NumPy arrays)
# Replace this with your actual data loading and preprocessing
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

# 2. Define Your Model Architecture
# def create_model():
#     model = keras.Sequential(
#         [
#             keras.layers.Flatten(input_shape=(28, 28)),
#             keras.layers.Dense(128, activation="relu"),
#             keras.layers.Dense(10, activation="softmax"),
#         ]
#     )
#     return model

# Create an initial model
# model = create_model()
model = model.load_model()

# 3. Compile Your Model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 4. Train Your Model (Initial Training)
print("Initial Training:")
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# 5. Save Model Weights
weights_path = "my_model_weights.weights.h5"
model.save_weights(weights_path)
print(f"Saved model weights to: {weights_path}")

# --- Later, when you want to retrain ---

# 6. Load Model Weights
loaded_model = create_model()  # Create a new instance of the same model architecture
loaded_model.load_weights(weights_path)

# Compile the loaded model (important!)
loaded_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Evaluate the loaded model (optional)
print("\nEvaluation after loading weights:")
loss, accuracy = loaded_model.evaluate(x_test, y_test, verbose=0)
print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

# # 7. Continue Training (Optional)
# print("\nContinuing Training:")
# loaded_model.fit(x_train, y_train, epochs=3, batch_size=32, validation_split=0.2)

# # Save the weights after continued training (optional)
# continued_weights_path = "my_model_weights_continued.weights.h5"
# loaded_model.save_weights(continued_weights_path)
# print(f"Saved continued training weights to: {continued_weights_path}")