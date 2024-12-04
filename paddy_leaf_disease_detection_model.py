from tensorflow.keras.models import Sequential,load_model # type: ignore
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import numpy as np

# Data Preparation
train_gen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2)
path=r"C:\Users\arthe\paddy_disease_app\dataset\paddy_disease_dataset\train"
train_data = train_gen.flow_from_directory(path,
    target_size=(224, 224),
    class_mode="categorical"
)
val_gen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2)
path=r"C:\Users\arthe\paddy_disease_app\dataset\paddy_disease_dataset\valid"
val_data = val_gen.flow_from_directory(path,
    target_size=(224, 224),
    class_mode="categorical"
)
# Build Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_data.num_classes, activation='softmax')
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_data, epochs=7)

test_gen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2)
path=r"C:\Users\arthe\paddy_disease_app\dataset\paddy_disease_dataset\test"
test_data = test_gen.flow_from_directory(path,
    target_size=(224, 224),
    class_mode="categorical"
)
predictions = model.predict(test_data)
predicted_classes = np.argmax(predictions, axis=1)

# Map class indices to their respective labels (based on training data)
class_labels = list(test_data.class_indices.keys())
predicted_labels = [class_labels[idx] for idx in predicted_classes]

# Print predictions
for i, filepath in enumerate(test_data.filenames):
    print(f"Image: {filepath} -> Predicted label: {predicted_labels[i]}")
