# scripts/train_transfer.py
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import (
    VGG16, ResNet50, MobileNetV2, InceptionV3, EfficientNetB0
)
from tensorflow.keras.callbacks import ModelCheckpoint
from utils import get_data_generators

def build_transfer_model(base_model, num_classes):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=predictions)

def get_base_model(name, input_shape):
    model_dict = {
        'VGG16': VGG16,
        'ResNet50': ResNet50,
        'MobileNetV2': MobileNetV2,
        'InceptionV3': InceptionV3,
        'EfficientNetB0': EfficientNetB0
    }
    return model_dict[name](weights='imagenet', include_top=False, input_shape=input_shape)

if __name__ == "__main__":
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 10
    DATA_DIR = 'dataset'

    train_gen, val_gen, _ = get_data_generators(DATA_DIR, IMG_SIZE, BATCH_SIZE)
    input_shape = (224, 224, 3)
    num_classes = train_gen.num_classes

    model_names = ['VGG16', 'ResNet50', 'MobileNetV2', 'InceptionV3', 'EfficientNetB0']
    os.makedirs('models', exist_ok=True)

    for model_name in model_names:
        print(f"\nTraining: {model_name}")
        base_model = get_base_model(model_name, input_shape)
        base_model.trainable = False  # Freeze base

        model = build_transfer_model(base_model, num_classes)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        checkpoint_path = f'models/{model_name}.h5'
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True)

        model.fit(
            train_gen,
            epochs=EPOCHS,
            validation_data=val_gen,
            callbacks=[checkpoint]
        )
