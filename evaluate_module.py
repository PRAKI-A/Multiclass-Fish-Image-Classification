# scripts/evaluate_models.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from utils import get_data_generators

def evaluate_model(model_path, test_gen):
    model = load_model(model_path)
    preds = model.predict(test_gen)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes
    labels = list(test_gen.class_indices.keys())

    print(f"\nEvaluation for {os.path.basename(model_path)}")
    print(classification_report(y_true, y_pred, target_names=labels))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.title(f'Confusion Matrix - {os.path.basename(model_path)}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    DATA_DIR = 'dataset'
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    _, _, test_gen = get_data_generators(DATA_DIR, IMG_SIZE, BATCH_SIZE)

    model_dir = 'models'
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]

    for model_file in model_files:
        evaluate_model(os.path.join(model_dir, model_file), test_gen)
