
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# data_dir = "dataset"
# data_dir = "my_data_folder"

# # filepath: f:\first_image\scripts.py
# if __name__ == "__main__":
#     data_dir = "dataset"  # Make sure this matches your actual folder name
#     for sub in ['train', 'val', 'test']:
#         path = os.path.join(data_dir, sub)
#         print(f"Checking: {path} -> Exists: {os.path.exists(path)}")
#     train_gen, val_gen, test_gen = get_data_generators(data_dir)
#     print("✅ Data generators created.")
#     print("Classes found:", train_gen.class_indices)

def get_data_generators(data_dir, img_size=(224, 224), batch_size=32):
    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'val')
    test_path = os.path.join(data_dir, 'test')

    # Augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )

    # Only rescaling for val/test
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_gen = test_datagen.flow_from_directory(
        val_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_gen = test_datagen.flow_from_directory(
        test_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_gen, val_gen, test_gen

if __name__ == "__main__":
    data_dir = "dataset"  # 🔁 Make sure this matches your actual folder name
    train_gen, val_gen, test_gen = get_data_generators(data_dir)

    print("✅ Data generators created.")
    print("Classes found:", train_gen.class_indices)



# import os
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# def get_data_generators(data_dir, img_size=(224, 224), batch_size=32):
#     train_path = os.path.join(data_dir, 'train')
#     val_path = os.path.join(data_dir, 'val')
#     test_path = os.path.join(data_dir, 'test')

#     # Augmentation for training
#     train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         rotation_range=20,
#         zoom_range=0.2,
#         horizontal_flip=True
#     )

#     # Only rescaling for val/test
#     test_datagen = ImageDataGenerator(rescale=1./255)

#     train_gen = train_datagen.flow_from_directory(
#         train_path,
#         target_size=img_size,
#         batch_size=batch_size,
#         class_mode='categorical'
#     )

#     val_gen = test_datagen.flow_from_directory(
#         val_path,
#         target_size=img_size,
#         batch_size=batch_size,
#         class_mode='categorical'
#     )

#     test_gen = test_datagen.flow_from_directory(
#         test_path,
#         target_size=img_size,
#         batch_size=batch_size,
#         class_mode='categorical',
#         shuffle=False
#     )

#     return train_gen, val_gen, test_gen

# if __name__ == "__main__":
#     data_dir = "dataset"  # Make sure this matches your actual folder name
#     for sub in ['train', 'val', 'test']:
#         path = os.path.join(data_dir, sub)
#         print(f"Checking: {path} -> Exists: {os.path.exists(path)}")
#     train_gen, val_gen, test_gen = get_data_generators(data_dir)
#     print("✅ Data generators created.")
#     print("Classes found:", train_gen.class_indices)


