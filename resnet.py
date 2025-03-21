import numpy as np
import pandas as pd
from pathlib import Path
import os.path
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalAveragePooling2D
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

def load_images(image_dir='resnet_dataset'):
    image_dir = Path(image_dir)
    filepaths = list(image_dir.glob(r'**/*.jpg'))
    print(f"Found {len(filepaths)} images for training")
    labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    images = pd.concat([filepaths, labels], axis=1)
    return images

def split_data(images, train_size=0.7):
    return train_test_split(images, train_size=train_size, shuffle=True, random_state=1)

def create_generators(train_df, test_df, img_size=(256, 256), batch_size=32):
    """
    Create train, validation, and test generators.
    """
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input, validation_split=0.2)

    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input)

    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df, x_col='Filepath', y_col='Label', target_size=img_size,
        color_mode='rgb', class_mode='categorical', batch_size=batch_size, shuffle=True, seed=42, subset='training')

    val_images = train_generator.flow_from_dataframe(
        dataframe=train_df, x_col='Filepath', y_col='Label', target_size=img_size,
        color_mode='rgb', class_mode='categorical', batch_size=batch_size, shuffle=True, seed=42, subset='validation')

    test_images = test_generator.flow_from_dataframe(
        dataframe=test_df, x_col='Filepath', y_col='Label', target_size=img_size,
        color_mode='rgb', class_mode='categorical', batch_size=batch_size, shuffle=False)

    return train_images, val_images, test_images

def build_model():
    """
    Create and compile the ResNet50 model.
    """
    pretrained_model = tf.keras.applications.resnet50.ResNet50(
        input_shape=(256, 256, 3), include_top=False, weights='imagenet', pooling='avg')

    pretrained_model.trainable = False

    inputs = pretrained_model.input
    x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
    x = tf.keras.layers.Dense(50, activation='relu')(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_model(model, train_images, val_images, epochs=25):
    """
    Train the model and return history.
    """
    callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(train_images, validation_data=val_images, epochs=epochs, callbacks=[callbacks])
    return history

def evaluate_model(model, test_images):
    results = model.evaluate(test_images, verbose=0)
    print(f"Test Accuracy: {np.round(results[1] * 100, 2)}%")
    return results

def plot_metrics(history):
    data_his = pd.DataFrame(history.history)
    
    plt.figure(figsize=(18, 4))
    plt.plot(data_his['loss'], label='train')
    plt.plot(data_his['val_loss'], label='val')
    plt.legend()
    plt.title('Loss Function')
    plt.show()

    plt.figure(figsize=(18, 4))
    plt.plot(data_his['accuracy'], label='train')
    plt.plot(data_his['val_accuracy'], label='val')
    plt.legend()
    plt.title('Accuracy Function')
    plt.show()

def save_model(model, filename="industry.h5"):
    model.save(filename)

def generate_reports(model, test_images):
    predictions = np.argmax(model.predict(test_images), axis=1)
    matrix = confusion_matrix(test_images.labels, predictions)
    report = classification_report(test_images.labels, predictions, target_names=list(test_images.class_indices.keys()), zero_division=0)
    
    plt.figure(figsize=(10, 10))
    sns.heatmap(matrix, annot=True, cmap='viridis')
    plt.xticks(ticks=np.arange(10) + 0.5, labels=test_images.class_indices, rotation=90)
    plt.yticks(ticks=np.arange(10) + 0.5, labels=test_images.class_indices, rotation=0)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    print("Classification Report:\n", report)


def predict_image(df, model):
    industries_dict = {0:"Accessories", 1:"Chlotes", 2:"Cosmetic",
                       3:"Electronic", 4:"Food", 5:"Institution",
                       6:"Leisure", 7:"Medical", 8:"Necesities", 9:"Transportion"}
    filenames = df['filename'].values
    filenames = [os.path.join('logos', x) for x in filenames]
    predictions = []
    for filename in filenames:
        img = image.load_img(filename, target_size=(256, 256))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
        prediction = np.argmax((model.predict(img_array))[0])
        predictions.append(industries_dict[prediction])
    df['classification'] = predictions
    return df