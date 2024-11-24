import os
import random
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical

# Load Images function
def load_image_dataset(image_folder, max_images=70, target_size=(224, 224)):
    """
    Load a set number of images from a folder and preprocess them.
    """
    images = []
    valid_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [file for file in os.listdir(image_folder) if file.endswith(valid_extensions)]
    selected_files = random.sample(image_files, min(max_images, len(image_files)))
    
    for filename in selected_files:
        file_path = os.path.join(image_folder, filename)
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, target_size)
        images.append(image)
    
    return np.array(images, dtype='float32')

#Assign labels function
def create_dataset_with_labels(melanoma_images, naevus_images):
    """
    Combine melanoma and naevus images, assign labels, and shuffle the dataset.
    """
    melanoma_labels = np.zeros(len(melanoma_images)) # Label: 0 for melanoma
    naevus_labels = np.ones(len(naevus_images))     # Label: 1 for naevus
    
    combined_images = np.concatenate([melanoma_images, naevus_images], axis=0)
    combined_labels = np.concatenate([melanoma_labels, naevus_labels], axis=0)
    
    # Shuffle the dataset
    shuffled_indices = np.random.permutation(len(combined_images))
    return combined_images[shuffled_indices], combined_labels[shuffled_indices]

# Define AlexNet Model
def build_alexnet_model(input_shape=(224, 224, 3), num_classes=2, dropout_rate=0.5):
    """
    Define the AlexNet architecture.
    """
    model = Sequential()
    model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=4, padding='valid', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
    model.add(Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
    model.add(Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    
    opt = SGD(learning_rate=0.001, momentum=0.9)  
    model.compile(loss=categorical_crossentropy, optimizer=opt, metrics=['accuracy'])
    return model

def main():
    # Define dataset paths
    melanoma_dir = os.path.join("complete_mednode_dataset", "melanoma")
    naevus_dir = os.path.join("complete_mednode_dataset", "naevus")
        
    # Load and balance the dataset
    melanoma_images = load_image_dataset(melanoma_dir, max_images=70)
    naevus_images = load_image_dataset(naevus_dir, max_images=100)[:70]

    # Split into training and testing sets .28 of 70 is 20
    train_melanoma, test_melanoma = train_test_split(melanoma_images, test_size=0.28, random_state=42)
    train_naevus, test_naevus = train_test_split(naevus_images, test_size=0.28, random_state=42)
    
    # Create datasets with labels
    train_data, train_labels = create_dataset_with_labels(train_melanoma, train_naevus)
    test_data, test_labels = create_dataset_with_labels(test_melanoma, test_naevus)
    
    # Normalize data
    train_data /= 255.0
    test_data /= 255.0

    # One-hot encode labels
    train_labels_onehot = to_categorical(train_labels, num_classes=2)
    test_labels_onehot = to_categorical(test_labels, num_classes=2)

    # Hyperparameter tuning: Dropout rates
    dropout_rates = [0.2]  #, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
    best_dropout = 0
    best_training_accuracy = 0
    results = []

    
    print("Table 1: Dropout Rate vs Image Classification Accuracy")
    accuracy_results = []
    
    for dropout_rate in dropout_rates:
        model = build_alexnet_model(dropout_rate=dropout_rate)
    
        batch_size = 32
        num_epochs = 5
        model.fit(train_data, train_labels_onehot, batch_size=batch_size, epochs=num_epochs, validation_split=0.1)
        
        loss, accuracy = model.evaluate(train_data, train_labels_onehot, verbose=0)
        print(f"Dropout Rate: {dropout_rate}, Training Accuracy: {accuracy * 100:.2f}%")
        accuracy_results.append((dropout_rate, accuracy * 100))
        
        
        if accuracy > best_training_accuracy:
            best_dropout = dropout_rate
            best_training_accuracy = accuracy
            
    # Output best dropout rate
    print(f"\nBest Dropout Rate: {best_dropout}, with an image classification accuracy of: {best_training_accuracy * 100:.2f}%\n")
    print(f"\nTable 1: Dropout Rate vs Image Classification Accuracy")
    
    for dropout_rate, accuracy in accuracy_results:
        print(f"Dropout Rate: {dropout_rate}, Accuracy: {accuracy:.2f}%")
    
   # Test the model with the best dropout rate
    print("\nEvaluating best model on test data...")
    best_model = build_alexnet_model(dropout_rate=best_dropout)
    best_model.fit(train_data, train_labels_onehot, batch_size=32, epochs=5, validation_split=0.1, verbose=0)
    test_loss, test_accuracy = best_model.evaluate(test_data, test_labels_onehot, verbose=0)

    print(f"\nTable 2: Image Classification Accuracy of Best dropout rate")
    print(f"Best Dropout Rate: {best_dropout}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


if __name__ == '__main__':
    main()
