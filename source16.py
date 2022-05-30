"""
Codebase for the research paper:
    Automated image splicing detection using deep CNN-learned features and ANN-based classifier
    DOI: https://doi.org/10.1007/s11760-021-01895-5
    Authors: Souradip Nath & Ruchira Naskar
    Correspondence to Souradip Nath.

"""

# The Dependencies
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, FalseNegatives, FalsePositives, TrueNegatives, \
    TruePositives, AUC
from tensorflow.keras.losses import BinaryCrossentropy

# Define the image shape
IMG_HEIGHT = 240
IMG_WIDTH = 240
BATCH_SIZE = 20

img_data_gn = ImageDataGenerator()

# Replace the paths accordingly
train_path = "D:\\PyCharm Projects2\\casia-dataset2\\Original\\Train"
valid_path = "D:\\PyCharm Projects2\\casia-dataset2\\Original\\Test"

# Define batches of the dataset
train_batches = img_data_gn.flow_from_directory(train_path,
                                                target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                class_mode='binary',
                                                classes=['Spliced', 'Authentic'],
                                                batch_size=BATCH_SIZE,
                                                shuffle=True)

valid_batches = img_data_gn.flow_from_directory(valid_path,
                                                target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                class_mode='binary',
                                                classes=['Spliced', 'Authentic'],
                                                batch_size=BATCH_SIZE, shuffle=True)

# Load the pre trained ResNet50 Model
model = ResNet50(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), weights='imagenet', include_top=False)

# Freeze the model
for layer in model.layers:
    layer.trainable = False

# Extract the last output layer
last_layer = model.get_layer('conv5_block3_out')
last_output = last_layer.output

# Create the additional layers for the classifier
x = layers.Flatten()(last_output)
x = layers.Dense(1000, activation='relu')(x)
x = layers.Dense(1000, activation='relu')(x)
x = layers.Dense(1, activation='sigmoid')(x)

# Create my own model
newModel = Model(model.input, x)

# Compile the model
newModel.compile(Adam(learning_rate=0.001), loss=BinaryCrossentropy(),
                 metrics=[BinaryAccuracy(), Precision(), Recall(), FalseNegatives(),
                          FalsePositives(), TrueNegatives(), TruePositives(), AUC()])

# Define Callbacks
file_path = 'Experiment_8_2.h5'
callbacks = [
    ModelCheckpoint(file_path, monitor='val_loss', mode='min', save_best_only=True, verbose=1),
]

# Load weights from the training session
newModel.load_weights(file_path)

# Training
# history1 = newModel.fit(train_batches, verbose=1, callbacks=callbacks, validation_data=valid_batches,
#                         epochs=25, steps_per_epoch=len(train_batches), validation_steps=len(valid_batches))


test_path = "D:\\PyCharm Projects2\\casia-dataset2\\Original\\Valid"
test_batches = img_data_gn.flow_from_directory(test_path,
                                               target_size=(IMG_HEIGHT, IMG_WIDTH),
                                               class_mode='binary',
                                               classes=['Spliced', 'Authentic'],
                                               batch_size=240)

for steps in range(len(test_batches)):
    test_images, test_labels = next(test_batches)
    (newModel.evaluate(test_images, test_labels))


# Save the model
# model_json = newModel.to_json()

# with open("CNNImageSplicingDetectorModelResNet50_11.json", "w") as json_file:
#     json_file.write(model_json)

# serialize weights to HDF5
# newModel.save_weights("Experiment_8_2.h5")
