import tensorflow as tf
import os
from tiramisu_net import Tiramisu
import numpy as np
import keras
import datagenerator
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger

is_use_gpu = True
if is_use_gpu:
  gpu_devices = tf.config.experimental.list_physical_devices('GPU')
  print(len(gpu_devices))
  tf.config.experimental.set_visible_devices(gpu_devices[0], 'GPU')
  tf.config.experimental.set_memory_growth(gpu_devices[0], True)
  os.environ['TF_USE_CUDNN'] = '1'
else:
  os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Create the model with the input_shape"

mdl = Tiramisu(input_shape=(240,320,3))

# Create optimizer and loss function for compile the model

# This optimizer is use in the paper of Valenzuela et.al (2021)
opt = keras.optimizers.RMSprop(learning_rate= 0.00001, decay = 0.995)

# The loss used, is Categorical cross entropy due the one-hot-encoding of the masks
loss = keras.losses.CategoricalCrossentropy(name="categorical_crossentropy")

# Compile the model
mdl.compile(loss = loss, optimizer = opt, metrics = ['accuracy'])

# Do the augmenters for training dataset
seq = iaa.OneOf([iaa.Affine(rotate=(-30, 30)), iaa.Affine(translate_percent=0.15),iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)})])

# Prepare all the datasets, this datasets inherit from tensorflow.keras.utils.Sequence

##### Train dataset #####
train_dataset = datagenerator.EyeDataset(batch_size=10, dim=(240, 320), augmentation = seq)
train_dataset.load_eyes('dataset','train')
train_dataset.prepare()

print("Image Count (Training): {}".format(len(train_dataset.image_ids)))
print("Class Count: {}".format(train_dataset.num_classes))
print("Batch Size: {}".format(train_dataset.batch_size))
for i, info in enumerate(train_dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))

##### Validation dataset #####
val_dataset = datagenerator.EyeDataset(batch_size=10, dim=(240, 320))
val_dataset.load_eyes('dataset','val')
val_dataset.prepare()

print("Image Count (Validation): {}".format(len(val_dataset.image_ids)))


# Model callbacks
checkpoint_path = 'checkpoints/'
logs = 'logs/'
model_checkpoint = ModelCheckpoint(filepath=checkpoint_path+'fc_densenet_4_3_4_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                    monitor='val_loss',
                                    verbose=1,
                                    save_best_only = True,
                                    save_weights_only=False,
                                    mode='auto',
                                    period=1)

early_stopping = EarlyStopping(monitor= 'val_loss', patience= 20, verbose = 1)

csv_logger = CSVLogger(filename=logs+'fc_densenet_4_3_4_training_log.csv', separator=',',append=True)

callbacks = [model_checkpoint, csv_logger, early_stopping]
#### START THE TRAINING ####

EPOCHS = 200
history = mdl.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, callbacks = callbacks) 


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss FC-Densenet')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('loss_vs_epochs.png')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy FC-Densenet')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('accuracy_vs_epochs.png')