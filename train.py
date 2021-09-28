# Import Libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import cv2
import random
import shutil
import time
import matplotlib
import glob
import operator
import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from shutil import copyfile
import pandas as pd
import PIL
from mlxtend.plotting import plot_confusion_matrix
from tqdm import tqdm
from classical_cnn import classical_cnn

# Setting up hardware
physical_devices = tf.config.experimental.list_physical_devices('GPU')

if physical_devices != []:
    print("Using GPU")
    for i in physical_devices:
        tf.config.experimental.set_memory_growth(i, True)
else:
    print("Using CPU")
    pass

# Load Dataset
open_root_dir = open("path_of_train_dir.txt", "r")
root_dir = open_root_dir.read()
open_root_dir.close()
classify_train = os.path.join(root_dir, 'classify train')

train_directory = os.path.join(classify_train, 'training')
validation_directory = os.path.join(classify_train, 'validation')
test_directory = os.path.join(classify_train, 'testing')

train_directory, validation_directory, test_directory

# Hyper-parameter setting-1
learning_rate = 0.001
epoch = 100
batch_size = 8
lambd = 0

# Characteristics folder
char = 'result'

if not os.path.exists(char):
    os.mkdir(char)
else:
    shutil.rmtree(char)
    os.mkdir(char)

# Learning rate decay
steps = 10 # change steps to 1 to apply exponential decay

def lr_schedule(epoch):
    return learning_rate * (0.1 ** int(epoch / steps))
    
best_model_address = os.path.join(char, 'model.h5')

# Callbacks
patience = 10
metric = 'loss'
mode = 'min'

callback = [keras.callbacks.LearningRateScheduler(lr_schedule, verbose = 1),
            keras.callbacks.EarlyStopping(monitor = metric, min_delta = 0.001, patience = patience, verbose=1, mode = mode, restore_best_weights = True),
            keras.callbacks.ModelCheckpoint(best_model_address, monitor = metric, verbose=1, save_best_only=True, save_weights_only=False, mode = mode)]

class_no = len(os.listdir(train_directory))

if class_no <= 2:
    class_mode = 'binary'
    output_activation = 'sigmoid'
    output_neurons = 1
    losses = 'binary_crossentropy'

else:
    class_mode = 'categorical'
    output_activation = 'softmax'
    output_neurons = class_no
    losses = 'categorical_crossentropy'



h = 128
w = h
color_mode = 'rgb'
dim = (h,w,3)


train_datagen = ImageDataGenerator(rescale=1.0/255.)
train_generator = train_datagen.flow_from_directory(train_directory,
                                                    batch_size = batch_size,
                                                    class_mode = class_mode,
                                                    color_mode = color_mode,
                                                    target_size = (h,w),
                                                    shuffle=True)

validation_datagen = ImageDataGenerator(rescale=1.0/255.)
validation_generator = validation_datagen.flow_from_directory(validation_directory,
                                                              batch_size = batch_size,
                                                              class_mode = class_mode,
                                                              color_mode = color_mode,
                                                              target_size = (h,w),
                                                              shuffle=True)

test_datagen = ImageDataGenerator(rescale=1.0/255.)
test_generator = test_datagen.flow_from_directory(test_directory,
                                                  batch_size = batch_size,
                                                  class_mode = class_mode,
                                                  color_mode = color_mode,
                                                  target_size = (h,w),
                                                  shuffle=True)


optimizer = Adam(learning_rate = learning_rate, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False)

model = classical_cnn(lambd, dim, output_neurons, output_activation)
model.compile(optimizer = optimizer, loss = losses, metrics = ['accuracy', 
                                                               tf.keras.metrics.Precision(), 
                                                               tf.keras.metrics.Recall(), 
                                                               tf.keras.metrics.TruePositives(), 
                                                               tf.keras.metrics.TrueNegatives(), 
                                                               tf.keras.metrics.FalsePositives(),
                                                               tf.keras.metrics.FalseNegatives()])


model.summary()

start = time.time()
history = model.fit(train_generator,
                    epochs = epoch,
                    verbose = 1,
                    callbacks = callback,
                    validation_data = validation_generator,
                    shuffle=True)

end = time.time()

duration = end-start

train_score = model.evaluate(train_generator)
val_score = model.evaluate(validation_generator)
test_score = model.evaluate(test_generator)

print("Execution Time: {} seconds".format(duration))

# Plot characteristic curves
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(len(acc))

plt.plot(epochs, acc, 'r', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title('Training and validation accuracy vs Epochs')
plt.legend()
fig_name_eps = "accuracy.eps"
fig_name_jpg = "accuracy.jpg"
plt.savefig(os.path.join(char, fig_name_eps))
plt.savefig(os.path.join(char, fig_name_jpg))

plt.plot(epochs, loss, 'r', label="Training Loss")
plt.plot(epochs, val_loss, 'b', label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title('Training and validation loss vs Epochs')
plt.legend()
fig_name_eps = "loss.eps"
fig_name_jpg = "loss.jpg"
plt.savefig(os.path.join(char, fig_name_eps))
plt.savefig(os.path.join(char, fig_name_jpg))

training_accuracy = train_score[1]*100
validation_accuracy = val_score[1]*100
test_accuracy = test_score[1]*100

print("The training accuracy is: " + str(training_accuracy) + ' %')
print("The validation accuracy is: " + str(validation_accuracy) + ' %')
print("The test accuracy is: " + str(test_accuracy) + ' %')

test_accuracy = test_score[1]*100
test_precision = test_score[2]*100
test_recall = test_score[3]*100
tp = int(test_score[4])
tn = int(test_score[5])
fp = int(test_score[6])
fn = int(test_score[7])

f1 = 2*((test_precision*test_recall)/(test_precision+test_recall))
sensitivity_k = (tp/(tp+fn))*100
specificity_k = (tn/(tn+fp))*100

print("Test Accuracy: {}".format(test_accuracy))
print("Test Precision: {}".format(test_precision))
print("Test Recall: {}".format(test_recall))

def report(y_true, y_pred, labels):
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix 
    
    print("Calculating CLASSIFICATION REPORT: ")
    classification_reports = classification_report(y_true, y_pred, target_names=labels)
    print(classification_reports)

    print("\nCalculating SENSITIVITY & SPECIFICITY..........:")
    cm = confusion_matrix(y_true, y_pred)
    total = sum(sum(cm))
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

    print("sensitivity = {:.4f}".format(sensitivity))
    print("specificity = {:.4f}".format(specificity))
    
    return cm, classification_reports, sensitivity, specificity
    
def conf_mat(cm, labels, char, file_name):
    fig, ax = plot_confusion_matrix(conf_mat=cm,
                                    colorbar=True,
                                    show_absolute=True,
                                    class_names=labels,
                                    show_normed=True)

    plt.savefig(os.path.join(char, f'{file_name}.eps'))
    plt.savefig(os.path.join(char, f'{file_name}.jpg'))
    
labels = test_generator.class_indices

def pred_sklearn(test_directory, test_generator, class_no, best_model_address, dim):
    test_class_list = []
    for test_name in os.listdir(test_directory):
        test = os.path.join(test_directory, test_name)
        test_class_list.append(test)
    test_class_list.sort()
    
    y_true = test_generator.classes
    labels = test_generator.class_indices
    
    y_pred = []
    tot = len(os.listdir(test_class_list[1]))*class_no

    best_model = load_model(best_model_address)
    
    with tqdm(total=tot) as pbar:
        for i in range(class_no):
            for filename in os.listdir(test_class_list[i]):
                file = os.path.join(test_class_list[i], filename)
                img = cv2.imread(file)
                res = cv2.resize(img, (dim[0], dim[1]))
                normed = res / 255.0
                im_arr = normed.reshape(1, dim[0], dim[1], dim[2])

                pred = best_model.predict(im_arr)
                pred_categorical = keras.utils.to_categorical(pred)

                if class_no >= 2:
                    max_pred = np.argmax(pred)
                else:
                    max_pred = np.argmax(pred_categorical)

                y_pred.append(max_pred)

                pbar.set_description("Creating classification report")
                pbar.update()
                
    return y_true, y_pred, labels
    
y_true, y_pred, labels = pred_sklearn(test_directory, test_generator, class_no, best_model_address, dim)

cm, classification_reports_sklearn, sensitivity_sklearn, specificity_sklearn = report(y_true, y_pred, labels)

conf_mat(cm, labels, char, 'confusion-matrix')

# README
from contextlib import redirect_stdout

readme_name_text = "readme.txt"
print(f"Please read the text file named {readme_name_text} for detailed information of the model.")

completeName_txt = os.path.join(char, readme_name_text) 

readme = open(completeName_txt, "w")

if len(os.listdir(train_directory)) > 2:
    readme.write(f"This is a {len(os.listdir(train_directory))}-class CLASSIFICATION")
else:
    readme.write("This is a BINARY CLASSIFICATION")


readme.write("\n\n--HYPERPARAMETERS--\n")
readme.write(f"\nInitial Learning Rate = {learning_rate}")
readme.write(f"\nNo. of epochs = {len(acc)}")
readme.write(f"\nBatch Size = {batch_size}")


readme.write("\n\n--MODEL-PARAMETERS--")
# readme.write(f"\nDropout for feature extraction = {(int(f_dropout*100))} %")
# readme.write(f"\nDropout for dense layer = {(int(d_dropout*100))} %")
readme.write(f"\nOptimizer = {optimizer}\n\n")


readme.write("Trained on a Custom Prebuilt Model\n")
# readme.write(f"\nFilter size = {size_filter}x{size_filter}\n\n")
with redirect_stdout(readme):
    model.summary()
        
    
readme.write("\n\n--MODEL-PERFORMANCE--")
readme.write(f"\nTest Accuracy = {test_accuracy} %")
readme.write(f"\nTest Precision = {test_precision} %")
readme.write(f"\nTest Recall = {test_recall} %")


readme.write("\n\n--MODEL-CHARACTERISTICS--")
readme.write(f"\nacc = {acc}")
readme.write(f"\n\nval_acc = {val_acc}")
readme.write(f"\n\nloss = {loss}")
readme.write(f"\n\nval_loss = {val_loss}")

readme.write("\n\n--Classification Report--\n")
readme.write(classification_reports_sklearn)
readme.write(f"\nSensitivity = {sensitivity_sklearn*100} %")
readme.write(f"\nSpecificity = {specificity_sklearn*100} %")


readme.write(f"\nExecution Time: {duration} seconds")

readme.close()