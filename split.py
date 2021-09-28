import os
import operator
import cv2
import shutil
import random
import os
from shutil import copyfile

txt = open("path_of_train_dir.txt", "w") 
root_dir = input("Path where 'train' directory belongs: ")
txt.write(root_dir)
txt.close()

source_folder_dir = os.path.join(root_dir, 'train')
source_folder_dir_class = []
for class_name in os.listdir(source_folder_dir):
    source_folder_dir_class.append(os.path.join(source_folder_dir, class_name))


directory = os.path.join(root_dir, 'classify train')
train_directory = os.path.join(directory, 'training')
validation_directory = os.path.join(directory, 'validation')
test_directory = os.path.join(directory, 'testing')

if not os.path.exists(directory):
    os.mkdir(directory)
else:
    shutil.rmtree(directory)
    os.mkdir(directory)

os.mkdir(train_directory)
os.mkdir(validation_directory)
os.mkdir(test_directory)

def split_data(SOURCE, TRAINING, VALIDATION, TESTING, VALIDATION_SPLIT_SIZE, TEST_SPLIT_SIZE):
    
    '''
    SOURCE = path of each classes where the images are stored;
    TRAINING = path where the data for training is to be stored;
    VALIDATION = path where the data for validation is to be stored;
    TESTING = path where the data for testing is to be stored;
    VALIDATION_SPLIT_SIZE =  % of total data to be used for validation (in float)
    TEST_SPLIT_SIZE =  % of total data to be used for blind testing (in float)
    '''
    
    files = []
    for file_name in os.listdir(SOURCE):
        file_path = os.path.join(SOURCE, file_name)
        if os.path.getsize(file_path) > 0:
            files.append(file_name)
        else:
            print(file_name + " is zero length, so ignoring!")

    validation_length = int(len(files) * VALIDATION_SPLIT_SIZE)
    testing_length = int(len(files) * TEST_SPLIT_SIZE)
    training_length = len(files) - (validation_length + testing_length)
    
    shuffled_set = random.sample(files, len(files))
    
    training_set = shuffled_set[0:training_length]
    validation_set = shuffled_set[(training_length):(training_length+validation_length)]
    testing_set = shuffled_set[(training_length+validation_length):(training_length+validation_length+testing_length)]

    for file_name in training_set:
        this_file = os.path.join(SOURCE, file_name)
        destination = os.path.join(TRAINING ,file_name)
        shutil.copyfile(this_file, destination)
        
    for file_name in validation_set:
        this_file = os.path.join(SOURCE, file_name)
        destination = os.path.join(VALIDATION, file_name)
        shutil.copyfile(this_file, destination)

    for file_name in testing_set:
        this_file = os.path.join(SOURCE, file_name)
        destination = os.path.join(TESTING, file_name)
        shutil.copyfile(this_file, destination)

    
source_data = sum([len(files) for r, d, files in os.walk(source_folder_dir)])
print("Total no. of files: {}".format(source_data))

VALIDATION_SPLIT_SIZE = 10
VALIDATION_SPLIT_SIZE = VALIDATION_SPLIT_SIZE/100

TEST_SPLIT_SIZE = 10
TEST_SPLIT_SIZE = TEST_SPLIT_SIZE/100

train_class = []
validation_class = []
test_class = []

for class_name in os.listdir(source_folder_dir):
    train_class.append(os.path.join(train_directory, class_name))
    validation_class.append(os.path.join(validation_directory, class_name))
    test_class.append(os.path.join(test_directory, class_name))
    
for source_dir, train_dir, val_dir, test_dir in zip(source_folder_dir_class,
                                                    train_class, 
                                                    validation_class, 
                                                    test_class):
    os.mkdir(train_dir)
    os.mkdir(val_dir)
    os.mkdir(test_dir)
    
    split_data(source_dir, train_dir, val_dir, test_dir, VALIDATION_SPLIT_SIZE, TEST_SPLIT_SIZE)

    print("\nSplitted the RAW data from " + str(source_dir) + " & storing it into 3 folders at:\n" + str(train_dir) + '\n' + str(val_dir) + '\n' + str(test_dir))
