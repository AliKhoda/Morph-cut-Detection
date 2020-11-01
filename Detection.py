# Part1: extract prediction error images

import os
import cv2
import pickle
import matplotlib

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from datetime import datetime
from vgg16 import Vgg16
from CyclicGen_model_large import Voxel_flow_model

# Fixed bounding-box across the whole video
def bigbox(txt):
    minX = np.min(txt[:,1])
    maxX = np.min(txt[:,1]+txt[:,3])
    minY = np.min(txt[:,2])
    maxY = np.min(txt[:,2]+txt[:,4])

    return [minX, minY, maxX, maxY]

# Get a specific frame from a video
def get_image(filepath, frame):
    
    reader = cv2.VideoCapture(filepath+'.mp4')
    reader.set(cv2.CAP_PROP_POS_FRAMES, frame)
    check, img = reader.read()
    assert(check)
        
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/127.5 - 1.0

# Get three consequtive frames at frame location
def get_set(filepath, frame):
    
    reader = cv2.VideoCapture(filepath+'.mp4')
    reader.set(cv2.CAP_PROP_POS_FRAMES, frame-1)
    
    f = []
    for k in range(3):
        _, img = reader.read()
        f.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/127.5 - 1.0)
        
    return f

# Get prediction error for a specific frame in a video
def get_diff(sess, vid, frame):

    # Get a set of frames
    image_set = get_set(vid, frame)

    # Reshape to fit network input shape
    data_frame1 = np.expand_dims(image_set[0], 0)
    data_frame3 = np.expand_dims(image_set[2], 0)

    # Calculate height and width
    H = data_frame1.shape[1]
    W = data_frame1.shape[2]

    # Convert to adaptive height and width
    adatptive_H = int(np.ceil(H / 32.0) * 32.0)
    adatptive_W = int(np.ceil(W / 32.0) * 32.0)

    # Calculate padding
    pad_up = int(np.ceil((adatptive_H - H) / 2.0))
    pad_bot = int(np.floor((adatptive_H - H) / 2.0))
    pad_left = int(np.ceil((adatptive_W - W) / 2.0))
    pad_right = int(np.floor((adatptive_W - W) / 2.0))

    # Create feed dictionary
    feed_dict = {input_placeholder: np.concatenate((data_frame1, data_frame3), 3)}
    # Run single step update.
    prediction_np = sess.run(prediction, feed_dict=feed_dict)

    # Return prediction error after cropping
    return image_set[1] - prediction_np[-1, pad_up:adatptive_H - pad_bot, pad_left:adatptive_W - pad_right, :]

# pretrained_model_checkpoint_path
model_cp = os.path.join(base,'ckpt/CyclicGen_large/model')

# input placeholder parameter initialization
H, W = (480, 852)

# Calculate adaptive height and width
adatptive_H = int(np.ceil(H / 32.0) * 32.0)
adatptive_W = int(np.ceil(W / 32.0) * 32.0)

# Padding
pad_up = int(np.ceil((adatptive_H - H) / 2.0))
pad_bot = int(np.floor((adatptive_H - H) / 2.0))
pad_left = int(np.ceil((adatptive_W - W) / 2.0))
pad_right = int(np.floor((adatptive_W - W) / 2.0))

"""Perform test on a trained model."""
with tf.Graph().as_default():
    # Create input and target placeholder.
    input_placeholder = tf.placeholder(tf.float32, shape=(None, H, W, 6))

    input_pad = tf.pad(input_placeholder, [[0, 0], [pad_up, pad_bot], [pad_left, pad_right], [0, 0]], 'SYMMETRIC')

    edge_vgg_1 = Vgg16(input_pad[:, :, :, :3], reuse=None)
    edge_vgg_3 = Vgg16(input_pad[:, :, :, 3:6], reuse=True)

    edge_1 = tf.nn.sigmoid(edge_vgg_1.fuse)
    edge_3 = tf.nn.sigmoid(edge_vgg_3.fuse)

    edge_1 = tf.reshape(edge_1, [-1, input_pad.get_shape().as_list()[1], input_pad.get_shape().as_list()[2], 1])
    edge_3 = tf.reshape(edge_3, [-1, input_pad.get_shape().as_list()[1], input_pad.get_shape().as_list()[2], 1])

    with tf.variable_scope("Cycle_DVF"):
        # Prepare model.
        model = Voxel_flow_model(is_train=False)
        prediction = model.inference(tf.concat([input_pad, edge_1, edge_3], 3))[0]

    # Create a saver and load.
    sess = tf.Session()

    # Restore checkpoint from file.
    restorer = tf.train.Saver()
    restorer.restore(sess, model_cp)
    print('%s: Pre-trained model restored from %s' %
          (datetime.now(), model_cp))

# For each frame in each video, calculate prediction error and dump to the output folder
for mode in ['train', 'dev', 'test']:
    # Video and label dirs
    viddir = os.path.join(base, mode, 'videos')
    labdir = os.path.join(base, mode, 'labels')

    # Output frame prediction folder
    dcfdir = os.path.join(base, mode, 'prediction_errors')
  
    for vid in vidlist:
        # define label file
        labfile = os.path.join(labdir, mode, vid)
        if os.path.exists(labfile + '.txt'):
            print('({}) {} ...'.format(count, vid), end = '')

            # define video file
            vidfile = os.path.join(viddir, mode, vid)

            # define output file
            dcffile = os.path.join(dcfdir, mode, vid)

            # makedirs
            if not os.path.exists(dcffile):
                os.makedirs(dcffile)

            # load label
            txt = readtxt(labfile)
            # calculate fixed bounding box
            box = bigbox(txt)
            boxp = (np.array([box[0]*W, box[1]*H, max((box[2] - box[0])*W, (box[3] - box[1])*H)]).astype(int))

            # for each line in label file
            for c, line in enumerate(txt):
                # Get prediction error image
                d = get_diff(sess, vidfile, line[0])

                # Get cropped face region and save to output file
                cropped_face = d[boxp[1]:(boxp[1]+boxp[2]), boxp[0]:(boxp[0]+boxp[2])]
                cimg = os.path.join(dcffile,'{:.0f}.npy'.format(line[0]))
                np.save(cimg, cropped_face)

            print('  Done!')
            

            
# Part 2: Detection

import os
import datetime
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt

layers = tf.keras.layers

# Read label file
def readtxt(txt):
    mat = pd.read_csv(txt+'.tsv', sep='\t').to_numpy()
    return mat[1:-1]

# Load files in folder to a dataframe
def dirtodataframe(split, base):
    pred_error_dir = os.path.join(base, split, 'prediction_errors')
    label_list = os.listdir(os.path.join(base, split, 'labels'))
    
    files = []
    labels = []
    for label in label_list:
        labels = readtxt(label)['Label']
        
        for j,k in enumerate(labels):
            files.append(os.path.join(pred_error_dir,os.path.split(label)[1],f'{j+1}.npy'))
            labels.append('real' if k == 0 else 'fake')

    dataframe = pd.DataFrame(data={'filename': files, 'class': labels})
    dataframe = dataframe.reset_index(drop=True)
    return dataframe

# Concatenate frames and create tensorflow dataset
def create_dataset_concat_gray(dataframe):
    num_features = imgsize*imgsize*3
    header_offset = npy_header_offset(dataframe.to_numpy()[0,0])
    
    imgdataset = tf.data.FixedLengthRecordDataset(dataframe.to_numpy()[:,0], num_features * dtype.size, header_bytes=header_offset)
    imgdataset = imgdataset.map(lambda s: tf.image.rgb_to_grayscale(tf.reshape(tf.io.decode_raw(s, dtype), (imgsize,imgsize, 3))))
    imgdataset = imgdataset.window(size=5, shift=1, stride=1, drop_remainder=True).flat_map(lambda x: x.batch(5))
    imgdataset = imgdataset.map(lambda x: tf.reshape(x, (imgsize*5,imgsize,1)))
    
    labels = list(np.array([dataframe.to_numpy()[2:-2,1] == k for k in output_classes]).T.astype(np.float))
    labdataset = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((imgdataset, labdataset))

# Get ratio of real and fake
def get_ratio(dataframe):
    labels = list(np.array([dataframe.to_numpy()[2:-2,1] == k for k in output_classes][0]).T)
    return sum(labels)/len(labels)

dtype = tf.float32
imgsize = 64
epochs = 20
workers = 4
batchsize = 32
output_classes=['fake','real']

base = ''# features and labels directory

# create train, val, and test dataframes
Train = dirtodataframe('train',base)
Val = dirtodataframe('dev',base)
Test = dirtodataframe('test',base)

# define class_weights
ratio = get_ratio(Train)
class_weights = {0:1/(2*ratio), 1:1/(2*(1-ratio))}

# Create tensorflow datasets for train, val, and test sets
traingen = create_dataset_concat_gray(Train).batch(batchsize)#.prefetch(100)
valgen = create_dataset_concat_gray(Val).batch(batchsize)#.prefetch(100)
testgen = create_dataset_concat_gray(Test).batch(batchsize)#.prefetch(100)

# Define model
model = tf.keras.models.Sequential()
model.add(layers.Conv2D(128, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(64, 64, 5)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(512, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(2, activation='softmax'))

print(model.summary())

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(.0001),
    loss=tf.keras.losses.binary_crossentropy,
    metrics=['acc'])

# Train
traingen.cache()
log = model.fit(traingen, initial_epoch=0, epochs=12, verbose=1, validation_data=valgen, class_weight=class_weights,
                      max_queue_size=workers*10, workers=workers, use_multiprocessing=True, shuffle=True)

# Evaluate
elog = model.evaluate(testgen, max_queue_size=workers*10, workers=workers, use_multiprocessing=False, verbose=1)

# Score
prob = model.predict(valgen)


