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

# Read label file
def readtxt(txt):
    mat = pd.read_csv(txt+'.txt', sep='\t').to_numpy()
    return mat[1:-1]

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

# Video and label dirs
viddir = os.path.join(base, 'morphcuts')
labdir = os.path.join(base, 'Labels')

# Output frame prediction folder
dcfdir = os.path.join(base, 'Cropped_Face_Differences')

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
    vidlist = ''# List of videos for the mode
  
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

