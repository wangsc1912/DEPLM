import sys
import os
import h5py
import numpy as np
import csv
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from PyAedatTools.ImportAedat import ImportAedat
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import extractdata_uti as uti


NUM_CLASSES = 10
DATA_PATH = '/home/swang/datasets/dvs_gesture_dataset/DvsGesture'
TRAIN_FILE = os.path.join(DATA_PATH, 'trials_to_train.txt')
SAVE_PATH = BASE_DIR
DATA_PER_FILE = 4000


def generate_train_data(WINDOW_SIZE, STEP_SIZE, SEQ_LEN, NUM_POINTS):

    EXPORT_PATH = uti.get_export_path(SAVE_PATH, NUM_CLASSES, WINDOW_SIZE, STEP_SIZE, SEQ_LEN, NUM_POINTS)
    print('Data will save to', EXPORT_PATH)    
    train_data, train_timelabel = uti.get_file_list(TRAIN_FILE)
    NUM_TRAIN_FILE = len(train_data)

    row_count = 0
    exp_count = 0


    for j in range(NUM_TRAIN_FILE):#NUM_TRAIN_FILE
        data = []
        label = []
        #--------Get time lable for each class in each video----------------
        print('----------Processing File No.',j, '------------')
        print('Processing Train Data File: ',train_data[j])
        print('Reading Train Lable File: ',train_timelabel[j])
        class_label = []
        class_start_timelabel = []
        class_end_timelabel = []
        with open(os.path.join(DATA_PATH,train_timelabel[j])) as csvfile:
            csvreader = csv.reader(csvfile, delimiter = ',')
            for row in csvreader:
                class_label.append(row[0])
                class_start_timelabel.append(row[1])
                class_end_timelabel.append(row[2])
        del class_label[0]
        del class_start_timelabel[0]
        del class_end_timelabel[0]
        class_label = list(map(int, class_label))
        class_start_timelabel = list(map(int, class_start_timelabel))
        class_end_timelabel   = list(map(int, class_end_timelabel))

        #-------------Extract raw data (timestep,x,y)----------------------- 
        aedat = {}
        aedat['importParams'] = {}
        aedat['importParams']['filePath'] = os.path.join(DATA_PATH,train_data[j])
        aedat = ImportAedat(aedat)
        timestep = np.array(aedat['data']['polarity']['timeStamp']).tolist()

        #---------Extract each class from video------------------------------

        class_start_index, class_end_index = uti.get_class_index(timestep,class_start_timelabel,class_end_timelabel)

        #---------Extract data by sliding window for each class--------------
        for i in range(len(class_label)):
            data_temp = []
            label_temp = []
            if class_label[i] > NUM_CLASSES:
                continue
            print('EXtraciting class-',class_label[i]-1)
       
            class_timestep = timestep[class_start_index[i]:class_end_index[i]]
            class_events = np.zeros(shape=(len(class_timestep),3),dtype=np.int32)
            class_events[:,0] = class_timestep
            class_events[:,1] = aedat['data']['polarity']['x'][class_start_index[i]:class_end_index[i]]
            class_events[:,2] = aedat['data']['polarity']['y'][class_start_index[i]:class_end_index[i]]
            win_start_index,win_end_index = uti.get_window_index(class_timestep,class_timestep[0],stepsize=STEP_SIZE*1000000,windowsize = WINDOW_SIZE*1000000)
         

            NUM_WINDOWS = len(win_start_index)

            for n in range(NUM_WINDOWS):#NUM_WINDOWS

                window_events = class_events[win_start_index[n]:win_end_index[n],:].copy()

                window_events = uti.shuffle_downsample(window_events,NUM_POINTS)
                
                #-------------Downsample---------------------------------                    
                extracted_events = uti.shuffle_downsample(window_events,NUM_POINTS)

                #------------Normalize Data------------------------------
                extracted_events[:,0] = extracted_events[:,0]-extracted_events[:,0].min(axis=0)
                events_normed = extracted_events / extracted_events.max(axis=0)
                events_normed[:,1] = extracted_events[:,1] / 127
                events_normed[:,2] = extracted_events[:,2] / 127
                #------------Append data---------------------------------
                
                data_temp.append(events_normed)
                label_temp.append(class_label[i]-1)
                if (n + 1)%SEQ_LEN == 0:
                    data.append(data_temp)
                    label.append(label_temp)
                    label_temp = []
                    data_temp =[]

               
        #------------------------Shuffle and Reshape Data-------------------------
        data = np.array(data)
        label = np.array(label)
       
        idx_out = np.arange(data.shape[0])
        np.random.shuffle(idx_out)
        data = data[idx_out,:]
        label = label[idx_out,:]
        print(data.shape)
        print(label.shape)
        print(label.max())
        #------------------------Store Data as HDF5 file-------------------------
        print(row_count)
        if row_count > DATA_PER_FILE:
            exp_count += 1
            row_count = 0
            print('New file created....')

        with h5py.File(os.path.join(EXPORT_PATH,uti.train_file_name(exp_count,lstm=False)), 'a') as hf:
            if row_count == 0:
                dset = hf.create_dataset('data', shape=data.shape, maxshape = (None,SEQ_LEN,NUM_POINTS,3), chunks=True, dtype='float32')
                lset = hf.create_dataset('label',shape=label.shape, maxshape = (None,SEQ_LEN), chunks=True, dtype='int16')
            else:
                hf['data'].resize((row_count + data.shape[0],SEQ_LEN,NUM_POINTS,3))
                hf['label'].resize((row_count + label.shape[0],SEQ_LEN))
                
            hf['data'][row_count:] = data
            hf['label'][row_count:] = label
            row_count += label.shape[0]
            print(data.shape,'Data saved to '+uti.train_file_name(exp_count))
        
if __name__ == "__main__":
    generate_train_data(WINDOW_SIZE=0.5, STEP_SIZE=0.25, SEQ_LEN=1, NUM_POINTS=1024)
