#!/usr/bin/env bash

#create-tfrecords.py data/train/original data/train/segmentation train.tfrecords
#create-tfrecords.py data/eval/original data/eval/segmentation eval.tfrecords

DG="./data/DahmannGroup"

#create-tfrecords.py $DG/Abdomen/abdomen1_org/abdomen1_grey $DG/Abdomen/abdomen1_seg/abdomen1_seg_grey abdomen1.tfrecords "\d\d\d"
#create-tfrecords.py $DG/Abdomen/abdomen2_orig/abdomen2_grey $DG/Abdomen/abdomen2_seg/abdomen2_seg_grey abdomen2.tfrecords "\d\d\d"
#create-tfrecords.py $DG/Abdomen/abdomen3_orig/abdomen3_grey $DG/Abdomen/abdomen3_seg/abdomen3_seg_grey abdomen3.tfrecords "\d\d\d"
create-tfrecords.py $DG/Wingdisc/wingdisk_org/wingdisk_grey $DG/Wingdisc/wingdisk_seg/wingdisk_seg_grey wingdisk.tfrecords
