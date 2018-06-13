#!/usr/bin/env bash

create-tfrecords.py data/train/original data/train/segmentation train.tfrecords
create-tfrecords.py data/eval/original data/eval/segmentation eval.tfrecords
