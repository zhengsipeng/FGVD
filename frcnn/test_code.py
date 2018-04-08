#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 20:34:07 2018

@author: zhengsipeng
"""
from keras_frcnn.pascal_voc_parser import get_data
all_imgs, classes_count, class_mapping = get_data('/data1/keras-frcnn-master/VOC')