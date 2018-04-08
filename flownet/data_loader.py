# -*- coding: utf-8 -*-
'''
Created on Sun Apr 8 2018

@ zhengsipeng
'''
import ctypes

def get_dataset(dataset_config, split_name):
    '''

    :param dataset_config: A dataet_config defined in datasets.py
    :param split_name: 'train/validate'
    :return:
    '''

    so = ctypes.CDLL()

