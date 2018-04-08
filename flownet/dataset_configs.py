# -*- coding: utf-8 -*-
'''
Created on Sun Apr 8 2018

@ zhengsipeng

Add dataset configurations here. Each dataset must have the following structure

NAME = {
    IMAGE_HEIGHT: int,
    IMAGE_WIDTH: int,
    ITEMS_TO_DESCRIPTIONS: {
        'image_a': 'A 3-channel image.',
        'image_b': 'A 3-channel image.',
        'flow': 'A 2-channel optical flow field',
    },
    BATCH_SIZE: int
    PATHS:{
        'train': '',
        'validate': '', (optional)
    }
'''

'''
note that one step = one batch of data processed, ~not~ an entire epoch
'coeff_schedule_param': {
    'half_life': 50000,     after this many steps, the value will be i + (f - i) / 2
    'initial_coeff': 0.5    initial value
    'final_coeff':1,        final value
}
'''

FLYING_CHAIRS_DATASET_CONFIG = {
    'IMAGE_HEIGHT': 384,
    'IMAGE_WIDTH': 512,
    'ITEMS_TO_DESCRIPTIONS': {
        'image_a': 'A 3-channel image.',
        'image_b': 'A 3-channel image.',
        'flow': 'A 2-channel optical flow field',
    },
    'SIZES': {
        'train': 22232,
        'validate': 640,
        'sample': 8,
    }
    'BATCH_SIZE': 8,
    'PATHS': {
        'train': './data'
    }
}
