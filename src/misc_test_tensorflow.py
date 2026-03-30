#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 12:22:12 2026

@author: bipin
"""

import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))

with tf.device('/GPU:0'):
    a = tf.random.normal([2000,2000])
    b = tf.matmul(a, a)
print("Done")

print(tf.sysconfig.get_build_info())