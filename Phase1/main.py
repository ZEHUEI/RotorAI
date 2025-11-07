#train with tensorflow and keras
import tensorflow as tf
import json
import os
import numpy as np
from tensorflow.keras import layers, models


#COCO files
train_img="Data/images/train"
val_img="Data/images/validation"
train_json = "Data/annotations/train"
val_json = "Data/annotations/validation"


#cross validation, train_test_split