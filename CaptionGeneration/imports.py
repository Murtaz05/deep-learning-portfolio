import pandas as pd
from collections import defaultdict
import os
from sklearn.model_selection import train_test_split
import random
import numpy as np


import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import nltk
# nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
import re


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

import pickle
from tqdm import tqdm  # For showing progress bar (optional)
import math



from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input

from tensorflow.keras.layers import Embedding, LSTM, Dense, Input
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Concatenate, RepeatVector, LSTM, TimeDistributed, Dense, Input
from tensorflow.keras.models import Model

import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np



#callbacks
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
