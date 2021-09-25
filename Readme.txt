1. “Baseline Neural network.py” is the baseline implementation and “complete Neural network.py” is technique implementation in python. “Dataframes.py” generates the dataframes and histogram of class labels of overall dataset and divided folders.
2. The images folder in the paper_v1 contains histogram images and network architecture image
3. "Fuzzy.py" and "Fuzzy and Ga.py" can give results related FCM and FCM+GA. The respective care has to be taken for while the dataset. For these files type ids are used as targets.

Note: Fuzzy.ipynb and Fuzzy_and_GA.ipynb are also being upload in code_v2 folder.

The dependecies needed for FCM and FCMandGA py files :

import pandas as pd # reading all required header files
import numpy as np
import random
import operator
import math
import matplotlib.pyplot as plt 
from scipy.stats import multivariate_normal
import math
from sklearn.metrics import silhouette_samples, silhouette_score 