import os
import random

import pandas as pd
import numpy as np
from kabuki.hierarchical import Knode
import hddm
import pymc as pm

def gen_rand_str():
    """Generate random strings to append to file names, preventing overwriting"""
    return str(random.randint(1000, 9999))       
