import pandas as pd
import numpy as np
def read_csv(file):
    data = pd.read_csv(file)
    data = np.array(data)
    data = np.delete(data,0,1)
    return data