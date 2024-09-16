import pickle
import pandas as pd
from IPython.display import display

# Load the pickle file
with open('../data/Illustris_Analogs/TNG300-2_119294_analogs_with_histories.pkl', 'rb') as file:
    data = pickle.load(file)

display(pd.DataFrame(data))  # Works well in Jupyter notebooks or IPython environments