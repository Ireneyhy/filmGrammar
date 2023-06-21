import pandas as pd
import numpy as np

s = pd.Series([1, 3, 5, np.nan, 6, 8])
print(s)

data = {
    "Name": ["John", "Anna", "Peter", "Linda"],
    "Age": [28, 24, 22, 32],
    "City": ["New York", "Paris", "Berlin", "London"],
}
df = pd.DataFrame(data)
print(df)
