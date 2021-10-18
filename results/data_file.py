import os
import numpy as np
import pandas as pd

roots = []

for i in range(1, 108):
    p = "p{}/".format(i)

    for v in os.listdir(p):
        v = os.path.join(p,v)
        v +="/"
        for dir in os.listdir(v):
            path = os.path.join(v, dir)
            path+="/"
            roots.append(path)


hr = []

for root in roots:
    hr_file = os.path.join(root, "gt_HR.csv")
    data = pd.read_csv(hr_file)
    hr.append(int(np.mean(data['HR'])))

data_file = pd.DataFrame({"root": roots, "mean_hr": hr})

data_file.to_csv("data.csv", index=False, sep=',')