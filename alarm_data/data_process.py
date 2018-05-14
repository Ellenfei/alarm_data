import pandas as pd
import numpy as np
from alarm_data_type import alarm_dict

filename = 'raw_data.xls'
table = pd.read_excel(filename, sheet_name=0, header=0)
datas = []
for i in range(len(table)):
    data = []
    data.append(alarm_dict.level[table.iloc[i].values[0]])
    data.append(alarm_dict.event[table.iloc[i].values[1]])
    data.append(alarm_dict.alarm_source[table.iloc[i].values[2]])
    data.append(alarm_dict.location[table.iloc[i].values[3]])
    data.append(alarm_dict.occur_time[table.iloc[i].values[4]])
    data.append(table.iloc[i].values[7])
    datas.append(data)
arr = np.array(datas)
print(arr[:, 0:1])