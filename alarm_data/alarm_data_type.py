# 导入excel
import pandas as pd
class alarm_dict:
    filename = 'raw_data.xls'
    table = pd.read_excel(filename, sheet_name=0, header=0)
    # 编号
    ID = 1

    # 告警级别字典
    level = {}
    level_keys = table['级别'].value_counts().keys()
    for key in level_keys:
        level[key] = ID
        ID += 1

    # 告警事件字典
    event = {}
    event_keys = table['名称'].value_counts().keys()
    for key in event_keys:
        event[key] = ID
        ID += 1

    # 告警源字典
    alarm_source = {}
    alarm_source_keys = table['告警源'].value_counts().keys()
    for key in alarm_source_keys:
        alarm_source[key] = ID
        ID += 1

    # 定位信息字典
    location = {}
    location_keys = table['定位信息'].value_counts().keys()
    for key in location_keys:
        location[key] = ID
        ID += 1

    # 发生时间字典
    occur_time = {}
    occur_time_keys = table['发生时间'].value_counts().keys()
    for key in occur_time_keys:
        occur_time[key] = ID
        ID += 1
# print(alarm_dict.level['紧急'])