# -*- coding: utf-8 -*-

# __author__ = "Mengxuan Chen"
# __email__  = "chenmx19@mails.tsinghua.edu.cn"
# __date__   = "20201030"

import pandas as pd
import numpy as np
class Para():
    data_path = '.\\data\\'
    result_path = '.\\result\\'
    pass
para = Para()
# In[]
SEO_data = pd.read_excel('建筑行业股票池.xlsx',sheet_name='证监会行业建筑类')
SEO_data.columns = ['codes','sec_name','class']
wind_data1 = pd.read_excel('建筑行业股票池.xlsx',sheet_name='wind建材III')
wind_data2 = pd.read_excel('建筑行业股票池.xlsx',sheet_name='wind建筑产品III')
wind_data3 = pd.read_excel('建筑行业股票池.xlsx',sheet_name='wind建筑与工程III')
wind_data4 = pd.read_excel('建筑行业股票池.xlsx',sheet_name='wind建筑机械与重型卡车')
wind_data1.columns = ['codes','sec_name','class']
wind_data2.columns = ['codes','sec_name','class']
wind_data3.columns = ['codes','sec_name','class']
wind_data4.columns = ['codes','sec_name','class']
# In[]
wind_data = pd.concat([wind_data1, wind_data2, wind_data3, wind_data4],axis=0)
sw_data1 = pd.read_excel('建筑行业股票池.xlsx',sheet_name='sw建筑材料')
sw_data2 = pd.read_excel('建筑行业股票池.xlsx',sheet_name='sw建筑装饰')
sw_data1.columns = ['codes','sec_name','class']
sw_data2.columns = ['codes','sec_name','class']
sw_data = pd.concat([sw_data1, sw_data2],axis=0)
# In[]
zx_data1 = pd.read_excel('建筑行业股票池.xlsx',sheet_name='中信建筑')
zx_data2 = pd.read_excel('建筑行业股票池.xlsx',sheet_name='中信建材')
zx_data1.columns = ['codes','sec_name','class']
zx_data2.columns = ['codes','sec_name','class']
zx_data = pd.concat([zx_data1, zx_data2],axis=0)
# In[]
stocks1 = pd.merge(SEO_data,wind_data,how='outer',on=['codes','sec_name'])
stocks1.columns = ['codes','sec_name','SEO_class','wind_class']
stocks2 = pd.merge(stocks1,sw_data,how='outer',on=['codes','sec_name'])
stocks2.columns = ['codes','sec_name','SEO_class','wind_class','sw_class']
stocks = pd.merge(stocks2,zx_data,how='outer',on=['codes','sec_name'])
stocks.columns = ['codes','sec_name','SEO_class','wind_class','sw_class','zx_class']

# In[]
DES = stocks.describe()
seo_class_list = stocks['SEO_class'].drop_duplicates().dropna()
wind_class_list = stocks['wind_class'].drop_duplicates().dropna()
sw_class_list = stocks['sw_class'].drop_duplicates().dropna()
zx_class_list = stocks['zx_class'].drop_duplicates().dropna()
print(seo_class_list,wind_class_list,sw_class_list,zx_class_list)

# In[]
stocks.to_csv(para.data_path + 'Stocks.csv')

