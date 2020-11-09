# -*- coding: utf-8 -*-
# @author = "Mengxuan Chen"
# @email  = "chenmx19@mails.tsinghua.edu.cn"
# @description:
#     选出最好的单因子
# @revise log:
#     2020.10.30
import numpy as np
import pandas as pd

class Para():
    data_path = '.\\data\\'
    result_path = '.\\result\\'
    dataPathPrefix = 'D:\caitong_security'
    weightMethod = '简单加权' # 简单加权 市值加权
    ret_calMethod = '简单' # 对数
    normalize = 'Size' # None Size Size_and_Industry
    pass
para = Para()

# In[]
# factorlist = [
#     'Beta252',
#     'GPOA',
#     'GPOAQ',
#     'GrossProfitMargin',
#     'GrossProfitMarginQ',
#     'NetProfitMargin',#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#     'NetProfitMarginQ',#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#     'ROA',
#     'ROAQ',
#     'ROE_Diluted',
#     'ROE_DilutedQ',
#     'ROE_ExDiluted',
#     'ROE_ExDilutedQ',
#     # 'SUE',  ##################################################
#     # 'SUR',  ##################################################
#     'GGPOAQ',
#     'GGrossProfitMarginQ',#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#     'GROAQ',
#     'GROEQ',
#     'NetOperateCashFlowQYOY',
#     'NetProfitQYOY',
#     'OperatingRevenueQYOY',
#     'BLEV',
#     'DTOA',
#     'MLEV',
#     'AmihudILLIQ',
#     'TurnOver_1M',
#     'TurnOver_1Y',
#     'TurnOver_3M',
#     'TurnOver_6M',
#     'VSTD_1M',
#     'VSTD_3M',
#     'VSTD_6M',
#     'MaxRet21',
#     'MinRet21',
#     'Ret21',
#     'Ret63',
#     'Ret126',
#     'Ret252_21',
#     # 'LnNegotiableMV',  ###########################
#     'LnTotalMV',
#     'NegotiableMV',
#     # 'NegotiableMVNL',  ################################
#     'TotalMV',
#     'TotalMVNL',
#     'IMFFFactorNoAlpha',
#     'APBFactor_1M',
#     'APBFactor_5D',
#     'AssetsTurn',
#     'CFO',
#     'CurrentRatio',
#     'NetProfitCashCover',
#     'QualityFactor',
#     # 'QualityIncrease',  ################################
#     'BP',
#     'DividendRatioTTM',
#     'EPTTM',
#     'NCFPTTM',
#     'OCFPTTM',
#     # ######################### 缺少这个因子的数据 'EPCut',
#     'SPTTM',
#     'HighLow_1M',
#     'HighLow_3M',
#     'HighLow_6M',
#     'IVFF3_1M',
#     'IVFF3_3M',
#     'RSquare_1M',#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#     'RSquare_3M',#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#     'ResVol',
#     'STD_1M',
#     'STD_1M_Excess',
#     'STD_1Y',
#     'STD_1Y_Excess',
#     'STD_3M',
#     'STD_3M_Excess',
#     'STD_6M',
#     'STD_6M_Excess'
# ]
factorlist = \
    [
        'financial_asset', 'operating_asset', 'financial_liability',
        'net_ROE', 'net_operating_asset_net_profit', 'leverage_contrib',
        # 'net_profit_after_tax',
        'net_operating_asset_turnover', 'operating_diff',
        'financail_leverage', 'g_sustainable', 'g_implicit',
        # 'ROE_DilutedQ'
    ]
per_sharpe = []
per_ret = []
for factor_i in factorlist:
    per_i = pd.read_csv(para.result_path +'_'+ factor_i +'_'+para.weightMethod+
                    '_'+para.normalize+'_performance_abs.csv')
    per_sharpe.append(np.array(per_i['Sharp']))
    per_ret.append(np.array(per_i['RetYearly']))
per_sharpe_df = pd.DataFrame(per_sharpe,index = factorlist)
per_sharpe_df = per_sharpe_df.sort_values(by = 0)

per_ret_df = pd.DataFrame(per_ret,index = factorlist)
per_ret_df = per_ret_df.sort_values(by = 0)

per_sharpe_df.to_csv(para.result_path+'per_sharpe_df.csv')
per_ret_df.to_csv(para.result_path+'per_ret_df.csv
