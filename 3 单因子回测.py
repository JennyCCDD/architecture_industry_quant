# -*- coding: utf-8 -*-
# __author__ = "Mengxuan Chen"
# __email__  = "chenmx19@mails.tsinghua.edu.cn"
# __date__   = "20200715"
# @description:
#     single factor test
# @revise log:
#     2020.10.30 for single industry

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr
from getTradingDate import getTradingDateFromJY
from utils import weightmeanFun, basic_data, stock_dif, performance, performance_anl
from datareader import loadData
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def orthogonalize(regressY, regressX):
    # regressY: 因变量数据
    # regressX: 自变量数据

    # 首先给自变量因子加上截距项
    regressX = sm.add_constant(regressX)

    data = pd.concat([regressX, regressY], axis=1)
    data = data.dropna()

    # 我们不能直接用下面的这种形式，因为它丢失了标签栏的很多信息
    # est = sm.OLS(regressY, regressX, missing = 'drop').fit()

    # 注意,iloc这个是不包含
    est = sm.OLS(data.iloc[:, -1], data.iloc[:, 0:2]).fit()

    df = pd.Series(np.nan, regressX.index)
    df[data.index] = est.resid

    return df

class Para():
    startDate = 20091231
    endDate = 20200508
    groupnum = 5
    weightMethod = '简单加权' # 简单加权 市值加权
    ret_calMethod = '简单' # 对数
    normalize = 'Size' # None Size Size_and_Industry
    # factor = 'ROE_DilutedQ'
    # [financial_asset, operating_asset, financial_liability,
    # net_ROE, net_operating_asset_net_profit, leverage_contrib,
    # net_profit_after_tax, net_operaitng_asset_turnover, operating_diff,
    # financail_leverage, g_sustainable, g_implicit
    # ]
    sample = 'out_of_sample' # in_sample out_of_sample
    data_path = '.\\data\\'
    result_path = '.\\result\\'
    listnum = 121 # for stock sample at least be listed for listnum of days
    backtestwindow = 60 # number of days used to form portfolios
    fin_stock = 'no' # include finnacial stock or not
    dataPathPrefix = 'D:\caitong_security'
    pass
para = Para()

class SingleFactor():
    def __init__(self,para,factori):
        # get trading date list as monthly frequancy
        self.tradingDateList = getTradingDateFromJY(para.startDate,
                                                    para.endDate,
                                                    ifTrade=True,
                                                    Period='M')

        self.Price, self.LimitStatus, self.Status, self.listDateNum, self.Industry, self.Size = basic_data(para)
        self.factor = factori
        if self.factor in [
        'Beta252',
        'GPOA',
        'GPOAQ',
        'GrossProfitMargin',
        'GrossProfitMarginQ',
        'NetProfitMargin',
        'NetProfitMarginQ',
        'ROA',
        'ROAQ',
        'ROE_Diluted',
        'ROE_DilutedQ',
        'ROE_ExDiluted',
        'ROE_ExDilutedQ',
        'SUE',
        'SUR',
        'GGPOAQ',
        'GGrossProfitMarginQ',
        'GROAQ',
        'GROEQ',
        'NetOperateCashFlowQYOY',
        'NetProfitQYOY',
        'OperatingRevenueQYOY',
        'BLEV',
        'DTOA',
        'MLEV',
        'AmihudILLIQ',
        'TurnOver_1M',
        'TurnOver_1Y',
        'TurnOver_3M',
        'TurnOver_6M',
        'VSTD_1M',
        'VSTD_3M',
        'VSTD_6M',
        'MaxRet21',
        'MinRet21',
        'Ret21',
        'Ret63',
        'Ret126',
        'Ret252_21',
        'LnNegotiableMV',
        'LnTotalMV',
        'NegotiableMV',
        'NegotiableMVNL',
        'TotalMV',
        'TotalMVNL',
        'IMFFFactorNoAlpha',
        'APBFactor_1M',
        'APBFactor_5D',
        'AssetsTurn',
        'CFO',
        'CurrentRatio',
        'NetProfitCashCover',
        'QualityFactor',
        'QualityIncrease',
        'BP',
        'DividendRatioTTM',
        'EPTTM',
        'NCFPTTM',
        'OCFPTTM',
        ######################### 缺少这个因子的数据 'EPCut',
        'SPTTM',
        'HighLow_1M',
        'HighLow_3M',
        'HighLow_6M',
        'IVFF3_1M',
        'IVFF3_3M',
        'RSquare_1M',
        'RSquare_3M',
        'ResVol',
        'STD_1M',
        'STD_1M_Excess',
        'STD_1Y',
        'STD_1Y_Excess',
        'STD_3M',
        'STD_3M_Excess',
        'STD_6M',
        'STD_6M_Excess'
    ]:
            DATA = loadData(self.factor)
            Factor = DATA.BasicDailyFactorAlpha.loc[para.startDate:para.endDate, :]
            self.Factor = stock_dif(Factor, self.LimitStatus)
        else:
            Factor = pd.read_csv(para.result_path + self.factor + '.csv', index_col=0).T
            self.Factor = stock_dif(Factor, self.LimitStatus)
            self.Factor.fillna(method='ffill', axis=0, inplace=True)
            # self.Factor[self.Factor == np.nan] = 0
        self.df = pd.read_csv(para.data_path + 'whole.csv', index_col=0)


        pass

    def DES(self):
        Des = pd.DataFrame(self.Factor.describe())
        Des['all'] = Des.apply(lambda x: x.sum(), axis = 1)
        return Des['all']

    def every_month(self):
        # deal with the data every month
        self.meanlist= []
        meanlist_abs = []
        self.corr_list = []
        for i,currentDate in enumerate(tqdm(self.tradingDateList[:-2])):
            lastDate = self.tradingDateList[self.tradingDateList.index(currentDate) - 1]
            nextDate = self.tradingDateList[self.tradingDateList.index(currentDate) + 1]
            if para.sample == 'in_sample':
                # use different method to calculate the return
                # logreturn for short time period and simple return calculation for long time period
                if para.ret_calMethod == '对数':
                    self.ret = np.log(self.Price.loc[currentDate, :] / self.Price.loc[lastDate, :])
                elif para.ret_calMethod == '简单':
                    self.ret = self.Price.loc[currentDate, :] / self.Price.loc[lastDate, :] - 1
                self.benchmark = pd.Series([self.df.iloc[i, 0]] * len(self.Factor.columns),
                                           index=self.ret.index.copy())
            elif para.sample == 'out_of_sample':
                if para.ret_calMethod == '对数':
                    self.ret = np.log(self.Price.loc[nextDate, :] / self.Price.loc[currentDate, :])
                elif para.ret_calMethod == '简单':
                    self.ret = self.Price.loc[nextDate, :] / self.Price.loc[currentDate, :] - 1
                self.benchmark = pd.Series([self.df.iloc[i + 1, 0]] * len(self.Factor.columns),
                                           index=self.ret.index.copy())

            self.dataFrame = pd.concat([self.Factor.loc[str(currentDate),:],
                                           self.ret,
                                           self.benchmark,
                                           self.LimitStatus.loc[currentDate,:],
                                           self.Status.loc[currentDate,:],
                                           self.listDateNum.loc[currentDate,:],
                                           self.Industry.loc[currentDate,:],
                                           self.Size.loc[currentDate,:]],
                                           axis=1, sort=True)
            self.dataFrame = self.dataFrame.reset_index()

            self.dataFrame.columns = ['stockid',
                                     'factor',
                                     'RET',
                                     'Bechmark',
                                     'LimitStatus',
                                     'Status',
                                     'listDateNum',
                                     'Industry',
                                     'Size']

            # industry = pd.read_csv(para.data_path + 'Stocks.csv',index_col=0)
            # industry.columns = ['stock_id','sec_name','SEO_class','wind_class','sw_class','zx_class']
            # industry['stockid'] = industry['stock_id']
            # self.dataFrame = pd.merge(self.dataFrame,industry, how = 'outer',on='stockid').dropna(axis=0,subset =['stock_id'])

            if para.normalize == 'Size':
                # 市值中性化
                self.dataFrame['factor'] = orthogonalize(self.dataFrame['factor'],self.dataFrame['Size'])
            elif para.normalize == 'Size_and_Industry':
                # 市值中性化与行业中性化
                dummy_Industry = pd.get_dummies(self.dataFrame['Industry'],prefix = 'Industry')
                X = pd.concat([dummy_Industry,self.dataFrame['Size']],axis = 1, sort = False)
                self.dataFrame['factor'] = orthogonalize(self.dataFrame['factor'],X)
            elif para.normalize == 'None':
                pass
            self.dataFrame = self.dataFrame.dropna()
            # self.dataFrame = self.dataFrame.loc[self.dataFrame['factor'] != 0]
            self.dataFrame = self.dataFrame.loc[self.dataFrame['LimitStatus'] == 0]# 提取非涨跌停的正常交易的数据
            self.dataFrame = self.dataFrame.loc[self.dataFrame['Status'] == 1]# 提取非ST/ST*/退市的正常交易的数据
            self.dataFrame = self.dataFrame.loc[self.dataFrame['listDateNum'] >= para.listnum]# 提取上市天数超过listnum的股票
            if para.fin_stock == 'no': # 非银行金融代号41
                self.dataFrame = self.dataFrame.loc[self.dataFrame['Industry'] != 41]
                self.dataFrame = self.dataFrame.loc[self.dataFrame['Industry'] != 40]

            self.dataFrame['premium'] = self.dataFrame['RET'] - self.dataFrame['Bechmark']
            # 对单因子进行排序打分
            self.dataFrame = self.dataFrame.sort_values(by = 'factor', ascending = False) # 降序排列
            Des = self.dataFrame['factor'].describe()

            ############################################ 计算spearman秩相关系数
            corr, t = spearmanr(
                self.dataFrame.loc[:, 'factor'],
                self.dataFrame.loc[:, 'RET'])
            self.corr_list.append(corr)


            self.dataFrame['Score'] = ''
            eachgroup = int(Des['count']/ para.groupnum)
            for groupi in range(0,para.groupnum-1):
                self.dataFrame.iloc[groupi*eachgroup:(groupi+1)*eachgroup,-1] = groupi+1
            self.dataFrame.iloc[(para.groupnum-1) * eachgroup:, -1] = para.groupnum

            self.dataFrame['premium'] = self.dataFrame['RET'] - self.dataFrame['Bechmark']

            self.dataFrame['Score'].type = np.str
            if para.weightMethod == '简单加权':
                self.meanlist.append(np.array(self.dataFrame.groupby('Score')['premium'].mean()))
                meanlist_abs.append(np.array(self.dataFrame.groupby('Score')['RET'].mean()))
            elif para.weightMethod == '市值加权':
                meanlist_group = []
                meanlist_abs_group = []
                for groupi in range(0,para.groupnum):
                    self.dataFrame_ = self.dataFrame.iloc[groupi * eachgroup:(groupi + 1) * eachgroup, :]
                    meanlist_abs_g = weightmeanFun(self.dataFrame_)
                    self.dataFrame_['RET']=self.dataFrame_['premium'].copy()
                    meanlist_g = weightmeanFun(self.dataFrame_)
                    meanlist_group.append(meanlist_g)
                    meanlist_abs_group.append(meanlist_abs_g)
                self.meanlist.append(meanlist_group)
                meanlist_abs.append(meanlist_abs_group)

        self.meanDf = pd.DataFrame(self.meanlist,index = self.tradingDateList[1:-1])
        self.meanDf_abs = pd.DataFrame(meanlist_abs,index = self.tradingDateList[1:-1])
        self.corr_avg = np.mean(self.corr_list)
        print('RankIC', round(self.corr_avg, 6))
        return self.meanDf, self.meanDf_abs

    def portfolio_test(self, meanDf):
        sharp_list = []
        ret_list = []
        std_list = []
        mdd_list = []
        r2var_list = []
        cr2var_list = []
        anl = []
        compare = pd.DataFrame()
        for oneleg in tqdm(range(len(meanDf.columns))):
            portfolioDF = pd.DataFrame()
            portfolioDF['ret'] = meanDf.iloc[:, oneleg]
            portfolioDF['nav'] = (portfolioDF['ret'] + 1).cumprod()
            performance_df = performance(portfolioDF, para)
            performance_df_anl = performance_anl(portfolioDF, para)
            sharp_list.append(np.array(performance_df.iloc[:, 0].T)[0])
            ret_list.append(np.array(performance_df.iloc[:, 1].T)[0])
            std_list.append(np.array(performance_df.iloc[:, 2].T)[0])
            mdd_list.append(np.array(performance_df.iloc[:, 3].T)[0])
            r2var_list.append(np.array(performance_df.iloc[:, 4].T)[0])
            cr2var_list.append(np.array(performance_df.iloc[:, 5].T)[0])
            anl.append(np.array(performance_df_anl.iloc[:, 0].T))
            compare[str(oneleg)] = portfolioDF['nav']
        performanceDf = pd.concat([pd.Series(sharp_list),
                                   pd.Series(ret_list),
                                   pd.Series(std_list),
                                   pd.Series(mdd_list),
                                   pd.Series(r2var_list),
                                   pd.Series(cr2var_list)],
                                  axis=1, sort=True)
        performanceDf.columns = ['Sharp',
                                 'RetYearly',
                                 'STD',
                                 'MDD',
                                 'R2VaR',
                                 'R2CVaR']
        anlDf = pd.DataFrame(anl)
        print(anlDf)
        compare.index = meanDf.index
        plt.plot(range(len(compare.iloc[1:, 1])),
                 compare.iloc[1:, :])
        plt.title(self.factor)
        plt.xticks([0, 25, 50, 75, 100, 125],
                   ['2009/12/31', '2011/01/31', '2013/02/28', '2015/03/31', '2017/04/30', '2020/04/30'])
        plt.grid(True)
        plt.xlim((0, 125))
        plt.legend()
        plt.savefig(para.result_path + self.factor + '_' + para.weightMethod +
                    '_' + para.normalize + '_performance_nav.png')
        plt.show()
        return performanceDf, compare

if __name__ == "__main__":
#     factorlist = [
#     'Beta252',
#     'GPOA',
#     'GPOAQ',
#     'GrossProfitMargin',
#     'GrossProfitMarginQ',
#     'NetProfitMargin',
#     'NetProfitMarginQ',
#     'ROA',
#     'ROAQ',
#     'ROE_Diluted',
#     'ROE_DilutedQ',
#     'ROE_ExDiluted',
#     'ROE_ExDilutedQ',
#     # 'SUE',  ##################################################
#     # 'SUR',  ##################################################
#     'GGPOAQ',
#     'GGrossProfitMarginQ',
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
#     'RSquare_1M',
#     'RSquare_3M',
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
        # 'financial_asset', 'operating_asset', 'financial_liability',
        # 'net_ROE', 'net_operating_asset_net_profit', 'leverage_contrib',
        'net_profit_after_tax',
        # 'net_operating_asset_turnover', 'operating_diff',
        # 'financail_leverage', 'g_sustainable', 'g_implicit',
        # 'ROE_DilutedQ'
    ]
    for factor_i in factorlist:
        main_fun = SingleFactor(para,factor_i)
        des = main_fun.DES()
        result, result_abs = main_fun.every_month()
        test, test_nav = main_fun.portfolio_test(result)
        test_abs, test_nav_abs = main_fun.portfolio_test(result_abs)
        print(test)
        print(test_abs)
        print(test_nav_abs)
        test_nav_abs.to_csv(para.result_path+'_' + factor_i +'_'+para.weightMethod+
                    '_'+para.normalize+'_test_nav_abs.csv')
        test.to_csv(para.result_path +'_'+ factor_i +'_'+para.weightMethod+
                    '_'+para.normalize+'_performance.csv')
        test_abs.to_csv(para.result_path +'_'+ factor_i +'_'+para.weightMethod+
                    '_'+para.normalize+'_performance_abs.csv')