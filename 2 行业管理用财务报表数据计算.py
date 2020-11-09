# -*- coding: utf-8 -*-
# __author__ = "Mengxuan Chen"
# __email__  = "chenmx19@mails.tsinghua.edu.cn"
# __date__   = "20201030"
# @description:
#     封装函数
# @revise log:
#     2020.10.30

import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import stock_dif
from getTradingDate import getTradingDateFromJY

class Para():
    data_path = '.\\data\\'
    result_path = '.\\result\\'
    startDate = 20051231
    endDate = 20200430
    dataPathPrefix = 'D:\caitong_security'
    pass
para = Para()

def deal_data(str):
    df = pd.read_csv(para.result_path + '%s'%str+'.csv',index_col=0)
    df.fillna(method = 'ffill', axis = 1, inplace = True)
    df.fillna(0,inplace=True)
    return df

def mgt_bs_asset(total_asset,
            trading_fin_asset,
            sell_fin_asset,
            holding_bond,
            bond_invest,
            other_bond_invest,
            other_equity_invest,
            other_nonliq_invest,
            fv_invest):
    financial_asset = np.array(trading_fin_asset) + np.array(sell_fin_asset) \
                      + np.array(holding_bond) \
                      + np.array(bond_invest) \
                      + np.array(other_bond_invest) \
                      + np.array(other_equity_invest) \
                      + np.array(other_nonliq_invest) \
                      + np.array(fv_invest)
    financial_asset = pd.DataFrame(financial_asset, index=total_asset.index, columns=total_asset.columns)

    operating_asset = np.array(total_asset) - np.array(financial_asset)
    operating_asset = pd.DataFrame(operating_asset, index=total_asset.index, columns=total_asset.columns)
    return financial_asset, operating_asset

def mgt_bs_liability(total_liability,
                     short_debt,
                     long_debt,
                     one_y_nonliq,
                     payable_note,
                     payable_interest,
                     payable_bond):
    financial_liability = np.array(short_debt) \
                          + np.array(long_debt) \
                          + np.array(one_y_nonliq) \
                          + np.array(payable_note) \
                          + np.array(payable_interest) \
                          + np.array(payable_bond)
    financial_liability = pd.DataFrame(financial_liability, index=total_liability.index,
                                       columns=total_liability.columns)
    operating_liability = np.array(total_liability) - np.array(financial_liability)
    operating_liability = pd.DataFrame(operating_liability, index=total_liability.index,
                                       columns=total_liability.columns)
    return financial_liability, operating_liability

def int_cost(total_liability, interest, interest_out, interest_cap, interest_in):
    financial_loss = np.array(interest_out) - np.array(interest_cap) - np.array(interest_in)
    financial_loss = pd.DataFrame(financial_loss, index= total_liability.index, columns=total_liability.columns)
    financial_loss[financial_loss == np.nan] = interest
    return financial_loss

def cal_mgt(financial_liability,
            financial_asset,
            financial_loss,
            operating_asset,
            operating_liability,
            tax_expenditure,
            profit_all,
            EBIT,
            equity,
            sales):

    # 净金融负债 = 金融负债 − 金融资产
    net_financial_liability = np.array(financial_liability) - np.array(financial_asset)
    net_financial_liability = pd.DataFrame(net_financial_liability,
                                           index=financial_liability.index,
                                           columns=financial_liability.columns)
    # 税后利息率 = 税后利息费用 ÷ 净金融负债
    interest_rate_after_tax = np.array(financial_loss) / (np.array(net_financial_liability) + 0.000000001)
    interest_rate_after_tax = pd.DataFrame(interest_rate_after_tax,
                                           index=financial_loss.index,
                                           columns=financial_loss.columns)
    # 净经营资产 = 经营资产 − 经营负债 = 净金融负债 + 股东权益
    net_operating_asset = np.array(operating_asset) - np.array(operating_liability)
    net_operating_asset = pd.DataFrame(net_operating_asset,
                                       index=operating_asset.index,
                                       columns=operating_asset.columns)

    # 所得税率 = 所得税费用/利润总额
    tax_rate = np.array(tax_expenditure) / np.array(profit_all)
    tax_rate = pd.DataFrame(tax_rate,
                            index=profit_all.index,
                            columns=profit_all.columns)

    # 税后经营净利润 = 息税前利润 × （1 − 所得税率）
    EBI = np.array(EBIT) * (1 - np.array(tax_rate))
    EBI = pd.DataFrame(EBI,
                       index=EBIT.index,
                       columns=EBIT.columns)

    # 净财务杠杆 = 净金融负债 ÷ 股东权益
    financail_leverage = np.array(net_financial_liability) / np.array(equity)
    financail_leverage = pd.DataFrame(financail_leverage,
                                      index=equity.index,
                                      columns=equity.columns)
    # 经营差异率 = 净经营资产净利率 − 税后利息率
    operating_diff = np.array(financail_leverage) - np.array(interest_rate_after_tax)
    operating_diff = pd.DataFrame(operating_diff,
                                  index=financail_leverage.index,
                                  columns=financail_leverage.columns)
    # 净经营资产周转次数 = 销售收入 ÷ 净经营资产
    net_operating_asset_turnover = np.array(sales) / np.array(net_operating_asset)
    net_operating_asset_turnover = pd.DataFrame(net_operating_asset_turnover,
                                                index=sales.index,
                                                columns=sales.columns)
    # 税后经营净利率 = 税后经营净利润 ÷ 销售收入
    net_profit_after_tax = np.array(EBI) / np.array(sales)
    net_profit_after_tax = pd.DataFrame(net_profit_after_tax,
                                        index=EBI.index,
                                        columns=EBI.columns)

    # 杠杆贡献率 = 经营差异率 × 净财务杠杆
    leverage_contrib = np.array(operating_diff) * np.array(financail_leverage)
    leverage_contrib = pd.DataFrame(leverage_contrib,
                                    index=operating_diff.index,
                                    columns=operating_diff.columns)

    # 净经营资产净利率 = 税后经营净利率 × 净经营资产周转次数
    net_operating_asset_net_profit = np.array(net_profit_after_tax) * np.array(net_operating_asset_turnover)
    net_operating_asset_net_profit = pd.DataFrame(net_operating_asset_net_profit,
                                                  index=net_profit_after_tax.index,
                                                  columns=net_profit_after_tax.columns)
    # 净资产收益率 = （税后经营净利润 − 税后利息费用） ÷ 股东权益
    # = 净经营资产净利率 + 杠杆贡献率
    net_ROE = np.array(net_operating_asset_net_profit) + np.array(leverage_contrib)
    net_ROE[np.isinf(net_ROE)] = 0
    net_ROE = pd.DataFrame(net_ROE,
                           index=net_operating_asset_net_profit.index,
                           columns=net_operating_asset_net_profit.columns)

    return financial_asset, operating_asset, financial_liability,\
            net_ROE, net_operating_asset_net_profit, leverage_contrib,\
            net_profit_after_tax, net_operating_asset_turnover, operating_diff,\
            financail_leverage



def conPre(net_profit):
    # 预计经营资产负债
    # 各项经营资产（或负债）=预计营业收入×各项目销售百分比
    # 预计营业收入, 一致预期数据, 单位百万元
    conPreSales = pd.read_hdf(para.dataPathPrefix +
                              '\DataBase\Data_AShareConsensusData\Data_Consensus\Data_StockForecast\Data\BasicDailyFactor_Stock_con_or.h5')
    # 一致预期净利润, 单位百万元
    conPreNetprofit = pd.read_hdf(para.dataPathPrefix +
                                  '\DataBase\Data_AShareConsensusData\Data_Consensus\Data_StockForecast\Data\BasicDailyFactor_Stock_con_np.h5')
    # 一致预期净资产， 单位百万元
    conPreNetAsset = pd.read_hdf(para.dataPathPrefix +
                                 '\DataBase\Data_AShareConsensusData\Data_Consensus\Data_StockForecast\Data\BasicDailyFactor_Stock_con_na.h5')

    PreSales_M = []
    PreNetprofit_M = []
    PreNetasset_M = []
    tradingDateList = getTradingDateFromJY(para.startDate, para.endDate, ifTrade=True, Period='M')
    for i, currentDate in enumerate(tqdm(tradingDateList)):
        PreSales_M.append(conPreSales.loc[currentDate, :])
        PreNetprofit_M.append(conPreNetprofit.loc[currentDate, :])
        PreNetasset_M.append(conPreNetAsset.loc[currentDate, :])
    # conPreSales_M
    conPreSales_M = pd.DataFrame(PreSales_M,
                                 columns=conPreSales.columns.copy(),
                                 index=tradingDateList)
    conPreSales_M = conPreSales_M.loc[str(para.startDate):str(para.endDate), :]
    conPreSales_M = stock_dif(conPreSales_M, net_profit.T).T
    # conPreNetprofit_M
    conPreNetprofit_M = pd.DataFrame(PreNetprofit_M,
                                     columns=conPreSales.columns.copy(),
                                     index=tradingDateList)
    conPreNetprofit_M = conPreNetprofit_M.loc[str(para.startDate):str(para.endDate), :]
    conPreNetprofit_M = stock_dif(conPreNetprofit_M, net_profit.T).T
    # conPreNetAsset_M
    conPreNetAsset_M = pd.DataFrame(PreNetasset_M,
                                    columns=conPreSales.columns.copy(),
                                    index=tradingDateList)
    conPreNetAsset_M = conPreNetAsset_M.loc[str(para.startDate):str(para.endDate), :]
    conPreNetAsset_M = stock_dif(conPreNetAsset_M, net_profit.T).T

    return conPreSales_M, conPreNetAsset_M

def g_sustainable(conPreSales_M, conPreNetAsset_M, equity, asset, dividend_rate, net_profit):
    g_sustainable = (np.array(conPreSales_M.iloc[:, :-1]) / np.array(conPreNetAsset_M.iloc[:, :-1])) \
                    * (np.array(equity.iloc[:, :-1]) / np.array(asset)) \
                    * (1 - np.array(dividend_rate))
    tradingDateList = getTradingDateFromJY(para.startDate, para.endDate, ifTrade=True, Period='M')
    g_sustainable = pd.DataFrame(g_sustainable,
                                 index=net_profit.index,
                                 columns=tradingDateList[1:])

    return g_sustainable

def g_implicit(operating_asset_sales,
               operating_liability_sales,
               net_profit,
               conPreNetprofit_M,
               conPreSales_M,
               dividend_rate,
               ):
    numerator = np.array(operating_asset_sales.iloc[:, :-1]) - np.array(operating_liability_sales.iloc[:, :-1])
    sales = pd.read_csv(para.result_path + '营业总收入.csv', index_col=0)
    sales.fillna(method='ffill', axis=0, inplace=True)
    sales = stock_dif(sales.T, net_profit.T).T
    sales = sales.loc[:, str(para.startDate):str(para.endDate)]
    # 分析师一致预期净利率=分析师一致预期净利润/分析师一致预期营业收入
    conPreNetprofitrate = np.array(conPreNetprofit_M.iloc[:, :-1]) / np.array(conPreSales_M.iloc[:, :-1])
    denominator = np.array(conPreNetprofitrate) * (1 - np.array(dividend_rate))
    g_implicit = denominator / (numerator - denominator)
    tradingDateList = getTradingDateFromJY(para.startDate, para.endDate, ifTrade=True, Period='M')
    g_implicit = pd.DataFrame(g_implicit,
                              index=net_profit.index,
                              columns=tradingDateList[1:])
    return g_implicit


# if __name__ == "__main__":
    # for i in ['totsl_asset',
    #           'trading_fin_asset',
    #           'sell_fin_asset',
    #           'holding_bond',
    #           'bond_invest',
    #           'other_bond_invest',
    #           'other_equity_invest',
    #           'other_nonliq_invest',
    #           'fv_invest',
    #           '']:
    #     locals()['a' + str(i)] = deal_data(i)
