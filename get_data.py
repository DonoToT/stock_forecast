import baostock as bs
import pandas as pd

lg = bs.login()

print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)

rs = bs.query_history_k_data_plus("sh.600000",
    "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
    start_date='2022-03-09', end_date='2023-03-09',
    frequency="d", adjustflag="3")
print('query_history_k_data_plus respond error_code:'+rs.error_code)
print('query_history_k_data_plus respond  error_msg:'+rs.error_msg)

# 打印结果集
data_list = []
while (rs.error_code == '0') & rs.next():
    # 获取一条记录，将记录合并在一起
    data_list.append(rs.get_row_data())
result = pd.DataFrame(data_list, columns=rs.fields)
# 结果集输出到csv文件
result.to_csv("./dataset/dataset_test.csv", index=False)
print(result)

bs.logout()

# lossfn权重    √
# 训练测试集分开   √
# 同一个向量每个维度范围统一 归一化   特征值相当   找换手率    √
# 直接取最后一天收盘价    √
# 股票中的涨跌幅=（现价-上一个交易日的收盘价）/上一个交易日的收盘价×10    √
# pctChg = (close - preclose) / preclose    √


# 试试3天
# 看看概率高的信息量, 修改一下 比如70%有多少, 正确率    √
# 错的会不会很影响    √
# 设计loss如何调整, 主要颠覆性错误
# 变成回归


# 500次测试之后，你按照他给的建议去买，盈亏情况如何    √

# 为了防止出现极端错误，Loss可以在原来基础上，再加一项：就是按照估计出来的各个类别的概率，去算一下类别中心和GT值的距离平方，然后按照估计出来的概率去加权
