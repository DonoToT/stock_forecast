import baostock as bs
import pandas as pd

lg = bs.login()

print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)

rs = bs.query_history_k_data_plus("sh.000001",
    "date,code,open,high,low,close,preclose,volume,amount,pctChg",
    start_date='2023-01-18', end_date='2023-03-07', frequency="d")
print('query_history_k_data_plus respond error_code:'+rs.error_code)
print('query_history_k_data_plus respond  error_msg:'+rs.error_msg)

# 打印结果集
data_list = []
while (rs.error_code == '0') & rs.next():
    # 获取一条记录，将记录合并在一起
    data_list.append(rs.get_row_data())
result = pd.DataFrame(data_list, columns=rs.fields)
# 结果集输出到csv文件
result.to_csv("./dataset_test.csv", index=False)
print(result)

bs.logout()

# lossfn权重
# 训练测试集分开
# 同一个向量每个维度范围统一 归一化   特征值相当   找换手率
# 直接取最后一天收盘价
