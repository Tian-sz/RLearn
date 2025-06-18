import json
import pandas as pd

# 读取 txt 文件
with open('res.txt', 'r', encoding='utf-8') as f:
    data = json.load(f)  # 解析 JSON
res_data = data["res"]

df = pd.DataFrame.from_dict(res_data, orient="index")

print()
df.to_excel('./result.xlsx')
print("转换完成！已保存为 result.csv")