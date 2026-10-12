# 通威农发饲料供应链数据分析岗位_作品集

面向通威农发「供应链数据分析师」岗位准备的三个作品：**P1 采购成本监测**、**P2 进厂 NIR 快检**、**P3 旺季生产库存配送优化**。

---

## Code目录结构

```text
portfolio/
├── common/                     # 公共底座
│   ├── build_base.py           #   星型模型（SQLite）与维度表
│   ├── build_docs.py           #   指标字典.xlsx + 假设登记表.xlsx
│   └── formulation.py          #   最低成本配方 LP（PuLP）
├── p1/                         # P1 原料成本 × 鱼价监测
│   ├── collect_futures.py      #   tushare 期货数据采集（断点续传 / 限速重试）
│   ├── build_series.py         #   主力认定、比例复权、周频指标、每周配方重算
│   └── analysis.py             #   敏感度 / 比价低位统计 / 领先滞后 / 套保回测 + 看板图
├── p2/                         # P2 原料进厂 NIR 快检
│   ├── nir_model.py            #   预处理比较、KS 划分、PLS/RF、VIP、20 次重复划分
│   └── nir_qc_transfer.py      #   T²/Q 控制图、异常拦截、仪器转移、省钱测算
├── p3/                         # P3 旺季生产、库存与配送优化
│   ├── demand_network.py       #   需求测算（气温驱动）、运输成本、服务成本地图
│   ├── model_a.py              #   多期生产-库存-运输 LP：四策略 + 影子价格 + 情景
│   └── model_b.py              #   旺季单日派车：直送 vs 片区 vs Clarke-Wright 拼车
├── data/                       # 原始与成品数据
│                               #   raw_futures/、corn.mat、corn_nir.npz、
│                               #   supply_chain.db、维度表 CSV
├── outputs/                    # 指标 CSV、结果 JSON、figures/ 图表、两个 xlsx
├── reports/作品集报告.md        # 主报告（技术口径，含全部数字与图）
└── docs/业务说明文档.md         # 给业务读者的白话版
```

---

## 运行顺序

```bash
# 安装依赖
pip install -r requirements.txt
# pandas numpy matplotlib scipy scikit-learn pulp==2.9.0 tushare openpyxl

# 公共底座
python common/build_base.py         # 建库 + 维度表
python common/build_docs.py         # 指标字典 + 假设登记表

# P1 采购成本监测
python p1/collect_futures.py ALL    # 拉期货数据（约 15 分钟，接口限流，可断点续传）
python p1/build_series.py
python p1/analysis.py

# P2 进厂 NIR 快检
# 数据下载见 p2 注释：http://www.eigenvector.com/data/Corn/corn.mat
# 放入 data/ 后转 npz
python p2/nir_model.py
python p2/nir_qc_transfer.py

# P3 旺季生产、库存与配送优化
python p3/demand_network.py         # 内含 Open-Meteo 气温下载（已缓存为 csv）
python p3/model_a.py
python p3/model_b.py
```

---

## 数据说明

| 数据 | 状态 |
|---|---|
| 豆粕 / 玉米 / 菜粕期货日线 2022-07 ~ 2026-10 | ✅ 真实（tushare 代理接口） |
| 玉米 NIR 光谱与化验值 | ✅ 真实（Eigenvector / Cargill 公开数据集） |
| 成都 2025 日气温 | ✅ 真实（Open-Meteo 历史接口） |
| 现货基差、草鱼批发价、鱼粉周价 | 【模拟】接口 / 版权待核，按历史公开水平模拟并内嵌文档化结构 |
| 工厂名单产能、订单、运输参数 | 【假设 / 模拟】部分来源为本人物流工作经验，详见假设登记表 |

> 所有假设集中在 `outputs/假设登记表.xlsx`（共 21 条），每条含取值、依据、影响范围与灵敏度。

---

## 说明

- 本仓库为个人作品集，代码与文档为原创。
- 所有模拟 / 假设数据均已标注来源与依据，详见 `outputs/假设登记表.xlsx`。
