# Taobao Ad CTR Prediction — Project Brief

## 项目背景

本项目使用阿里巴巴天池平台提供的淘宝展示广告点击率预估数据集（Ali_Display_Ad_Click），目标是通过用户历史行为和广告特征，分析并最终预测用户在看到某个广告时是否会点击（CTR 预估）。

数据集来源：https://tianchi.aliyun.com/dataset/56

---

## 数据集说明

原始数据共 4 个 CSV 文件，放置于项目根目录：

### 1. `raw_sample.csv`（~1.1 GB）
广告曝光与点击日志，核心事实表。

| 字段 | 类型 | 说明 |
|------|------|------|
| user | int | 脱敏用户 ID |
| time_stamp | int | Unix 时间戳 |
| adgroup_id | int | 脱敏广告单元 ID |
| pid | str | 资源位（广告展示位置），只有两个值 |
| nonclk | int8 | 1=未点击，0=点击 |
| clk | int8 | 1=点击，0=未点击（与 nonclk 互补） |

- 共约 2,656 万条记录
- 时间范围：2017-05-06 ~ 2017-05-13（8天，前7天训练，最后1天测试）
- 整体 CTR ≈ 5.14%

### 2. `ad_feature.csv`（~30 MB）
广告静态属性表。

| 字段 | 类型 | 说明 |
|------|------|------|
| adgroup_id | int | 广告 ID（主键） |
| cate_id | int | 商品类目 ID |
| campaign_id | int | 广告计划 ID |
| customer | int | 广告主 ID |
| brand | float | 品牌 ID（有缺失） |
| price | float | 商品价格 |

- 共约 84.7 万条记录
- `brand` 缺失约 29%

### 3. `user_profile.csv`（~23 MB）
用户人口统计特征表。

| 字段 | 类型 | 说明 |
|------|------|------|
| userid | int | 用户 ID（主键） |
| cms_segid | int | 微群 ID |
| cms_group_id | int | 用户组 ID |
| final_gender_code | int8 | 性别（1=男，2=女） |
| age_level | int8 | 年龄层（0~6） |
| pvalue_level | float | 消费档次（1=低，2=中，3=高；**缺失率 54%**） |
| shopping_level | float | 购物深度（1=浅层，2=中度，3=深度） |
| occupation | float | 是否大学生（1=是，0=否） |
| new_user_class_level | float | 城市层级（缺失率 32%） |

- 共约 106 万条记录

### 4. `behavior_log.csv`（**~23 GB**）
用户历史行为日志，**最大也最重要的表**。

| 字段 | 类型 | 说明 |
|------|------|------|
| user | int | 用户 ID |
| time_stamp | int | Unix 时间戳 |
| btag | str | 行为类型：`pv`=浏览，`cart`=加购，`fav`=收藏，`buy`=购买 |
| cate | int | 商品类目 ID |
| brand | float | 品牌 ID |

- 原始约 7.23 亿条记录
- 涵盖 raw_sample 全部用户的 22 天历史行为（约 2017-04-14 ~ 2017-05-13）
- **包含大量重复行**：不同部门的日志系统打包时会产生完全一致的重复记录，需清洗
- **包含少量异常时间戳**：负数或极远未来，占比 < 0.01%，需过滤

### 表间关系
```
raw_sample.user       → user_profile.userid
raw_sample.adgroup_id → ad_feature.adgroup_id
behavior_log.user     → user_profile.userid
```

---

## 硬件约束（重要）

运行环境内存仅约 **3.8 GB**，必须严格控制内存使用：

- `behavior_log.csv` 23 GB，**绝对不能一次性加载**，必须分块处理
- `raw_sample.csv` 1.1 GB，加载时须使用 dtype 优化（int32/int8/category）
- 每张表处理完后立即 `del` + `gc.collect()`
- 各表分析脚本应独立运行，避免多张大表同时驻留内存

---

## 项目结构（建议）

```
TaoBao-Project/
├── data/                        # 原始 CSV 文件（不纳入 git）
│   ├── raw_sample.csv
│   ├── ad_feature.csv
│   ├── user_profile.csv
│   └── behavior_log.csv
├── data/processed/              # 清洗后的中间数据
│   ├── behavior_log_parquet/    # behavior_log 的 parquet 分块
│   └── user_behavior_stats.csv  # 用户行为聚合统计
├── notebooks/                   # 可选：Jupyter notebooks
├── src/                         # 结构化 Python 代码
│   ├── 01_clean_ad_feature.py
│   ├── 02_clean_user_profile.py
│   ├── 03_clean_raw_sample.py
│   ├── 04_clean_behavior_log.py
│   ├── 05_aggregate_behavior.py
│   └── 06_hypothesis_verification.py
├── outputs/
│   ├── plots/                   # 所有图表
│   └── stats/                   # JSON 统计摘要
├── PROJECT_BRIEF.md             # 本文件
├── dataset_description.md       # 数据集字段说明
├── requirements.txt
└── README.md
```

---

## 第一阶段任务：EDA 与数据清洗

### 每张表统一要完成的内容

1. **加载**：使用合理的 dtype 减少内存占用
2. **空值分析**：每列空值数量与百分比
3. **去重**：`drop_duplicates()`，记录删除数量
4. **取值合理性检查**：分类变量的合法取值，数值变量的范围
5. **Histogram**：每个有意义的列画分布图
6. **Correlation heatmap**：数值列之间的相关性矩阵
7. **统计摘要**：保存为 JSON 文件供后续引用

### behavior_log 的特殊处理

由于文件 23 GB，采用以下策略：

**分块读取**（`chunksize=2_000_000`）：
- 每块做：精确去重（5列全同才算重复）、时间戳过滤
- 时间戳过滤规则：保留 `1_483_228_800 < time_stamp < 1_577_836_800`（即 2017-01-01 ~ 2020-01-01）
- 每块转存为 parquet（snappy 压缩），压缩比约 5:1（23 GB → ~4.9 GB）
- 逐块累计统计：btag 分布、空值计数、用户行为计数

**关于重复行的判断**（重要背景知识）：
- 5列完全相同 → 日志系统重复打点，**应删除**（占约 7.1%，约 5100 万行）
- `(user, btag, cate, brand)` 相同但 `time_stamp` 不同 → 用户在不同时间的真实重复行为，**应保留**
- `(user, time_stamp)` 相同但其他列不同 → 同一秒内的不同行为，**应保留**

**用户行为聚合**（为后续建模准备）：

清洗完成后，按用户聚合，生成 `user_behavior_stats.csv`，字段如下：

| 字段 | 说明 |
|------|------|
| user | 用户 ID |
| pv_count | 历史浏览次数 |
| cart_count | 历史加购次数 |
| fav_count | 历史收藏次数 |
| buy_count | 历史购买次数 |
| total_actions | 四类行为总和 |
| has_purchase | 是否有购买记录（0/1） |
| buy_rate | buy_count / pv_count |
| cart_rate | cart_count / pv_count |

---

## 第二阶段任务：假设验证

基于清洗后的数据，验证以下 5 个假设（每个假设对应一张图表，最终汇总为一张 6-panel 图）。

验证方法：将 raw_sample 按用户聚合得到每个用户的 CTR，再与 user_profile、user_behavior_stats、ad_feature 做 join，按分组计算加权 CTR（= 总点击数 / 总曝光数）。

### H1：购物深度越高的用户，CTR 越高
- 变量：`shopping_level`（1/2/3）
- 预期：深度用户 CTR 更高
- **实际结果：反转**，浅层用户 CTR 最高（5.41% > 5.17% > 5.11%）

### H2：有历史购买记录的用户，CTR 更高
- 变量：`has_purchase`（0/1，来自 user_behavior_stats）
- 预期：has_purchase=1 的用户 CTR 更高
- **实际结果：反转**，无购买记录用户 CTR 反而略高（5.38% vs 5.12%）

### H3：广告商品价格越低，CTR 越高
- 变量：`price`（来自 ad_feature），按五分位数分箱
- 预期：低价广告 CTR 更高
- **实际结果：成立**，单调递减（6.12% → 4.68%）

### H4：女性用户 CTR 高于男性
- 变量：`final_gender_code`（来自 user_profile）
- 预期：gender=2 CTR 更高
- **实际结果：成立**（5.24% vs 4.84%，+8.4%）

### H5：历史行为越活跃的用户，CTR 越高
- 变量：`total_actions`（来自 user_behavior_stats），按五分位数分箱
- 预期：活跃度越高 CTR 越高
- **实际结果：非常显著**，从 3.76% 到 5.77%（+53%）

### Bonus：年龄层次与 CTR 的关系
- 变量：`age_level`（0~6）
- 结果：呈 U 型，青少年（level 6）和老年（level 1）CTR 最高，中年（level 4）最低

---

## 代码规范要求

1. **每个脚本顶部**写清楚该脚本的目的、输入文件、输出文件
2. **关键步骤**写注释，说明"为什么"而不只是"做了什么"
3. **所有路径**使用相对路径或在脚本顶部统一定义 `BASE_DIR`
4. **每个脚本可独立运行**，不依赖其他脚本的内存状态（通过读取中间文件传递数据）
5. **图表统一保存**到 `outputs/plots/`，统计摘要保存到 `outputs/stats/`（JSON 格式）
6. **内存敏感操作**处理完后立即释放：`del df; gc.collect()`

---

## Git Commit 建议

建议按以下顺序提交，保持清晰的 commit history：

```
feat: init project structure and add requirements
feat: add dataset description and project brief
feat: EDA and cleaning for ad_feature
feat: EDA and cleaning for user_profile  
feat: EDA and cleaning for raw_sample
feat: chunked cleaning and parquet conversion for behavior_log
feat: aggregate user behavior stats from behavior_log
feat: hypothesis verification H1-H5 with visualization
```

---

## 依赖

```
pandas>=2.0
numpy
pyarrow          # parquet 读写
matplotlib
seaborn
```

---

## 完整项目 Pipeline

以下是项目从数据到模型的完整流程，第一、二阶段已完成，三至五阶段待实现。

```
raw_sample + behavior_log
        │
        ▼
[阶段一] EDA & 数据清洗          ← 已完成
        │
        ▼
[阶段二] 假设验证                ← 已完成
        │
        ▼
[阶段三] 构建交互矩阵 & 矩阵分解  ← 下一步
        │
        ▼
[阶段四] 特征增强（Enrich）
        │
        ▼
[阶段五] CTR 预测模型（Logistic Regression）
        │
        ▼
        评估（AUC，baseline = 0.622）
```

---

## 第三阶段任务：静态分析（矩阵分解）

### 目标

从 `raw_sample` 的点击日志出发，通过 SVD 矩阵分解，为每个用户和每个广告学习一个低维的隐向量表示（embedding），作为后续 CTR 预测的核心特征。

### 步骤一：构建用户-广告交互矩阵 X̃

- 行：用户（user_id），列：广告（adgroup_id），值：点击（1）或未点击（0）
- 矩阵极度稀疏：114万用户 × 84万广告，但每个用户只与极少数广告有交互
- 只使用训练集（2017-05-06 ~ 2017-05-12，前7天），保留测试集（05-13）做评估
- 由于矩阵过大，使用 `scipy.sparse.csr_matrix` 稀疏格式存储

### 步骤二：SVD 矩阵分解

对稀疏矩阵 X̃ 做截断 SVD（Truncated SVD）：

```
X̃ ≈ Ũ · Ṽ + ε

其中：
  Ũ (用户矩阵)：shape = (n_users, k)，每行是一个用户的 k 维隐向量
  Ṽ (广告矩阵)：shape = (k, n_ads)，每列是一个广告的 k 维隐向量
  k：低秩维度（超参数，建议从 k=32 或 k=64 开始试验）
  ε：重建误差
```

完整 SVD 为 X̃ = UΣV^T，截断后取前 k 个奇异值：
- Ũ = U_(uxk) · √Σ_(kxk)
- Ṽ = √Σ_(kxk) · V_(kxa)

使用 `sklearn.decomposition.TruncatedSVD` 实现（专为稀疏矩阵设计，不需要将矩阵稠密化）。

### 步骤三：保存 embedding

- `user_embeddings.csv`：user_id + k 维隐向量
- `ad_embeddings.csv`：adgroup_id + k 维隐向量

---

## 第四阶段任务：特征增强（Enrich）

将 SVD 得到的隐向量与原始特征拼接，构建最终的建模特征矩阵。

### 用户特征（user features）

| 来源 | 字段 |
|------|------|
| SVD embedding | k 维隐向量（Ũ 的对应行） |
| user_profile | gender, age_level, shopping_level, occupation, new_user_class_level |
| user_behavior_stats | pv_count, cart_count, fav_count, buy_count, buy_rate, cart_rate |

注意：`pvalue_level` 缺失 54%，需要决策——建议加一列 `pvalue_known`（0/1）作为缺失指示变量，缺失值填充为 0。

### 广告特征（ad features）

| 来源 | 字段 |
|------|------|
| SVD embedding | k 维隐向量（Ṽ 的对应列） |
| ad_feature | cate_id, price（建议取 log），brand（缺失 29%，同样加指示变量） |

### 最终样本构建

以 `raw_sample` 的每一行（一次曝光）为一个样本，拼接对应的用户特征和广告特征：

```
样本 = [user_embedding(k维) | user_profile特征 | user_behavior特征
       | ad_embedding(k维)  | ad_feature特征]
标签 = clk（0或1）
```

---

## 第五阶段任务：CTR 预测模型

### Baseline：Logistic Regression

- 输入：第四阶段构建的特征矩阵
- 输出：点击概率（0~1）
- 评估指标：**AUC**（数据集官方 baseline = 0.622）
- 注意正负样本不均衡（CTR ≈ 5%，约 19:1），训练时考虑 `class_weight='balanced'`

### 进阶方向（可选）

导师笔记中提到了 **Factorization Machine（FM）**，是这个方向的自然进阶：
- FM 在 LR 基础上自动学习特征两两交叉，不需要手动构造交叉特征
- 特别适合广告 CTR 这类高维稀疏特征场景
- 可使用 `lightfm` 或 `xlearn` 库实现

---

## Git Commit 建议（完整）

```
feat: init project structure and add requirements
feat: add dataset description and project brief
feat: EDA and cleaning for ad_feature
feat: EDA and cleaning for user_profile
feat: EDA and cleaning for raw_sample
feat: chunked cleaning and parquet conversion for behavior_log
feat: aggregate user behavior stats from behavior_log
feat: hypothesis verification H1-H5 with visualization
feat: build sparse user-ad interaction matrix from raw_sample
feat: SVD matrix factorization, extract user and ad embeddings
feat: enrich features with user_profile and ad_feature
feat: build final training dataset with joined features
feat: logistic regression CTR model, evaluate AUC
```

---

## 依赖

```
pandas>=2.0
numpy
scipy          # 稀疏矩阵
scikit-learn   # TruncatedSVD, LogisticRegression
pyarrow        # parquet 读写
matplotlib
seaborn
```
