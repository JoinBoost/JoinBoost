# JoinBoost: Grow Trees in Databases Using Only SQL

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

JoinBoost is a Python library to help you train tree-models (decision trees, gradient boosting, random forests). 

[Please Check out this notebook for Demo on DuckDB](https://colab.research.google.com/github/zachary62/JoinBoost/blob/main/demo/JoinBoostDemo.ipynb)

## Why JoinBoost?

JoinBoost algorithms follow LightGBM. However, JoinBoost trains models

1. **Inside Database**: JoinBoost translates ML algorithms into SQL queries, and directly executes these queries in your databases. This means:
    - **Safety**: There is no data movement.
    - **Transformation**: You can directly do OLAP and data transformation in SQL.
    - **Scalability**: In-DB ML is natively out-of-core, and JoinBoost could be connected to distributed databases. 
2. **Optimized for normalized tables**: JoinBoost applies **Factorized Learning** as query rewriting. Therefore, JoinBoost trains a model over the join result of many tables but without materializing the join. This provides large performance improvement and is super convenient. 

## Start JoinBoost

The easiest way to install JoinBoost is using pip:

```
pip install joinboost
```

JoinBoost APIs are similar to Sklearn, Xgboost and LightGBM. The main difference is that JoinBoost datasets are specified by database connector and join graph schema. Below, we specify a join graph of two tables sales and items:

```
import duckdb
from joinboost.joingraph import JoinGraph
from joinboost.app import DecisionTree

# DuckDB connector
con = duckdb.connect(database='duckdb')

dataset = JoinGraph(con)
dataset.add_relation("sales", [], y = 'total_sales')
dataset.add_relation("items", ["family","class","perishable"])
dataset.add_join("sales", "items", ["item_nbr"], ["item_nbr"])

reg = DecisionTree(learning_rate=1, max_leaves=8)
reg.fit(dataset)
```


[Demo on Spark](https://colab.research.google.com/github/zachary62/JoinBoost/blob/main/demo/JoinBoostSparkDemo.ipynb)

[Demo on CUDF](https://www.kaggle.com/zacharyhuang/joinboost-gpu-demo)

[Gitpod for code development]( https://gitpod.io/new#https://github.com/zachary62/JoinBoost)

## Paper and contribution

JoinBoost research paper has been accepted at [VLDB 2023](https://dl.acm.org/doi/10.14778/3611479.3611509)

```
@article{10.14778/3611479.3611509,
  author = {Huang, Zezhou and Sen, Rathijit and Liu, Jiaxiang and Wu, Eugene},
  title = {JoinBoost: Grow Trees over Normalized Data Using Only SQL},
  year = {2023},
  issue_date = {July 2023},
  publisher = {VLDB Endowment},
  volume = {16},
  number = {11},
  issn = {2150-8097},
  url = {https://doi.org/10.14778/3611479.3611509},
  doi = {10.14778/3611479.3611509},
  month = {aug},
  pages = {3071–3084},
  numpages = {14}
}
```

Upon the completion of the JoinBoost research project, we have made the code open source. We're deeply appreciative of the contributions to the open source project by Pavan Kalyan Damalapati (for improving the join graph's reliability and its extension to pandas), Weisheng Wang (for enhancing the prediction function's capabilities), and Matthew Schoenbauer (for documentation).
