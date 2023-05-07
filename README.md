# JoinBoost: In-Database Tree-Models over Many Tables

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

JoinBoost is a Python library to help you train tree-models (decision trees, gradient boosting, random forests). 

Note that many functionalities of JoinBoost are still under development. If you are interested in using JoinBoost, we are happy to provide direct supports. You can contact us through issues, or email zh2408@columbia.edu

## Why JoinBoost?

JoinBoost algorithms follow LightGBM. However, JoinBoost trains models

1. **Inside Database**: JoinBoost translates ML algorithms into SQL queries, and directly executes these queries in your databases. This means:
    - **Safety**: There is no data movement.
    - **Transformation**: You can directly do OLAP and data transformation in SQL.
    - **Scalability**: In-DB ML is natively out-of-core, and JoinBoost could be connected to distributed databases. 
2. **Many tables**: JoinBoost applies **Factorized Learning** with optimized algorithms. Therefore, JoinBoost trains a model over the join result of many tables but without materializing the join. This provides large performance improvement and is super convenient. 

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


[Please Check out this notebook for Demo](https://colab.research.google.com/github/zachary62/JoinBoost/blob/main/demo/JoinBoostDemo.ipynb)

For dev: https://gitpod.io/new#https://github.com/zachary62/JoinBoost

## Docs

Documentation is currently under development. To build docs locally, download [Sphinx](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html) and run
```bash
make html
```
in the folder `docs`. The docs will be generated in the folder `docs/build/html`.

## Reproducibility

The technical report of JoinBoost could be found under /technical directory.

We note that some optimizations discussed in the paper (e.g., inter-query parallelism, DP) are still under development in the main codes. To reproduce the experiment results from the paper, we include the prototype codes for JoinBoost under /proto folder, which includes all the optimizations. We also include Jupyter Notebook to help you use these codes to train models over Favorita. 

The Favorita dataset is too large to store in Github. Please download files from https://www.dropbox.com/sh/ymwn98pvederw6x/AAC-z6R_rKvU40KZDCyitjsda?dl=0 and uncompress the files. 

The modified DuckDB to support column swap is at https://anonymous.4open.science/r/duckdb-D056.
