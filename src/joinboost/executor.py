import re
import time
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Any, List
import types

import pandas as pd

from .aggregator import *
from .mini_joingraph import MiniJoinGraph
import numpy as np

ExecuteMode = Enum(
    "ExecuteMode", ["WRITE_TO_TABLE", "CREATE_VIEW", "EXECUTE", "NESTED_QUERY"]
)


class ExecutorException(Exception):
    pass


@dataclass(frozen=True)
class SPJAData:
    """
    Data structure for SPJA queries. Could be recursive (e.g, from_tables could be a list of SPJAData objects).
    Attributes:
        aggregate_expressions: dict mapping column names to tuples containing the aggregation expression and the aggregator object.
        from_tables: list of table names to select from.
        select_conds: list of conditions to apply to the SELECT statement.
        join_conds: list of conditions of the form "table1.col1 IS NOT DISTINCT FROM table2.col2".
        group_by: list of column names to group by.
        window_by: list of column names to use for windowing.
        order_by: list of columns to use for ordering the results.
        limit: maximum number of rows to return.
        sample_rate: sampling rate to use for the query.
        replace: if True, replaces an existing table or view with the same name.
        join_type: type of join to use for the query.
    """
    aggregate_expressions: dict = field(
        default_factory=lambda: {None: AggExpression(Aggregator.IDENTITY, "*")}
    )
    from_tables: List[str] = field(default_factory=list)
    select_conds: List[SelectionExpression] = field(default_factory=list)
    join_conds: List[SelectionExpression] = field(default_factory=list)
    group_by: List[QualifiedAttribute] = field(default_factory=list)
    window_by: List[QualifiedAttribute] = field(default_factory=list)
    order_by: List[QualifiedAttribute] = field(default_factory=list)
    limit: Optional[int] = None
    sample_rate: Optional[float] = None
    replace: bool = True
    join_type: str = "INNER"
    qualified: bool = True


def ExecutorFactory(con=None):
    """
    Factory function to create and return Executor objects for different connectors.

    Parameters
    ----------
    con : DuckDBPyConnection or Executor, optional
        The connector to use for creating Executor objects.
        By default, if `con` is not specified, the function uses a PandasExecutor.

    Returns
    -------
    executor : Executor
        An Executor object for the given connector.

    Raises
    ------
    ExecutorException
        If an unknown connector type is specified, or if the default `con` is used without installing duckdb.
    """

    if con is None:
        try:
            import duckdb
        except ImportError:
            raise ExecutorException(
                "To support Pandas dataframe, please install duckdb."
            )
        con = duckdb.connect(database=":memory:")
        return PandasExecutor(con)
    elif issubclass(type(con), Executor):
        return con
    elif type(con).__name__ == "DuckDBPyConnection":
        return DuckdbExecutor(con)
    else:
        raise ExecutorException("Unknown connector with type " + type(con).__name__)


class Executor(ABC):
    """
    Base executor object- defines a template for special executor objects.

    Parameters
    ----------
    view_id : int
        The id of the next view to be created.
    prefix : str
        The prefix to be used for the view names.
    """

    def __init__(self):
        self.view_id = 0
        # tables or views of this prefix is not safe and may be rewritten
        self.prefix = "joinboost_tmp_"

    def get_next_name(self):
        """Get a unique name of the next view to be created."""
        name = self.prefix + str(self.view_id)
        self.view_id += 1
        return name

    @abstractmethod
    def get_schema(self, table: str) -> list:
        """
        Get a list of column names in a table.

        Parameters
        ----------
        table : str
            The name of the table.

        Returns
        -------
        list
            A list of column names in the table.
        """

    @abstractmethod
    def add_table(self, table: str, table_address):
        """
        Add a new table to the database.

        Parameters
        ----------
        table : str
            The name of the table to add.
        table_address : str
            The address of the table to add.
        """

    @abstractmethod
    def delete_table(self, table: str):
        """
        Delete a table.

        Parameters
        ----------
        table : str
            The name of the table.
        """

    @abstractmethod
    def execute_spja_query(
        self,
        spja_data: SPJAData,
        mode: ExecuteMode = ExecuteMode.NESTED_QUERY,
    ) -> Any:
        """
        Executes an SPJA query using the current object's database connection.

        Parameters
        ----------
        spja_data : SPJAData
            The SPJAData object containing the query parameters.
        mode: ExecuteMode, optional
            The mode in which the query is executed. Default is ExecuteMode.NESTED_QUERY.
            if ExecuteMode.WRITE_TO_TABLE
                The query is executed and the results are stored in a new table.
                The table name is returned.
            if ExecuteMode.CREATE_VIEW
                The query is executed and the results are stored in a new view.
                The table name is returned.
            if ExecuteMode.EXECUTE
                The query is executed and the results are returned.
            if ExecuteMode.NESTED_QUERY
                Creates a parenthesized query and returns it as a string.

        Returns
        -------
        Any
            The result of the query. Determined by `mode`.

        """

    def set_query(self, param, set_left, set_right):
        pass


class DuckdbExecutor(Executor):
    """
    Executor object providing methods for executing queries on a DuckDB database.

    Attributes
    ----------
    conn : Connection
        A DuckDB connection object.
    debug : bool
        A flag to enable/disable debug mode.
    """

    def __init__(self, conn, debug=False):
        super().__init__()
        self.conn = conn
        self.debug = debug
        self.replace = True

    def get_schema(self, table: str) -> list:
        # duckdb stores table info in [cid, name, type, notnull, dflt_value, pk]
        table_info = self._execute_query("PRAGMA table_info(" + table + ")")
        return [x[1] for x in table_info]

    def _gen_sql_case(self, leaf_conds: list):
        conds = []
        for leaf_cond in leaf_conds:
            cond = "CASE\n"
            for (pred, annotations) in leaf_cond:
                cond += (
                    " WHEN "
                    + " AND ".join(annotations)
                    + " THEN CAST("
                    + str(pred)
                    + " AS DOUBLE)\n"
                )
            cond += "ELSE 0 END\n"
            conds.append(cond)
        return conds

    def delete_table(self, table: str):
        self.check_table(table)
        sql = "DROP TABLE IF EXISTS " + table + ";\n"
        self._execute_query(sql)

    def add_table(self, table: str, table_address):
        if table_address is None:
            raise ExecutorException("Please pass in the csv file location")
        self.conn.execute(
            f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM '{table_address}'"
        )

    def set_query(self, operation, expr1, expr2):
        return f"({expr1} {operation} {expr2})"

    def case_query(
        self,
        from_table: str,
        operator: str,
        cond_attr: str,
        base_val: str,
        case_definitions: list,
        select_attrs: list = [],
        table_name: str = None,
        order_by: str = None,
    ):
        # If no table name is provided, generate a new one
        if not table_name:
            view = self.get_next_name()
        else:
            view = table_name
        
        # for gradient boosting, the prediction is the base_val plus the sum of the tree predictions
        pred_agg = AggExpression(Aggregator.ADD, [base_val] + case_definitions)

        # Create the SELECT statement with the CASE statement
        attrs = ",".join(select_attrs)
        sql = (
            f"CREATE OR REPLACE TABLE {view} AS\n"
            + f"SELECT {attrs}, {agg_to_sql(pred_agg, qualified= False)} "
            + f"AS {cond_attr} FROM {from_table} "
        )

        if order_by:
            sql += f"ORDER BY {order_by};"

        self._execute_query(sql)
        # print(view)
        return view
    
    def check_table(self, table):
        if not table.startswith(self.prefix):
            raise Exception("Don't modify user tables!")

    def update_query(self, update_expression, table, select_conds: list = []):
        """
        Executes an SQL UPDATE statement on a specified table with the provided update_expression.

        Parameters
        ----------
        update_expression : str
            A string specifying the update expression to be executed.
        table : str
            A string specifying the name of the table to execute the update query on.
        select_conds : list, optional
            A list of strings specifying the selection conditions for the update query. Default is an empty list.

        Raises
        ------
        Exception
            If the specified table does not start with the prefix of the current DuckDBExecutor object.

        Returns
        -------
        None
        """
        self.check_table(table)
        sql = "UPDATE " + table + " SET " + update_expression + " \n"

        if len(select_conds) > 0:
            sql += "WHERE " + " AND ".join([selection_to_sql(cond) for cond in select_conds]) + "\n"
        self._execute_query(sql)

    def execute_spja_query(
        self, spja_data: SPJAData, mode: ExecuteMode = ExecuteMode.NESTED_QUERY
    ) -> Any:

        if mode == ExecuteMode.WRITE_TO_TABLE:
            return self._spja_query_to_table(spja_data)
        elif mode == ExecuteMode.CREATE_VIEW:
            return self._spja_query_as_view(spja_data)
        elif mode == ExecuteMode.EXECUTE:
            spja = self.spja_query(spja_data, parenthesize=False)
            return self._execute_query(spja)
        elif mode == ExecuteMode.NESTED_QUERY:
            return self.spja_query(spja_data)

    def _spja_query_to_table(self, spja_data: SPJAData) -> str:
        """
        Executes an SPJA query and stores the results in a new table.

        Parameters
        ----------
        spja_data : SPJAData
            The SPJAData object containing the query parameters.

        Returns
        -------
        str
            The name of the new table.
        """
        spja = self.spja_query(spja_data, parenthesize=False)
        name_ = self.get_next_name()
        entity_type_ = "TABLE "
        sql = (
            "CREATE "
            + ("OR REPLACE " if spja_data.replace else "")
            + entity_type_
            + name_
            + " AS "
        )
        sql += spja
        self._execute_query(sql)
        return name_

    def _spja_query_as_view(
        self,
        spja_data: SPJAData,
    ):
        """
        Create a view from the results of an SPJA query.

        Parameters
        ----------
        spja_data : SPJAData
            The SPJAData object containing the query parameters.

        Returns
        -------
        str
            The name of the view created by the method.
        """

        spja = self.spja_query(spja_data, parenthesize=False)

        name_ = self.get_next_name()
        entity_type_ = "VIEW "
        sql = (
            "CREATE "
            + ("OR REPLACE " if self.replace else "")
            + entity_type_
            + name_
            + " AS "
        )
        sql += spja
        self._execute_query(sql)
        return name_

    def spja_query(
        self,
        spja_data: SPJAData,
        parenthesize: bool = True,
    ):
        """
        Generates an SQL query based on the given SPJAData object and returns the query as a string.

        Parameters
        ----------
        spja_data : SPJAData
            The SPJAData object representing the query to be generated.
        parenthesize: bool, optional
            wrap the query in parentheses. Default is True

        Returns
        -------
        str
            The generated SQL query as a string.

        """

        parsed_aggregate_expressions = []
        for target_col, aggExp in spja_data.aggregate_expressions.items():
            # we need window clause if we have a window_by and the aggregate is not a simple aggregation
            window_clause = " OVER joinboost_window " if spja_data.window_by and is_agg(aggExp.agg) else ""
            rename_expr = (" AS " + target_col) if target_col is not None else ""
            parsed_aggregate_expressions.append(agg_to_sql(aggExp, qualified=spja_data.qualified) + window_clause + rename_expr)

        sql = "SELECT " + ", ".join(parsed_aggregate_expressions) + "\n"
        sql += "FROM " + ",".join(spja_data.from_tables) + "\n"

        if len(spja_data.select_conds) > 0 or len(spja_data.join_conds) > 0:
            sql += "WHERE " + " AND ".join([selection_to_sql(cond) for cond in spja_data.select_conds + spja_data.join_conds]) + "\n"
        if len(spja_data.window_by) > 0:
            sql += ("WINDOW joinboost_window AS (ORDER BY " + ",".join(spja_data.window_by) + ")\n")
        if len(spja_data.group_by) > 0:
            sql += "GROUP BY " + ",".join(spja_data.group_by) + "\n"
        if len(spja_data.order_by) > 0:
            sql += ("ORDER BY " + ",".join([f"{col} {order}" for (col, order) in spja_data.order_by])+ "\n")
        if spja_data.limit is not None:
            sql += "LIMIT " + str(spja_data.limit) + "\n"
        if spja_data.sample_rate is not None:
            sql += "USING SAMPLE " + str(spja_data.sample_rate * 100) + " %\n"

        if parenthesize:
            sql = f"({sql})"

        return sql

    def rename(self, table, old_name, new_name):
        sql = f"ALTER TABLE {table} RENAME COLUMN {old_name} TO {new_name};"
        self._execute_query(sql)

    def _execute_query(self, q):
        """
        Executes the given SQL query and returns the result.

        Parameters
        ----------
        q : str
            The SQL query to be executed.

        Returns
        -------
        Any
            The result of the query execution.
        """
        start_time = time.time()
        if self.debug:
            print(q)
        self.conn.execute(q)
        elapsed_time = time.time() - start_time

        if self.debug:
            print(elapsed_time)

        result = None
        try:
            result = self.conn.fetchall()
            if self.debug:
                print(result)
        except Exception as e:
            print(e)
        return result

class SparkExecutor(DuckdbExecutor):
    def add_table(self, table: str, table_address):
        if table_address is None:
            raise ExecutorException("Please pass in the pandas dataframe!")

        df = self.conn.read.csv(table_address, header=True, inferSchema=True)
        # register the DataFrame as a temporary view
        df.createOrReplaceTempView(table)

    def get_schema(self, table: str) -> list:
        # Get the schema of the source_table
        source_table_schema = self.conn.sql(f"DESCRIBE {table}")

        # Extract the column names as a list
        return [row.col_name for row in source_table_schema.collect()]

    def _execute_query(self, q, collect=True):
        """
        Executes the given SQL query and returns the result.

        Parameters
        ----------
        q : str
            The SQL query to be executed.

        Returns
        -------
        Any
            The result of the query execution.
        """
        start_time = time.time()
        if self.debug:
            print(q)
        result = self.conn.sql(q)
        elapsed_time = time.time() - start_time

        if self.debug:
            print(elapsed_time)

        if self.debug:
            result.show()

        if collect:
            return [tuple(row) for row in result.collect()]
        else:
            return result

    def _spja_query_to_table(self, spja_data: SPJAData) -> str:
        """
        Executes an SPJA query and stores the results in a new table.

        Parameters
        ----------
        spja_data : SPJAData
            The SPJAData object containing the query parameters.

        Returns
        -------
        str
            The name of the new table.
        """
        spja = self.spja_query(spja_data, parenthesize=False)
        name_ = self.get_next_name()

        result_df = self._execute_query(spja, collect=False)

        # Register the result DataFrame as a new temporary table
        result_df.createOrReplaceTempView(name_)
        return name_

    # {case: value} operator {case: value} ...
    def case_query(
        self,
        from_table: str,
        operator: str,
        cond_attr: str,
        base_val: str,
        case_definitions: list,
        select_attrs: list = [],
        table_name: str = None,
        order_by: str = None,
    ):
        """
        Executes a SQL query with a CASE statement to perform tree-model prediction.
        Each CASE represents a tree and each WHEN within a CASE represents a leaf.

        :param from_table: str, name of the source table
        :param operator: str, the operator used to combine predictions
        :param cond_attr: str, name of the column used in the conditions of the case statement
        :param base_val: int, base value for the entire tree-model
        :param case_definitions: list, a list of lists containing the (leaf prediction, leaf predicates) for each tree.
        :param select_attrs: list, list of attributes to be selected, defaults to empty
        :param table_name: str, name of the new table, defaults to None
        :param order_by: str, name of the table to be ordered by rowid, defaults to None
        :return: str, name of the new table
        """

        # If no table name is provided, generate a new one
        if not table_name:
            view = self.get_next_name()
        else:
            view = table_name

        # for gradient boosting, the prediction is the base_val plus the sum of the tree predictions
        pred_agg = AggExpression(Aggregator.ADD, [base_val] + case_definitions)
        
        # Create the SELECT statement with the CASE statement
        attrs = ",".join(select_attrs)
        sql = (
            f"SELECT {attrs}, {agg_to_sql(pred_agg, qualified= False)}"
            + f"AS {cond_attr} FROM {from_table} "
        )
        if order_by:
            sql += f"ORDER BY {order_by};"

        result_df = self._execute_query(sql, collect=False)

        # Register the result DataFrame as a new temporary table
        result_df.createOrReplaceTempView(view)
        return view


class PandasExecutor(DuckdbExecutor):
    # Because Pandas is not a database, we use a dictionary to store table_name -> dataframe
    table_registry = {}

    def __init__(self, conn, debug=False):
        super().__init__(conn)
        self.debug = debug

    def add_table(self, table: str, table_address):
        if table_address is None:
            raise ExecutorException("Please pass in the pandas dataframe!")

        # check if the table_address is a string path
        if isinstance(table_address, str):
            table_address = pd.read_csv(table_address)
        self.table_registry[table] = table_address

    def delete_table(self, table):
        if self.debug:
            print("deleting table: ", table)

        if table in self.table_registry:
            del self.table_registry[table]

    # set operations in pandas
    def set_query(self, operation, df1_name, df2_name):
        df1 = self.table_registry[df1_name]
        df2 = self.table_registry[df2_name]
        # unqualify the column names in both dataframes
        df1.columns = [col.split(".")[-1] for col in df1.columns]
        df2.columns = [col.split(".")[-1] for col in df2.columns]

        def intersect_all(df1, df2):
            # This function implements 'INTERSECT ALL' operation.
            df1['key'] = 1
            df2['key'] = 1
            merged = pd.merge(df1, df2, on=list(df1.columns[:-1]), suffixes=['', '_'])
            merged = merged[merged.key == merged.key_]
            return merged[list(df1.columns[:-1])]

        if operation == "UNION":
            df1 = pd.concat([df1, df2], ignore_index=True)
        elif operation == "UNION ALL":
            df1 = pd.concat([df1, df2])
        elif operation == "INTERSECT":
            df1 = pd.merge(df1, df2)
        elif operation == "INTERSECT ALL":
            df1 = intersect_all(df1, df2)
        elif operation == "EXCEPT":
            df1 = df1.merge(df2, how='outer', indicator=True).query('_merge=="left_only"').drop('_merge', axis=1)
        elif operation == "EXCEPT ALL":
            df1['_key'] = df1.groupby(list(df1.columns)).cumcount()
            df2['_key'] = df2.groupby(list(df2.columns)).cumcount()
            df1 = pd.merge(df1, df2, how='outer', on=list(df1.columns), indicator=True, suffixes=['', '_y'])
            df1 = df1.query('_merge=="left_only"').drop(['_merge', '_key'], axis=1)
        else:
            raise ExecutorException("Unsupported set operation!")
        
        self.table_registry[df1_name] = df1
        return df1_name

    def get_schema(self, table):
        # unqualify the column names, this is required as duckdb returns unqualified column names
        return [col.split(".")[-1] for col in self.table_registry[table].columns]


    def execute_spja_query(
        self, spja_data: SPJAData, mode: ExecuteMode = ExecuteMode.WRITE_TO_TABLE
    ):
        # TODO: may need to execute_spja_query, if the from_tables in spja_table is also a spja_table
        from_dfs = {}

        for table in spja_data.from_tables:
            from_dfs[table] = self.table_registry[table]

        agg_conditions = self.convert_agg_conditions(spja_data.aggregate_expressions)

        sqls = selections_to_sql(spja_data.select_conds + spja_data.join_conds)

        # search select_conds for join_conditions that are of the form "table1.col1 IS NOT DISTINCT FROM table2.col2"
        join_conds = [
            re.findall(r"(\w+\.\w+ IS NOT DISTINCT FROM \w+\.\w+)", cond)[0]
            for cond in sqls
            if "IS NOT DISTINCT FROM" in cond
        ]
        select_conds = self.convert_predicates(sqls)

        # join_conds are of the form "table1.col1 IS NOT DISTINCT FROM table2.col2".
        # extract the table1.col1 and table2.col2
        join_conds = [re.findall(r"(\w+\.\w+)", cond) for cond in join_conds]

        df = list(from_dfs.values())[0]

        if len(join_conds) > 0:
            df = self.execute_join(
                from_dfs,
                spja_data.join_conds
            )

        # filter by select_conds
        if len(select_conds) > 0:
            converted_select_conds = " and ".join(select_conds)
            df = df.query(converted_select_conds)
        
        # removal of all qualification of columns
        if df is not None:
            for col in df.columns:
                df = df.rename(columns={col: col.split(".")[-1]})
            # if there are duplicate columns, drop them
            df = df.loc[:, ~df.columns.duplicated()]


        # group by and aggregate
        df = self.apply_group_by_and_agg(
            agg_conditions, df, spja_data.group_by, spja_data.window_by
        )

        # drop columns that don't appear in aggregate_expressions
        final_cols = [
            col for col in spja_data.aggregate_expressions.keys() if col is not None
        ]
        df = df[final_cols]

        # sort by each column in order_by
        if len(spja_data.order_by) > 0:
            for col, order in spja_data.order_by:
                df = df.sort_values(col, ascending=(order == "ASC" or order is None))

        # limit
        if spja_data.limit is not None:
            df = df.head(spja_data.limit)

        df = self.reorder_columns(spja_data.aggregate_expressions, df)

        # TODO: clean up mode implementation
        if mode in (
            ExecuteMode.WRITE_TO_TABLE,
            ExecuteMode.NESTED_QUERY,
            ExecuteMode.CREATE_VIEW,
        ):
            name_ = self.get_next_name()
            # always qualify intermediate tables as future aggregations for these tables will come qualified
            # for col in df.columns:
                # if col not in ["s", "c"]:
                #     # strip any table name from the column name
                #     df = df.rename(columns={col: name_ + "." + col.split(".")[-1]})
            if self.debug:
                print("creating table " + name_)
                print(df.head())
            df.name = name_
            self.table_registry[name_] = df
            return name_
        elif mode == ExecuteMode.EXECUTE:
            if self.debug:
                print("returning result")
                print(df.head())

        return df.values.tolist()

    def reorder_columns(self, aggregate_expressions, df):
        # reorder the columns according to the order of aggregate_expressions
        if len(aggregate_expressions) > 0:
            # get list of column names from aggregate_expressions that's also in df
            agg_cols = [
                col for col in aggregate_expressions.keys() if col in df.columns
            ]
            # get list of column names from df
            df_cols = df.columns
            # merge the two lists, giving priority to agg_cols and removing duplicates
            cols = agg_cols + [col for col in df_cols if col not in agg_cols]
            df = df.reindex(columns=cols)
        return df

    def apply_group_by_and_agg(self, agg_conditions, df, group_by, window_by):
        # print(agg_conditions)

        if len(group_by) > 0:
            # if group_by element is not of the form joinboost_<digit>.col, then unqualify it
            for i, col in enumerate(group_by):
                # check if unqualified column name exists in df.columns
                if col.split(".")[-1] in df.columns:
                    group_by[i] = col.split(".")[-1]
            inter_df = df.groupby(group_by)
            if len(agg_conditions) > 0:
                # check if column does not exist and create it before applying agg_conditions
                for col in agg_conditions.keys():
                    # generate unqualified names in df.columns
                    unqualified_cols = [col.split(".")[-1] for col in df.columns]
                    if col not in df.columns and col not in unqualified_cols:
                        df[col] = 1

                for col in list(agg_conditions.keys()):
                    if agg_conditions[col].column == "*":
                        func = agg_conditions[col].aggfunc
                        df[col] = df.apply(func, axis=1)
                        del agg_conditions[col]
                    elif agg_conditions[col].aggfunc == "first" and (
                        col == agg_conditions[col].column
                        or col == agg_conditions[col].column.split(".")[-1]
                    ):
                        del agg_conditions[col]

                if len(agg_conditions) > 0:
                    df = inter_df.agg(**agg_conditions).reset_index()
                # unqualify all columns in df. This is to avoid nested columns being qualified with the table name
                for col in df.columns:
                    df = df.rename(columns={col: col.split(".")[-1]})
        else:
            if len(agg_conditions) > 0:
                if len(window_by) > 0:
                    # apply cumulative sum on window_by columns in pandas dataframe
                    df = df.sort_values(window_by)
                    # TODO: remove hack, propagate g_col and h_col instead of hardcoding
                    df["s"] = df["s"].cumsum()
                    df["c"] = df["c"].cumsum()
                else:
                    # check if column does not exist and create it before applying agg_conditions (for s anc c)
                    for col in agg_conditions.keys():
                        if col not in df.columns:
                            df[col] = 1

                    # check if column is * and apply aggfunc to the entire row
                    for col in list(agg_conditions.keys()):
                        if agg_conditions[col].column == "*":
                            func = agg_conditions[col].aggfunc
                            if isinstance(func, dict):
                                df[col] = np.where(eval(func['cond']), eval(func['true']), eval(func['false']))
                            else:
                                df[col] = df.eval(func)
                            del agg_conditions[col]
                        elif (
                            agg_conditions[col].aggfunc == "first"
                            and col == agg_conditions[col].column
                        ):
                            del agg_conditions[col]

                    if len(agg_conditions) > 0:
                        df = (
                            df.assign(temp=0)
                            .groupby("temp")
                            .agg(**agg_conditions)
                            .reset_index()
                            .drop(columns="temp")
                        )
                    for col in df.columns:
                        df = df.rename(columns={col: col.split(".")[-1]})

        return df

    def execute_join(self, df_to_join, join_conds):
        # Step 1: Extract all join keys for each pair of dataframes
        join_conditions = {}
        for sel in join_conds:
            left_attr, right_attr = sel.para[0], sel.para[1]
            left_table, right_table = left_attr.table(), right_attr.table()
            left_attribute_name, right_attribute_name = value_to_sql(left_attr, False), value_to_sql(right_attr, False)

            key = tuple(sorted((left_table, right_table)))
            if key not in join_conditions:
                join_conditions[key] = {'left_table': left_table, 'right_table': right_table, 'left_keys': [], 'right_keys': []}

            if left_table == join_conditions[key]['left_table']:
                join_conditions[key]['left_keys'].append(left_attribute_name)
                join_conditions[key]['right_keys'].append(right_attribute_name)
            else:
                join_conditions[key]['left_keys'].append(right_attribute_name)
                join_conditions[key]['right_keys'].append(left_attribute_name)

        # Step 1.5: Construct Join graph using join tables as nodes and join conditions as edges without using networkx
        join_graph = MiniJoinGraph()
        for key in join_conditions:
            join_graph.add_node(join_conditions[key]['left_table'])
            join_graph.add_node(join_conditions[key]['right_table'])
            join_graph.add_edge(join_conditions[key]['left_table'], join_conditions[key]['right_table'])

        dfs_order, dfs_join_order = join_graph.get_dfs_order()


        # Step 2: Implement a simple query optimizer to decide the join order
        # def simple_query_optimizer(join_conditions):
        #     # This is a simple example; you can implement more advanced optimization techniques if needed.
        #     return sorted(join_conditions.values(), key=lambda x: (x['left_table'], x['right_table']))
        #
        # optimized_join_order = simple_query_optimizer(join_conditions)

        # Step 3: Join all dataframes together using Pandas merge function
        result = None
        for left_table_name, right_table_name in dfs_join_order:
            left_table = df_to_join[left_table_name]
            right_table = df_to_join[right_table_name]
            join_cond = join_conditions[tuple(sorted((left_table_name, right_table_name)))]

            # Rename duplicate columns in the right table to match the left table
            for left_key, right_key in zip(join_cond['left_keys'], join_cond['right_keys']):
                if left_key != right_key:
                    right_table = right_table.rename(columns={right_key: left_key})

            if result is None:
                result = pd.merge(left_table, right_table, on=join_cond['left_keys'])
            else:
                result = pd.merge(result, right_table, on=join_cond['left_keys'])

        return result

    @staticmethod
    def get_unqualified_cols(df):
        return [col.split(".")[-1] for col in df.columns]

    @staticmethod
    def unqualify_df(df):
        for col in df.columns:
            df = df.rename(columns={col: col.split(".")[-1]})
        return df

    def convert_predicates(self, select_conds):
        # TODO: handle in and not in
        # replace ' = ' with ' == ' but only if = is not part of <> or <= or >=
        select_conds = [re.sub(r"(?<!<|>)=(?!=)", "==", cond) for cond in select_conds]
        # ignore predicates of the form 's.a is not distinct from t.b'
        select_conds = [cond for cond in select_conds if "DISTINCT" not in cond]
        # check if predicates do not start with joinboost_<number>.col <op> <value> and if so, remove the table name
        for i, cond in enumerate(select_conds):
            # split by dot and remove only the first element and return everything else as it is
            select_conds[i] = ".".join(cond.split(".")[1:])

        # wrap each operand (of the form identifier.identifier or identifier) with backticks, ignore any multi-digit numeric values
        select_conds = [
            re.sub(r"\b([a-zA-Z_0-9]+\.[a-zA-Z_]+|[a-zA-Z_]+)\b", r"`\g<1>`", cond)
            for cond in select_conds
        ]

        # wrap each select condition with parentheses
        select_conds = ["(" + cond + ")" for cond in select_conds]

        return select_conds

    def convert_agg_conditions(self, aggregate_expressions):
        agg_conditions = {}
        # handle aggregate expressions
        for target_col, aggregation_spec in aggregate_expressions.items():
            para, agg = aggregation_spec

            # if para is QualifiedName
            if isinstance(para, QualifiedAttribute):
                para = agg_to_sql(para)

            if target_col is None:
                target_col = para

            # use named aggregation and column renaming with dictionary
            if agg.value == Aggregator.COUNT.value:
                agg_conditions[target_col] = pd.NamedAgg(column=para, aggfunc="count")
            elif agg.value == Aggregator.SUM.value:
                # check if column is a number in string form, in that case use target_col as the column name
                if str(para).isnumeric():
                    para = target_col
                agg_conditions[target_col] = pd.NamedAgg(column=para, aggfunc="sum")
            elif agg.value == Aggregator.MAX.value:
                agg_conditions[target_col] = pd.NamedAgg(column=para, aggfunc="max")
            elif agg.value == Aggregator.MIN.value:
                agg_conditions[target_col] = pd.NamedAgg(column=para, aggfunc="min")
            elif agg.value == Aggregator.IDENTITY.value:
                if para == "1":
                    agg_conditions[target_col] = pd.NamedAgg(
                        column="*", aggfunc="1"
                    )
                elif para != "*":
                    agg_conditions[target_col] = pd.NamedAgg(
                        column=para, aggfunc="first"
                    )
                else:
                    pass
            elif agg.value == Aggregator.IDENTITY_LAMBDA.value:
                agg_conditions[target_col] = pd.NamedAgg(column="*", aggfunc=para)

            else:
                raise ExecutorException("Unsupported aggregation function!")
        return agg_conditions
