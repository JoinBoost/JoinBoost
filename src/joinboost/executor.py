import re
import time
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Any, List

import pandas as pd

from joinboost import aggregator


ExecuteMode = Enum(
    "ExecuteMode", ["WRITE_TO_TABLE", "CREATE_VIEW", "EXECUTE", "NESTED_QUERY"]
)


class ExecutorException(Exception):
    pass


@dataclass(frozen=True)
class SPJAData:
    """
    Dataclass representing an SPJA query and its associated parameters.

    Attributes
    ----------
    aggregate_expressions : doict
        A dictionary mapping column names to tuples containing the aggregation expression and the aggregator object.
    from_tables : list
        A list of table names to select from. By default, an empty list.
    select_conds : list
        A list of conditions to apply to the SELECT statement. By default, an empty list.
    join_conds : list
        A list of conditions of the form "table1.col1 IS NOT DISTINCT FROM table2.col2". By default, an empty list.
    group_by : list
        A list of column names to group by. By default, an empty list.
    window_by : list
        A list of column names to use for windowing. By default, an empty list.
    order_by : list
        The column to use for ordering the results. By default, an empty list.
    limit : int
        The maximum number of rows to return
    sample_rate : float
        The sampling rate to use for the query.
    replace : bool
        If True, replaces an existing table or view with the same name.
    join_type : str
        The type of join to use for the query. By default, "INNER".
    """

    aggregate_expressions: dict = field(
        default_factory=lambda: {None: ("*", aggregator.Aggregator.IDENTITY)}
    )
    from_tables: List[str] = field(default_factory=list)
    select_conds: List[str] = field(default_factory=list)
    join_conds: List[str] = field(default_factory=list)
    group_by: List[str] = field(default_factory=list)
    window_by: List[str] = field(default_factory=list)
    order_by: List[str] = field(default_factory=list)
    limit: Optional[int] = None
    sample_rate: Optional[float] = None
    replace: bool = True
    join_type: str = "INNER"


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
    def case_query(
        self,
        from_table: str,
        operator: str,
        cond_attr: str,
        base_val: str,
        case_definitions: list,
        select_attrs: list = [],
        table_name: str = None,
    ):
        """
        Executes a SQL query with a CASE statement to perform tree-model prediction.
        Each CASE represents a tree and each WHEN within a CASE represents a leaf.

        Parameters
        -----------
        from_table : str
            Name of the source table
        operator : str
            The operator used to combine predictions
        cond_attr : str
            Name of the column used in the conditions of the case statement
        base_val : int
            Base value for the entire tree-model
        case_definitions : list
            A list of lists containing the (leaf prediction, leaf predicates) for each tree.
        select_attrs : list, optional (default=[])
            List of attributes to be selected
        table_name : str, optional (default=None)
            Name of the new table
        order_by : str, optional (default=None)
            Name of the table to be ordered by rowid

        Returns
        --------
        str
            Name of the new table
        """

    @abstractmethod
    def window_query(
        self, view: str, select_attrs: list, base_attr: str, cumulative_attrs: list
    ):
        """
        A function to create a window query. TODO: Remove this function.

        Parameters
        ----------
        view : str
            The view name.
        select_attrs : list
            A list of attributes to select.
        base_attr : str
            The base attribute.
        cumulative_attrs : list
            A list of cumulative attributes.

        Returns
        -------
        str
            The view name.
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

    def _check_mode(self, mode: ExecuteMode):
        """
        Check if the given mode is supported by the executor.

        Parameters
        ----------
        mode: ExecuteMode
            The mode to check.

        Raises
        -------
        ValueError
            If the mode is not supported.

        """
        if not isinstance(mode, ExecuteMode):
            raise ValueError(
                f"mode parameter {mode} must be an instance of ExecuteMode."
            )


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

    # TODO: remove it
    def window_query(
        self, view: str, select_attrs: list, base_attr: str, cumulative_attrs: list
    ):
        view_name = self.get_next_name()
        sql = "CREATE OR REPLACE VIEW " + view_name + " AS SELECT * FROM\n"
        sql += "(\nSELECT " + ",".join(select_attrs)
        for attr in cumulative_attrs:
            sql += ",SUM(" + attr + ") OVER joinboost_window as " + attr
        sql += "\nFROM " + view
        sql += " WINDOW joinboost_window AS (ORDER BY " + base_attr + ")\n)"
        self._execute_query(sql)
        return view_name

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

        # If no select attributes are provided, retrieve all columns
        # except the one used in the conditions of the case statement
        if not select_attrs:
            attrs = self._execute_query("PRAGMA table_info(" + from_table + ")")
            for attr in attrs:
                if attr != cond_attr:
                    select_attrs.append(attr[1])

        # If no table name is provided, generate a new one
        if not table_name:
            view = self.get_next_name()
        else:
            view = table_name

        # Prepare the case statement using the provided operator
        cases = []
        for case_definition in case_definitions:
            sql_case = f"{operator}\nCASE\n"
            for val, cond in case_definition:
                conds = " AND ".join(cond)
                sql_case += f" WHEN {conds} THEN CAST({val} AS DOUBLE)\n"
            sql_case += "ELSE 0 END\n"
            cases.append(sql_case)
        sql_cases = "".join(cases)

        # Create the SELECT statement with the CASE statement
        attrs = ",".join(select_attrs)
        sql = (
            f"CREATE OR REPLACE TABLE {view} AS\n"
            + f"SELECT {attrs}, {base_val}"
            + f"{sql_cases}"
            + f"AS {cond_attr} FROM {from_table} "
        )
        if order_by:
            sql += f"ORDER BY {order_by};"
        self._execute_query(sql)
        print(view)
        return view

    # Write a method that can generate a function based on the case definitions
    # The function will take in a row and return a value
    # This function can be used to generate a new column
    # This function will not use SQL or the database and instead will be run in pandas dataframes
    # def case_function(self, from_table: str, operator: str, cond_attr: str, base_val: str,
    #                  case_definitions: list, select_attrs: list = [], table_name: str = None):
    #
    #     def case_function(row):
    #         result = base_val
    #         predicates = []
    #         for case_definition in case_definitions:
    #
    #             for val, conds in case_definition:
    #                 # each cond in conds is a string of the form "attr =/>=/</<=/> val"
    #                 # we need to split this string and then check if the row[attr] satisfies the condition
    #                 temp = []
    #                 for i, cond in enumerate(conds):
    #                     attr, op, val = cond.split()
    #                     temp += ["row['" + attr + "'] " + op + " " + val]
    #                 val + " if  (" + " and ".join(temp) + ") else 0"
    #
    #         return result

    def check_table(self, table):
        """
        Check if a table is a user table.

        Parameters
        ----------
        table : str
            The name of the table to check.

        Raises
        ------
        Exception
            If the table does not start with the prefix specified for user tables.

        Returns
        -------
        None
        """

        if not table.startswith(self.prefix):
            raise Exception("Don't modify user tables!")

    def add_table(self, table: str, table_address: str) -> None:
        ...

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
            sql += "WHERE " + " AND ".join(select_conds) + "\n"
        self._execute_query(sql)

    def execute_spja_query(
        self, spja_data: SPJAData, mode: ExecuteMode = ExecuteMode.NESTED_QUERY
    ) -> Any:

        self._check_mode(mode)

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
        for target_col, (para, agg) in spja_data.aggregate_expressions.items():
            parsed_expression = self._parse_aggregate_expression(
                target_col, para, agg, window_by=spja_data.window_by
            )
            parsed_aggregate_expressions.append(parsed_expression)

        sql = "SELECT " + ", ".join(parsed_aggregate_expressions) + "\n"
        sql += "FROM " + ",".join(spja_data.from_tables) + "\n"

        if len(spja_data.select_conds) > 0:
            sql += "WHERE " + " AND ".join(spja_data.select_conds) + "\n"
        if len(spja_data.window_by) > 0:
            sql += (
                "WINDOW joinboost_window AS (ORDER BY "
                + ",".join(spja_data.window_by)
                + ")\n"
            )
        if len(spja_data.group_by) > 0:
            sql += "GROUP BY " + ",".join(spja_data.group_by) + "\n"
        if len(spja_data.order_by) > 0:
            sql += (
                "ORDER BY "
                + ",".join([f"{col} {order}" for (col, order) in spja_data.order_by])
                + "\n"
            )
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

    def _parse_aggregate_expression(
        self, target_col: str, para, agg: aggregator.Aggregator, window_by: list = None
    ):
        """
        Parameters
        ----------
        target_col : str
            The name of the target column to rename the result to.
        para : Union[str, float, int, None]
            The parameter of the aggregation function.
        agg : aggregator.Aggregator
            An aggregator object that represents the aggregation function.
        window_by : Optional[List[str]], optional
            A list of column names to partition the window by, by default None.

        Returns
        -------
        str
            The parsed SQL statement for the aggregate expression.
        """

        window_clause = (
            " OVER joinboost_window " if window_by and aggregator.is_agg(agg) else ""
        )
        rename_expr = " AS " + target_col if target_col is not None else ""
        parsed_expression = (
            aggregator.parse_agg(agg, para) + window_clause + rename_expr
        )

        return parsed_expression


class PandasExecutor(DuckdbExecutor):
    # Because Pandas is not a database, we use a dictionary to store table_name -> dataframe
    table_registry = {}

    def __init__(self, conn, debug=False):
        super().__init__(conn)
        self.debug = debug
        self.prefix = "joinboost_"
        self.table_counter = 0

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

    def get_schema(self, table):
        # unqualify the column names, this is required as duckdb returns unqualified column names
        return [col.split(".")[-1] for col in self.table_registry[table].columns]

    # mode 1: write the query result to a table and return table name
    # mode 2: same as mode 1
    # mode 3: execute the query and return the result
    # mode 4: same as mode 1 (for now)
    def execute_spja_query(
        self, spja_data: SPJAData, mode: ExecuteMode = ExecuteMode.NESTED_QUERY
    ):

        self._check_mode(mode)

        intermediates = {}
        for table in spja_data.from_tables:
            intermediates[table] = self.table_registry[table]

        agg_conditions = self.convert_agg_conditions(spja_data.aggregate_expressions)

        select_conds = self.convert_predicates(spja_data.select_conds)

        # join_conds are of the form "table1.col1 IS NOT DISTINCT FROM table2.col2"p.
        # extract the table1.col1 and table2.col2
        join_conds = [re.findall(r"(\w+\.\w+)", cond) for cond in spja_data.join_conds]

        # filter list of tables that don't have any join conditions
        tables_to_join = [
            table
            for table in spja_data.from_tables
            if any(
                [
                    cond[0].startswith(table) or cond[1].startswith(table)
                    for cond in join_conds
                ]
            )
        ]

        # subtract tables_to_join from from_tables to get the tables that don't have any join conditions
        tables_to_cross = list(set(spja_data.from_tables) - set(tables_to_join))

        df = self.join(
            intermediates,
            join_conds,
            spja_data.join_type,
            tables_to_cross,
            tables_to_join,
        )

        # filter by select_conds
        if len(select_conds) > 0:
            converted_select_conds = " and ".join(select_conds)
            df = df.query(converted_select_conds)

        # group by and aggregate
        df = self.apply_group_by_and_agg(
            agg_conditions, df, spja_data.group_by, spja_data.window_by
        )

        should_drop_columns = True
        for keys, values in spja_data.aggregate_expressions.items():
            if values[1] == Aggregator.IDENTITY and values[0] == "*":
                should_drop_columns = False

        if should_drop_columns:
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
            for col in df.columns:
                if col not in ["s", "c"]:
                    # strip any table name from the column name
                    df = df.rename(columns={col: name_ + "." + col.split(".")[-1]})
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
                df = inter_df.agg(**agg_conditions).reset_index()
                # unqualify all columns in df. This is to avoid nested columns being qualified with the table name
                for col in df.columns:
                    df = df.rename(columns={col: col.split(".")[-1]})
        else:
            if len(agg_conditions) > 0:
                if len(window_by) > 0:
                    # apply cumulative sum on window_by columns in pandas dataframe
                    df = df.sort_values(window_by)
                    # TODO: remove hack, propogate g_col and h_col instead of hardcoding
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
                            df[col] = df.apply(func, axis=1)
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

    # computes join or cross (if no join condition) between all tables.
    def join(
        self, intermediates, join_conds, join_type, tables_to_cross, tables_to_join
    ):
        df = None
        # handle cross joins
        if len(tables_to_cross) > 0:
            for table in tables_to_cross:
                if df is None:
                    df = intermediates[table]
                else:
                    df = df.merge(
                        intermediates[table], how="cross", suffixes=("", "_drop")
                    ).filter(regex="^(?!.*_drop)")

        # unqualify all join conditions
        temp_join_conds = [
            [cond[0].split(".")[-1], cond[1].split(".")[-1]] for cond in join_conds
        ]
        # flatten temp_join_conds
        temp_join_conds = [item for sublist in temp_join_conds for item in sublist]

        # sort tables_to_join by the number of times their respective columns appear in join_conds.
        # TODO: this is a hacky way to avoid duplicate joining between tables.
        tables_to_join = sorted(
            tables_to_join,
            key=lambda x: len(
                [col for col in intermediates[x].columns if col in temp_join_conds]
            ),
            reverse=True,
        )

        for table in tables_to_join:

            if df is None:
                df = intermediates[table]
                if df is not None:
                    for col in df.columns:
                        df = df.rename(columns={col: col.split(".")[-1]})
                    # if there are duplicate columns, drop them
                    df = df.loc[:, ~df.columns.duplicated()]
                continue

            # search join_conds for the join conditions corresponding to table
            for cond in join_conds:
                df = self.universal_merge(cond, df, intermediates, join_type, table)

            if df is not None:
                for col in df.columns:
                    df = df.rename(columns={col: col.split(".")[-1]})
                # if there are duplicate columns, drop them
                df = df.loc[:, ~df.columns.duplicated()]

        # final removal of all qualification of columns
        if df is not None:
            for col in df.columns:
                df = df.rename(columns={col: col.split(".")[-1]})
            # if there are duplicate columns, drop them
            df = df.loc[:, ~df.columns.duplicated()]
        return df

    # Merges tables based on join conditions regardless of whether the join condition is on the left or right table
    # or whether the join condition is qualified or not. 'universal' here indicates its versatility.
    def universal_merge(self, cond, df, intermediates, join_type, table):
        # search join_conds for the join condition between table1 and table2
        if cond[0] in df.columns and cond[1] in intermediates[table].columns:
            # join the two tables
            df = df.merge(
                intermediates[table],
                how=join_type.lower(),
                left_on=cond[0],
                right_on=cond[1],
                suffixes=("", "_drop"),
            ).filter(regex="^(?!.*_drop)")
        elif cond[1] in df.columns and cond[0] in intermediates[table].columns:
            # join the two tables
            df = df.merge(
                intermediates[table],
                how=join_type.lower(),
                left_on=cond[1],
                right_on=cond[0],
                suffixes=("", "_drop"),
            ).filter(regex="^(?!.*_drop)")
        elif (
            cond[0].split(".")[-1] in df.columns
            and cond[1].split(".")[-1] in intermediates[table].columns
        ):
            # join the two tables
            df = df.merge(
                intermediates[table],
                how=join_type.lower(),
                left_on=cond[0].split(".")[-1],
                right_on=cond[1].split(".")[-1],
                suffixes=("", "_drop"),
            ).filter(regex="^(?!.*_drop)")
        elif (
            cond[1].split(".")[-1] in df.columns
            and cond[0].split(".")[-1] in intermediates[table].columns
        ):
            # join the two tables
            df = df.merge(
                intermediates[table],
                how=join_type.lower(),
                left_on=cond[1].split(".")[-1],
                right_on=cond[0].split(".")[-1],
                suffixes=("", "_drop"),
            ).filter(regex="^(?!.*_drop)")
        # check if one join condition is qualified and the other is not

        elif (
            cond[0].split(".")[-1] in df.columns
            and cond[1] in intermediates[table].columns
        ):
            # join the two tables
            df = df.merge(
                intermediates[table],
                how=join_type.lower(),
                left_on=cond[0].split(".")[-1],
                right_on=cond[1],
                suffixes=("", "_drop"),
            ).filter(regex="^(?!.*_drop)")
        elif (
            cond[1].split(".")[-1] in df.columns
            and cond[0] in intermediates[table].columns
        ):
            df = df.merge(
                intermediates[table],
                how=join_type.lower(),
                left_on=cond[1].split(".")[-1],
                right_on=cond[0],
                suffixes=("", "_drop"),
            ).filter(regex="^(?!.*_drop)")
        elif (
            cond[0] in df.columns
            and cond[1].split(".")[-1] in intermediates[table].columns
        ):
            # join the two tables
            df = df.merge(
                intermediates[table],
                how=join_type.lower(),
                left_on=cond[0],
                right_on=cond[1].split(".")[-1],
                suffixes=("", "_drop"),
            ).filter(regex="^(?!.*_drop)")
        elif (
            cond[1] in df.columns
            and cond[0].split(".")[-1] in intermediates[table].columns
        ):
            # join the two tables
            df = df.merge(
                intermediates[table],
                how=join_type.lower(),
                left_on=cond[1],
                right_on=cond[0].split(".")[-1],
                suffixes=("", "_drop"),
            ).filter(regex="^(?!.*_drop)")
        if df is not None:
            for col in df.columns:
                df = df.rename(columns={col: col.split(".")[-1]})
            # if there are duplicate columns, drop them
            df = df.loc[:, ~df.columns.duplicated()]
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

            if target_col is None:
                target_col = para

            # use named aggregation and column renaming with dictionary
            if agg == Aggregator.COUNT:
                agg_conditions[target_col] = pd.NamedAgg(column=para, aggfunc="count")
            elif agg == Aggregator.SUM:
                # check if column is a number in string form, in that case use target_col as the column name
                if str(para).isnumeric():
                    para = target_col
                agg_conditions[target_col] = pd.NamedAgg(column=para, aggfunc="sum")
            elif agg == Aggregator.MAX:
                agg_conditions[target_col] = pd.NamedAgg(column=para, aggfunc="max")
            elif agg == Aggregator.MIN:
                agg_conditions[target_col] = pd.NamedAgg(column=para, aggfunc="min")
            elif agg == Aggregator.IDENTITY:
                if para == "1":
                    agg_conditions[target_col] = pd.NamedAgg(
                        column="*", aggfunc=lambda x: 1
                    )
                else:
                    pass
            elif agg == Aggregator.IDENTITY_LAMBDA:
                agg_conditions[target_col] = pd.NamedAgg(column="*", aggfunc=para)

            else:
                raise ExecutorException("Unsupported aggregation function!")
        return agg_conditions
