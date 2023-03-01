import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any

from frozendict import frozendict

from joinboost import aggregator


class ExecutorException(Exception):
    pass


@dataclass(frozen=True)
class SPJAData:
    """
    Dataclass representing an SPJA query and its associated parameters.

    Attributes
    ----------
    aggregate_expressions : frozendict
        A dictionary mapping column names to tuples containing the aggregation expression and the aggregator object.
    from_tables : list
        A list of table names to select from. By default, an empty list.
    select_conds : list
        A list of conditions to apply to the SELECT statement. By default, an empty list.
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
    """

    aggregate_expressions: frozendict = frozendict(
        {None: ("*", aggregator.Aggregator.IDENTITY)}
    )
    from_tables: list[str] = field(default_factory=list)
    select_conds: list[str] = field(default_factory=list)
    group_by: list[str] = field(default_factory=list)
    window_by: list[str] = field(default_factory=list)
    order_by: Optional[str] = None
    limit: Optional[int] = None
    sample_rate: Optional[float] = None
    replace: bool = True


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

    Attributes
    ----------
    view_id : int
        The id of the next view to be created.
    prefix : str
        The prefix to be used for the view names.
    """

    def __init__(self):
        self.view_id = 0
        self.prefix = "joinboost_tmp_"

    def get_next_name(self):
        name = self.prefix + str(self.view_id)
        self.view_id += 1
        return name

    @abstractmethod
    def get_schema(self, table: str):
        pass

    @abstractmethod
    def add_table(self, table: str, table_address):
        pass

    @abstractmethod
    def delete_table(self, table: str):
        pass

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
        pass

    @abstractmethod
    def window_query(
        self, view: str, select_attrs: list, base_attr: str, cumulative_attrs: list
    ):
        pass

    @abstractmethod
    def execute_spja_query(
        self,
        mode: int,
        aggregate_expressions: dict,
        in_msgs: list = [],
        from_table: str = "",
        group_by: list = [],
        where_conds: dict = {},
        annotations: list = [],
        left_join: dict = {},
        table_name: str = "",
        replace: bool = True,
    ) -> str:
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
        # duckdb stores table info in [cid, name, type, notnull, dflt_value, pk]
        table_info = self._execute_query("PRAGMA table_info(" + table + ")")
        return [x[1] for x in table_info]

    def delete_table(self, table: str):
        """
        Delete a table.

        Parameters
        ----------
        table : str
            The name of the table.
        """
        self.check_table(table)
        sql = "DROP TABLE IF EXISTS " + table + ";\n"
        self._execute_query(sql)

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
        select_attrs: list = None,
        table_name: str = None,
    ):
        """
        This function creates a new table based on a query using a `CASE` statement with the specified operator.
        Parameters
        ----------
        from_table : str
            The table that the query will be run on.
        operator : str
            The operator to be used in the query (e.g. "+", "-", "*", etc.).
        cond_attr : str
            The attribute that the query will operate on.
        base_val : str
            The starting value for the operation.
        case_definitions : list
            A list of case definitions for the query. Each case definition is a list of tuples,
            where the first element of the tuple is a value and the second element is a condition.
        select_attrs : list
            A list of attributes to select in the query. If left empty, the function will
            retrieve all attributes from the table except for `cond_attr`.
        table_name : str
            The name of the new table that will be created. If left empty, the function will
            generate a new name.
        Returns
        -------
        str
            The name of the new table.
        """

        if not select_attrs:
            attrs = self._execute_query("PRAGMA table_info(" + from_table + ")")
            select_attrs = [attr[1] for attr in attrs if attr != cond_attr]
        if not table_name:
            view = self.get_next_name()
        else:
            view = table_name
        sql = "CREATE OR REPLACE TABLE " + view + " AS\n"
        sql += "SELECT " + ",".join(select_attrs) + ","
        sql += base_val
        for case_definition in case_definitions:
            sql += operator + "\nCASE\n"
            for val, cond in case_definition:
                sql += (
                    " WHEN "
                    + " AND ".join(cond)
                    + " THEN CAST("
                    + str(val)
                    + " AS DOUBLE)\n"
                )
            sql += "ELSE 0 END\n"
        sql += "AS " + cond_attr + " FROM " + from_table
        self._execute_query(sql)
        return view

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

    def add_table(self, table: str, table_address):
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

    def execute_spja_query(self, spja_data: SPJAData) -> Any:
        """
        Executes an SPJA query using the current object's database connection.

        Parameters
        ----------
        spja_data : SPJAData
            The SPJAData object containing the query parameters.

        Returns
        -------
        Any
            The result of the query.
        """
        spja = self.spja_query(spja_data, parenthesize=False)
        return self._execute_query(spja)

    def spja_query_to_table(self, spja_data: SPJAData) -> str:
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
            "CREATE " + ("OR REPLACE " if spja_data.replace else "")
            + entity_type_ + name_ + " AS "
        )
        sql += spja
        self._execute_query(sql)
        return name_

    def spja_query_as_view(
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
        if spja_data.order_by is not None:
            sql += "ORDER BY " + spja_data.order_by + "\n"
        if spja_data.limit is not None:
            sql += "LIMIT " + str(spja_data.limit) + "\n"
        if spja_data.sample_rate is not None:
            sql += "USING SAMPLE " + str(spja_data.sample_rate * 100) + " %\n"

        if parenthesize:
            sql = f'({sql})'

        return sql

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

        window_clause = " OVER joinboost_window " if window_by else ""
        rename_expr = " AS " + target_col if target_col is not None else ""
        parsed_expression = (
            aggregator.parse_agg(agg, para) + window_clause + rename_expr
        )

        return parsed_expression


class PandasExecutor(DuckdbExecutor):
    def add_table(self, table: str, table_address):
        if table_address is None:
            raise ExecutorException("Please pass in the pandas dataframe!")
        self.conn.register(table, table_address)
