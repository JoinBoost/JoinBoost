from abc import ABC, abstractmethod
import time

from joinboost import aggregator


class ExecutorException(Exception):
    pass


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
    def __init__(self, conn, debug=False):
        super().__init__()
        self.conn = conn
        self.debug = debug

    def get_schema(self, table: str) -> list:
        # duckdb stores table info in [cid, name, type, notnull, dflt_value, pk]
        table_info = self._execute_query("PRAGMA table_info(" + table + ")")
        return [x[1] for x in table_info]

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
        if not table.startswith(self.prefix):
            raise Exception("Don't modify user tables!")

    def add_table(self, table: str, table_address):
        # TODO
        ...

    def update_query(self, update_expression, table, select_conds: list = []):
        self.check_table(table)
        sql = "UPDATE " + table + " SET " + update_expression + " \n"
        if len(select_conds) > 0:
            sql += "WHERE " + " AND ".join(select_conds) + "\n"
        self._execute_query(sql)

    # mode = 1 will write the query result to a table and return table name, now execute_spja_query_to_table
    # mode = 2 will create the query as view and return view name, now execute_spja_query_as_view
    # mode = 3 will execute the query and return the result, now execute_spja_query
    # mode = 4 will create the sql query and return the query (for nested query), not needed, for SPJA query (be sure to add parens on the outside)
    def execute_spja_query(
        self,
        aggregate_expressions: dict = {None: ("*", aggregator.Aggregator.IDENTITY)},
        from_tables: list = [],
        select_conds: list = [],
        group_by: list = [],
        window_by: list = [],
        order_by: str = None,
        limit: int = None,
        sample_rate: float = None,
        replace: bool = True,
        mode: int = 4,
    ):

        spja = self.spja_query(
            aggregate_expressions=aggregate_expressions,
            from_tables=from_tables,
            select_conds=select_conds,
            group_by=group_by,
            window_by=window_by,
            order_by=order_by,
            limit=limit,
            sample_rate=sample_rate,
        )

        return self._execute_query(spja)

    def execute_spja_query_to_table(
        self,
        aggregate_expressions: dict = {None: ("*", aggregator.Aggregator.IDENTITY)},
        from_tables: list = [],
        select_conds: list = [],
        group_by: list = [],
        window_by: list = [],
        order_by: str = None,
        limit: int = None,
        sample_rate: float = None,
        replace: bool = True,
    ):

        spja = self.spja_query(
            aggregate_expressions=aggregate_expressions,
            from_tables=from_tables,
            select_conds=select_conds,
            group_by=group_by,
            window_by=window_by,
            order_by=order_by,
            limit=limit,
            sample_rate=sample_rate,
        )

        name_ = self.get_next_name()
        entity_type_ = "TABLE "
        sql = (
            "CREATE "
            + ("OR REPLACE " if replace else "")
            + entity_type_
            + name_
            + " AS "
        )
        sql += spja
        self._execute_query(sql)
        return name_

    def execute_spja_query_as_view(
        self,
        aggregate_expressions: dict = {None: ("*", aggregator.Aggregator.IDENTITY)},
        from_tables: list = [],
        select_conds: list = [],
        group_by: list = [],
        window_by: list = [],
        order_by: str = None,
        limit: int = None,
        sample_rate: float = None,
        replace: bool = True,
    ):

        spja = self.spja_query(
            aggregate_expressions=aggregate_expressions,
            from_tables=from_tables,
            select_conds=select_conds,
            group_by=group_by,
            window_by=window_by,
            order_by=order_by,
            limit=limit,
            sample_rate=sample_rate,
        )

        name_ = self.get_next_name()
        entity_type_ = "VIEW "
        sql = (
            "CREATE "
            + ("OR REPLACE " if replace else "")
            + entity_type_
            + name_
            + " AS "
        )
        sql += spja
        self._execute_query(sql)
        return name_

    def spja_query(
        self,
        aggregate_expressions: dict,
        from_tables: list = [],
        select_conds: list = [],
        window_by: list = [],
        group_by: list = [],
        order_by: str = None,
        limit: int = None,
        sample_rate: float = None,
    ):
        # TODO: remove default list params

        parsed_aggregate_expressions = []
        for target_col, (para, agg) in aggregate_expressions.items():
            parsed_expression = self._parse_aggregate_expression(
                target_col, para, agg, window_by=window_by
            )
            parsed_aggregate_expressions.append(parsed_expression)

        sql = "SELECT " + ", ".join(parsed_aggregate_expressions) + "\n"
        sql += "FROM " + ",".join(from_tables) + "\n"

        if len(select_conds) > 0:
            sql += "WHERE " + " AND ".join(select_conds) + "\n"
        if len(window_by) > 0:
            sql += "WINDOW joinboost_window AS (ORDER BY " + ",".join(window_by) + ")\n"
        if len(group_by) > 0:
            sql += "GROUP BY " + ",".join(group_by) + "\n"
        if order_by is not None:
            sql += "ORDER BY " + order_by + "\n"
        if limit is not None:
            sql += "LIMIT " + str(limit) + "\n"
        if sample_rate is not None:
            sql += "USING SAMPLE " + str(sample_rate * 100) + " %\n"
        return sql

    def _execute_query(self, q):
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
