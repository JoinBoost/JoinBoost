import itertools
import re
from abc import ABC, abstractmethod

import pandas as pd

from .aggregator import *
import time

class ExecutorException(Exception):
    pass


def ExecutorFactory(con=None):
    # By default if con is not specified, user uses Pandas dataframe
    if con is None:
        try:
            import duckdb
        except:
            raise ExecutorException("To support Pandas dataframe, please install duckdb.")
        con = duckdb.connect(database=':memory:')
        return PandasExecutor(con)
    elif issubclass(type(con), Executor):
        return con
    elif type(con).__name__ == 'DuckDBPyConnection':
        return DuckdbExecutor(con)
    else:
        raise ExecutorException("Unknown connector with type " + type(con).__name__)


class Executor(ABC):
    '''Assume input data are csvs'''

    def __init__(self):
        self.view_id = 0
        self.prefix = 'joinboost_tmp_'

    def get_next_name(self):
        name = self.prefix + str(self.view_id)
        self.view_id += 1
        return name

    def get_schema(self, table):
        pass

    def select_all(self, table):
        pass

    def add_table(self, table: str, table_address):
        pass

    def delete_table(self, table: str):
        pass

    def load_table(self, table_name: str, data_dir: str):
        pass

    def case_query(self, from_table: str, operator: str, cond_attr: str, base_val: str,
                   case_definitions: list, select_attrs: list = [], table_name: str = None):
        pass

    def window_query(self, view: str, select_attrs: list, base_attr: str, cumulative_attrs: list):
        pass

    def execute_spja_query(self,
                           mode: int,
                           aggregate_expressions: dict,
                           in_msgs: list = [],
                           from_table: str = '',
                           group_by: list = [],
                           where_conds: dict = {},
                           annotations: list = [],
                           left_join: dict = {},
                           table_name: str = '',
                           replace: bool = True) -> str:
        pass



class DuckdbExecutor(Executor):
    def __init__(self, conn, debug=False):
        super().__init__()
        self.conn = conn
        self.debug = debug

    def get_schema(self, table):
        # duckdb stores table info in [cid, name, type, notnull, dflt_value, pk]
        table_info = self._execute_query('PRAGMA table_info(' + table + ')')
        return [x[1] for x in table_info]

    def _gen_sql_case(self, leaf_conds: list):
        conds = []
        for leaf_cond in leaf_conds:
            cond = 'CASE\n'
            for (pred, annotations) in leaf_cond:
                cond += ' WHEN ' + ' AND '.join(annotations) + \
                        ' THEN CAST(' + str(pred) + ' AS DOUBLE)\n'
            cond += 'ELSE 0 END\n'
            conds.append(cond)
        return conds

    def delete_table(self, table: str):
        self.check_table(table)
        sql = 'DROP TABLE IF EXISTS ' + table + ';\n'
        self._execute_query(sql)

    # TODO: remove it
    def window_query(self, view: str, select_attrs: list, base_attr: str, cumulative_attrs: list):
        view_name = self.get_next_name()
        sql = 'CREATE OR REPLACE VIEW ' + view_name + ' AS SELECT * FROM\n'
        sql += '(\nSELECT ' + ",".join(select_attrs)
        for attr in cumulative_attrs:
            sql += ',SUM(' + attr + ') OVER joinboost_window as ' + attr
        sql += '\nFROM ' + view
        sql += ' WINDOW joinboost_window AS (ORDER BY ' + base_attr + ')\n)'
        self._execute_query(sql)
        return view_name

    # {case: value} operator {case: value} ...
    def case_query(self, from_table: str, operator: str, cond_attr: str, base_val: str,
                   case_definitions: list, select_attrs: list = [], table_name: str = None):
        # print(conditions)
        if not select_attrs:
            attrs = self._execute_query('PRAGMA table_info(' + from_table + ')')
            for attr in attrs:
                if attr != cond_attr: select_attrs.append(attr[1])
        if not table_name:
            view = self.get_next_name()
        else:
            view = table_name
        sql = 'CREATE OR REPLACE TABLE ' + view + ' AS\n'
        sql += 'SELECT ' + ','.join(select_attrs) + ','
        sql += base_val
        for case_definition in case_definitions:
            sql += operator + '\nCASE\n'
            for val, cond in case_definition:
                sql += ' WHEN ' + ' AND '.join(cond) + ' THEN CAST(' + str(val) + ' AS DOUBLE)\n'
            sql += 'ELSE 0 END\n'

        sql += 'AS ' + cond_attr + ' FROM ' + from_table
        self._execute_query(sql)
        print(view)
        return view

    # Write a method that can generate a function based on the case definitions
    # The function will take in a row and return a value
    # This function can be used to generate a new column
    # This function will not use SQL or the database and instead will be run in pandas dataframes
    def case_function(self, from_table: str, operator: str, cond_attr: str, base_val: str,
                     case_definitions: list, select_attrs: list = [], table_name: str = None):

        def case_function(row):
            result = base_val
            predicates = []
            for case_definition in case_definitions:

                for val, conds in case_definition:
                    # each cond in conds is a string of the form "attr =/>=/</<=/> val"
                    # we need to split this string and then check if the row[attr] satisfies the condition
                    temp = []
                    for i, cond in enumerate(conds):
                        attr, op, val = cond.split()
                        temp += ["row['" + attr + "'] " + op + " " + val]
                    val + " if  (" + " and ".join(temp) + ") else 0"

            return result


    def check_table(self, table):
        if not table.startswith(self.prefix):
            raise Exception("Don't modify user tables!")

    def update_query(self,
                     update_expression,
                     table,
                     select_conds: list = []):
        self.check_table(table)
        sql = "UPDATE " + table + " SET " + update_expression + " \n"
        if len(select_conds) > 0:
            sql += "WHERE " + " AND ".join(select_conds) + "\n"
        self._execute_query(sql)

    # mode = 1 will write the query result to a table and return table name
    # mode = 2 will create the query as view and return view name
    # mode = 3 will execute the query and return the result
    # mode = 4 will create the sql query and return the query (for nested query)
    # TODO: standardize the join conds, currently it is just a list of sql oriented strings
    def execute_spja_query(self,
                           # By default, we select all
                           aggregate_expressions: dict = {None: ('*', Aggregator.IDENTITY)},
                           from_tables: list = [],
                           join_conds: list = [],
                           select_conds: list = [],
                           group_by: list = [],
                           window_by: list = [],
                           order_by: list = [],
                           limit: int = None,
                           sample_rate: float = None,
                           replace: bool = True,
                           join_type: str = 'INNER',
                           mode: int = 4):

        spja = self.spja_query(aggregate_expressions=aggregate_expressions,
                               from_tables=from_tables,
                               join_conds=join_conds,
                               select_conds=select_conds,
                               group_by=group_by,
                               window_by=window_by,
                               order_by=order_by,
                               limit=limit,
                               sample_rate=sample_rate)

        if mode == 1:
            name_ = self.get_next_name()
            entity_type_ = 'TABLE '
            sql = 'CREATE ' + ('OR REPLACE ' if replace else '') + entity_type_  + name_ + ' AS '
            sql += spja
            self._execute_query(sql)
            return name_

        elif mode == 2:
            name_ = self.get_next_name()
            entity_type_ = 'VIEW '
            sql = 'CREATE ' + ('OR REPLACE ' if replace else '') + entity_type_  + name_ + ' AS '
            sql += spja
            self._execute_query(sql)
            return name_

        elif mode == 3:
            return self._execute_query(spja)

        elif mode == 4:
            sql = '(' + spja + ')'
            return sql

        else:
            raise ExecutorException('Unsupported mode for query execution!')



    def spja_query(self,
                   aggregate_expressions: dict,
                   from_tables: list = [],
                   join_conds: list = [],
                   select_conds: list = [],
                   window_by: list = [],
                   group_by: list = [],
                   order_by: str = None,
                   limit: int = None,
                   sample_rate: float = None,
                   ):

        parsed_aggregate_expressions = []
        for target_col, aggregation_spec in aggregate_expressions.items():
            para, agg = aggregation_spec
            # check if para is in the form of table.column using regex
            # if re.match(r'^[a-zA-Z0-9_]+\.[a-zA-Z0-9_]+$', para):
            #     para = para.split('.')[1]


            parsed_aggregate_expressions.append(parse_agg(agg, para) \
                                + (' OVER joinboost_window ' if len(window_by) > 0 and is_agg(agg) else '')\
                                + (' AS ' + target_col if target_col is not None else ''))


        sql = 'SELECT ' + ', '.join(parsed_aggregate_expressions) + '\n'
        sql += "FROM " + ",".join(from_tables) + '\n'

        if len(select_conds) > 0:
            sql += "WHERE " + " AND ".join(select_conds) + '\n'
        if len(window_by) > 0:
            sql += 'WINDOW joinboost_window AS (ORDER BY ' + ','.join(window_by) + ')\n'
        if len(group_by) > 0:
            sql += "GROUP BY " + ",".join(group_by) + '\n'
        if len(order_by) > 0:
            # generate multiple order bys with asc and desc from order_by
            sql += 'ORDER BY ' + ",".join([f"{col} {order}" for (col, order) in order_by]) + '\n'
        if limit is not None:
            sql += 'LIMIT ' + str(limit) + '\n'
        if sample_rate is not None:
            sql += 'USING SAMPLE ' + str(sample_rate*100) + ' %\n'
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

class PandasExecutor(DuckdbExecutor):
    table_registry = {}

    def __init__(self, conn, debug=False):
        super().__init__(conn)
        self.debug = debug
        self.prefix = 'joinboost_'
        self.table_counter = 0

    def add_table(self, table: str, table_address):
        if table_address is None:
            raise ExecutorException("Please pass in the pandas dataframe!")

        # check if the table_address is a string path
        if isinstance(table_address, str):
            table_address = pd.read_csv(table_address)
        self.table_registry[table] = table_address

    def get_schema(self, table):
        # unqualify the column names, this is required as duckdb returns unqualified column names
        return [col.split('.')[-1] for col in self.table_registry[table].columns]


    # mode 1: write the query result to a table and return table name
    # mode 2: same as mode 1
    # mode 3: execute the query and return the result
    # mode 4: same as mode 1 (for now)
    def execute_spja_query(self,
                           aggregate_expressions: dict = {None: ('*', Aggregator.IDENTITY)},
                           from_tables: list = [],
                           join_conds: list = [],
                           select_conds: list = [],
                           window_by: list = [],
                           group_by: list = [],
                           order_by: list = [],
                           limit: int = None,
                           sample_rate: float = None,
                           replace: bool = True,
                           join_type: str = 'INNER',
                           mode: int = 4
                           ):
        intermediates = {}
        for table in from_tables:
            intermediates[table] = self.table_registry[table]

        agg_conditions = self.convert_agg_conditions(aggregate_expressions)

        select_conds = self.convert_predicates(select_conds)

        # join_conds are of the form "table1.col1 IS NOT DISTINCT FROM table2.col2". extract the table1.col1 and table2.col2
        join_conds = [re.findall(r'(\w+\.\w+)', cond) for cond in join_conds]

        # filter list of tables that don't have any join conditions
        tables_to_join = [table for table in from_tables if any([cond[0].startswith(table) or cond[1].startswith(table) for cond in join_conds])]

        # subtract tables_to_join from from_tables to get the tables that don't have any join conditions
        tables_to_cross = list(set(from_tables) - set(tables_to_join))

        df = self.join(intermediates, join_conds, join_type, tables_to_cross, tables_to_join)

        # filter by select_conds
        if len(select_conds) > 0:
            converted_select_conds = ' and '.join(select_conds)
            df = df.query(converted_select_conds)

        # add window_by columns to group_by
        if len(window_by) > 0:
            # qualify window_by columns with table name in from_tables (there will only be one)
            window_by = [from_tables[0] + '.' + col for col in window_by]
            if len(group_by) > 0:
                group_by += window_by
            else:
                group_by = window_by

        # group by and aggregate
        df = self.apply_group_by_and_agg(agg_conditions, df, group_by)

        # sort by each column in order_by
        if len(order_by) > 0:
            for col, order in order_by:
                df = df.sort_values(col, ascending=(order == 'ASC' or order is None))

        if mode == 1 or mode == 2 or mode == 4:
            name_ = self.get_next_name()
            print("creating table " + name_)
            # always qualify intermediate tables as future aggregations for these tables will come qualified
            for col in df.columns:
                if col not in ['s', 'c']:
                    df = df.rename(columns={col: name_ + '.' + col})
            df.name = name_
            self.table_registry[name_] = df
            return name_
        elif mode == 3:
            print("returning result")
            print(df.head())
            return df.values.tolist()
        else:
            raise ExecutorException('Unsupported mode for query execution!')

    def apply_group_by_and_agg(self, agg_conditions, df, group_by):
        if len(group_by) > 0:
            # if group_by element is not of the form joinboost_<digit>.col, then unqualify it
            for i, col in enumerate(group_by):
                if not re.match(r'^joinboost_\d+\.\w+$', col):
                    group_by[i] = col.split('.')[-1]
            inter_df = df.groupby(group_by)
            if len(agg_conditions) > 0:
                # check if column does not exist and create it before applying agg_conditions
                for col in agg_conditions.keys():
                    # generate unqualified names in df.columns
                    unqualified_cols = [col.split('.')[-1] for col in df.columns]
                    if col not in df.columns and col not in unqualified_cols:
                        df[col] = 1
                df = inter_df.agg(**agg_conditions).reset_index()
                # unqualify all columns in df. This is to avoid nested columns being qualified with the table name
                for col in df.columns:
                    df = df.rename(columns={col: col.split('.')[-1]})
        else:
            if len(agg_conditions) > 0:
                # check if column does not exist and create it before applying agg_conditions (for s anc c)
                for col in agg_conditions.keys():
                    if col not in df.columns:
                        df[col] = 1

                # check if column is * and apply aggfunc to the entire row
                for col in list(agg_conditions.keys()):
                    if agg_conditions[col].column == '*':
                        func = agg_conditions[col].aggfunc
                        df[col] = df.apply(func, axis=1)
                        del agg_conditions[col]
                if len(agg_conditions) > 0:
                    df = df.assign(temp=0).groupby('temp').agg(**agg_conditions).reset_index().drop(columns='temp')
                for col in df.columns:
                    df = df.rename(columns={col: col.split('.')[-1]})
        return df

    # computes join or cross (if no join condition) between all tables.
    def join(self, intermediates, join_conds, join_type, tables_to_cross, tables_to_join):
        df = None
        # handle cross joins
        if len(tables_to_cross) > 0:
            for table in tables_to_cross:
                if df is None:
                    df = intermediates[table]
                else:
                    df = df.merge(intermediates[table], how='cross',
                                  suffixes=('', '_drop')).filter(regex='^(?!.*_drop)')
        # generate pairwise combination of tables to join
        for table1, table2 in itertools.combinations(tables_to_join, 2):
            # search join_conds for the join condition between table1 and table2
            for cond in join_conds:
                temp = None
                if (cond[0] in intermediates[table1].columns and cond[1] in intermediates[table2].columns) or \
                        (cond[1] in intermediates[table1].columns and cond[0] in intermediates[table2].columns):
                    # join the two tables

                    temp = intermediates[table1].merge(intermediates[table2],
                                                       how=join_type.lower(), left_on=cond[0], right_on=cond[1],
                                                       suffixes=('', '_drop')).filter(regex='^(?!.*_drop)')

                    # check join conditions when unqualified column names are used
                elif (cond[0].split('.')[1] in intermediates[table1].columns and cond[1].split('.')[1] in intermediates[
                    table2].columns) or \
                        (cond[1].split('.')[1] in intermediates[table1].columns and cond[0].split('.')[1] in
                         intermediates[table2].columns):
                    # join the two tables
                    temp = intermediates[table1].merge(intermediates[table2],
                                                       how=join_type.lower(), left_on=cond[0].split('.')[1],
                                                       right_on=cond[1].split('.')[1],
                                                       suffixes=('', '_drop')).filter(regex='^(?!.*_drop)')
                    # check if one join condition is qualified and the other is not
                elif (cond[0] in intermediates[table1].columns and cond[1].split('.')[1] in intermediates[
                    table2].columns):
                    # join the two tables
                    temp = intermediates[table1].merge(intermediates[table2],
                                                       how=join_type.lower(), left_on=cond[0],
                                                       right_on=cond[1].split('.')[1],
                                                       suffixes=('', '_drop')).filter(regex='^(?!.*_drop)')
                elif (cond[1] in intermediates[table2].columns and cond[0].split('.')[1] in intermediates[
                    table1].columns):
                    # join the two tables
                    temp = intermediates[table1].merge(intermediates[table2],
                                                       how=join_type.lower(), left_on=cond[1],
                                                       right_on=cond[0].split('.')[1],
                                                       suffixes=('', '_drop')).filter(regex='^(?!.*_drop)')
            if temp is not None:
                if df is None:
                    df = temp
                else:
                    df = df.merge(temp, how='cross', suffixes=('', '_drop')).filter(regex='^(?!.*_drop)')
                break

        return df

    def convert_predicates(self, select_conds):
        # TODO: handle in and not in
        # replace ' = ' with ' == ' but only if = is not part of <> or <= or >=
        select_conds = [re.sub(r'(?<!<|>)=(?!=)', '==', cond) for cond in select_conds]
        # ignore predicates of the form 's.a is not distinct from t.b'
        select_conds = [cond for cond in select_conds if 'DISTINCT' not in cond]
        # check if predicates do not start with joinboost_<number>.col <op> <value> and if so, remove the table name
        for i, cond in enumerate(select_conds):
            if not re.match(r'^joinboost_', cond):
                # split by dot and remove only the first element and return everything else as it is
                select_conds[i] = '.'.join(cond.split('.')[1:])

        # wrap each operand (of the form identifier.identifier or identifier) with backticks, ignore any multi-digit numeric values
        select_conds = [re.sub(r'\b([a-zA-Z_0-9]+\.[a-zA-Z_]+|[a-zA-Z_]+)\b', r'`\g<1>`', cond) for cond in select_conds]

        # wrap each select condition with parentheses
        select_conds = ['(' + cond + ')' for cond in select_conds]

        return select_conds

    def convert_agg_conditions(self, aggregate_expressions):
        agg_conditions = {}
        # handle aggregate expressions
        for target_col, aggregation_spec in aggregate_expressions.items():
            para, agg = aggregation_spec

            if target_col is None:
                target_col = para

            # check if column is a number in string form, in that case use target_col as the column name
            if str(para).isnumeric():
                para = target_col

            # use named aggregation and column renaming with dictionary
            if agg == Aggregator.COUNT:
                agg_conditions[target_col] = pd.NamedAgg(column=para, aggfunc='count')
            elif agg == Aggregator.SUM:
                agg_conditions[target_col] = pd.NamedAgg(column=para, aggfunc='sum')
            elif agg == Aggregator.MAX:
                agg_conditions[target_col] = pd.NamedAgg(column=para, aggfunc='max')
            elif agg == Aggregator.MIN:
                agg_conditions[target_col] = pd.NamedAgg(column=para, aggfunc='min')
            elif agg == Aggregator.IDENTITY:
                # don't do anything, these are automatically handled as the default
                pass
            elif agg == Aggregator.IDENTITY_LAMBDA:
                agg_conditions[target_col] = pd.NamedAgg(column='*', aggfunc=para)

            else:
                raise ExecutorException('Unsupported aggregation function!')
        return agg_conditions
