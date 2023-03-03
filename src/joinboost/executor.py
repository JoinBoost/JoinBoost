from abc import ABC, abstractmethod
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
    
    def case_query(self, from_table: str, operator: str, cond_attr: str, base_val: str,
                   case_definitions: list, select_attrs: list = [], table_name: str = None, order_by: str = None):
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
        
        # If no select attributes are provided, retrieve all columns
        # except the one used in the conditions of the case statement
        if not select_attrs:
            attrs = self._execute_query('PRAGMA table_info(' + from_table + ')')
            for attr in attrs:
                if attr != cond_attr: select_attrs.append(attr[1])
                    
        # If no table name is provided, generate a new one
        if not table_name:
            view = self.get_next_name()
        else:
            view = table_name
            
        # Prepare the case statement using the provided operator
        cases = []
        for case_definition in case_definitions:
            sql_case = f'{operator}\nCASE\n'
            for val, cond in case_definition:
                conds = ' AND '.join(cond)
                sql_case += f' WHEN {conds} THEN CAST({val} AS DOUBLE)\n'
            sql_case += 'ELSE 0 END\n'
            cases.append(sql_case)
        sql_cases = ''.join(cases)
        
        # Create the SELECT statement with the CASE statement
        attrs = ",".join(select_attrs)
        sql = f'CREATE OR REPLACE TABLE {view} AS\n' + \
              f'SELECT {attrs}, {base_val}' + \
              f'{sql_cases}' + \
              f'AS {cond_attr} FROM {from_table} '
        if order_by:
              sql += f'ORDER BY {order_by};'
        self._execute_query(sql)
        return view

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
    def execute_spja_query(self, 
                           # By default, we select all
                           aggregate_expressions: dict = {None: ('*', Aggregator.IDENTITY)},
                           from_tables: list = [],
                           select_conds: list = [],
                           group_by: list = [], 
                           window_by: list = [],
                           order_by: str = None,
                           limit: int = None,
                           sample_rate: float = None,
                           replace: bool = True,
                           mode: int = 4):
        
        spja = self.spja_query(aggregate_expressions=aggregate_expressions,
                               from_tables=from_tables,
                               select_conds = select_conds,
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
        if order_by is not None:
            sql += 'ORDER BY ' + order_by + '\n'
        if limit is not None:
            sql += 'LIMIT ' + str(limit) + '\n'
        if sample_rate is not None:
            sql += 'USING SAMPLE ' + str(sample_rate*100) + ' %\n'
        return sql
    
    def rename(self, table, old_name, new_name):
        sql = f"ALTER TABLE {table} RENAME COLUMN {old_name} TO {new_name};"
        self._execute_query(sql)
    
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
    def add_table(self, table: str, table_address):
        if table_address is None:
            raise ExecutorException("Please pass in the pandas dataframe!")
        self.conn.register(table, table_address)