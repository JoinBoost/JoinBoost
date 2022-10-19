from abc import ABC, abstractmethod
from aggregator import *
import time

class Executor(ABC):
    '''Assume input data are csvs'''

    def __init__(self):
        self.view_id = 0

    def get_next_name(self):
        name = 'joinboost_tmp_' + str(self.view_id)
        self.view_id += 1
        return name
    
    def get_schema(self, table):
        pass
    
    def select_all(self, table):
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

    def select_all(self, table: str):
        return self.execute_spja_query({None: ('*', Aggregator.IDENTITY)}, 
                                       from_tables=[table], 
                                       mode=3)
    
    def delete_table(self, table: str):
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
        return view
    
    # mode = 1 will create the query result as table and return table name
    # mode = 2 will create the query result as table and return view name
    # mode = 3 will create the query result and return the result
    # mode = 4 will create the sql query and return the query (for nested query)
    def execute_spja_query(self, 
                           aggregate_expressions: dict = {},
                           from_tables: list = [],
                           select_conds: list = [],
                           group_by: list = [], 
                           window_by: list = [],
                           table_name: str = None,
                           order_by: str = None,
                           limit: int = None,
                           replace: bool = True,
                           mode: int = 5):
        
        spja = self.spja_query(aggregate_expressions=aggregate_expressions,
                               from_tables=from_tables,
                               select_conds = select_conds,
                               window_by=window_by,
                               group_by=group_by, 
                               order_by=order_by,
                               limit=limit,)
        
        if mode == 1:
            name_ = (table_name if table_name is not None else self.get_next_name())
            entity_type_ = 'TABLE '
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
            name_ = (table_name if table_name is not None else self.get_next_name())
            entity_type_ = 'TABLE '
            sql = 'CREATE ' + ('OR REPLACE ' if replace else '') + entity_type_  + name_ + ' AS '
            sql += spja
            self._execute_query(sql)
            return name_
        
    
    
    def spja_query(self, 
                   aggregate_expressions: dict,
                   from_tables: list = [],
                   select_conds: list = [],
                   window_by: list = [],
                   group_by: list = [], 
                   order_by: str = None,
                   limit: int = None,
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
        if limit  is not None:
            sql += 'LIMIT ' + str(limit) + '\n'
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
        except Exception as e:
            print(e)
        return result

