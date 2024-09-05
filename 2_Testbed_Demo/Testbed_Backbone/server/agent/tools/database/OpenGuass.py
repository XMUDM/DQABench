import os
import sys
from configparser import ConfigParser
from typing import List
import psycopg2 as pg
import pandas as pd
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
from configs.database_config import DATABASE_CONFIG 

class OG:
    def __init__(self):
        defaults = DATABASE_CONFIG[0]
        self.host = defaults['pg_ip']
        self.port = defaults['pg_port']
        self.user = defaults['pg_user']
        self.password = defaults['pg_password']
        self.database = defaults['database']
        self.conn = pg.connect(database=self.database, user=self.user, password=self.password, host=self.host,
                                     port=self.port)

    def close(self):
        self.conn.close()

    def get_queries_cost(self, query_list):
        cost_list: List[float] = list()
        cur = self.conn.cursor()
        for i, query in enumerate(query_list):
            query = "explain " + query
            cur.execute(query)
            rows = cur.fetchall()
            df = pd.DataFrame(rows)
            cost_info = str(df[0][0])
            cost_list.append(float(cost_info[cost_info.index("..") + 2:cost_info.index(" rows=")]))
        return cost_list

    def get_storage_cost(self, oid_list):
        costs = list()
        cur = self.conn.cursor()
        for i, oid in enumerate(oid_list):
            if oid == 0:
                continue
            sql = "select * from hypopg_relation_size(" + str(oid) +");"
            cur.execute(sql)
            rows = cur.fetchall()
            df = pd.DataFrame(rows)
            cost_info = str(df[0][0])
            cost_long = int(cost_info)
            costs.append(cost_long)
            # print(cost_long)
        return costs

    def execute_sql(self, sql):
        cur = self.conn.cursor()
        cur.execute(sql)
        result = cur.fetchall()
        self.conn.commit()
        return result

    def delete_indexes(self):
        sql = 'select * from hypopg_reset();'
        self.execute_sql(sql)

    def get_sel(self, table_name, condition):
        cur = self.conn.cursor()
        totalQuery = "select * from " + table_name + ";"
        cur.execute("EXPLAIN " + totalQuery)
        rows = cur.fetchall()[0][0]
        #     print(rows)
        #     print(rows)
        total_rows = int(rows.split("rows=")[-1].split(" ")[0])

        resQuery = "select * from " + table_name + " Where " + condition + ";"
        # print(resQuery)
        cur.execute("EXPLAIN  " + resQuery)
        rows = cur.fetchall()[0][0]
        #     print(rows)
        select_rows = int(rows.split("rows=")[-1].split(" ")[0])
        return select_rows/total_rows

    def get_rel_cost(self, query_list):
        print("real")
        cost_list: List[float] = list()
        cur = self.conn.cursor()
        for i, query in enumerate(query_list):
            _start = time.time()
            query = "explain analyse" + query
            cur.execute(query)
            _end = time.time()
            cost_list.append(_end-_start)
        return cost_list

    def create_indexes(self, indexes):
        i = 0
        for index in indexes:
            schema = index.split("#")
            sql = 'CREATE INDEX START_X_IDx' + str(i) + ' ON ' + schema[0] + "(" + schema[1] + ');'
            print(sql)
            self.execute_sql(sql)
            i += 1

    def delete_t_indexes(self):
        sql = "SELECT relname from pg_class where relkind = 'i' and relname like 'start_x_idx%';"
        print(sql)
        cur = self.conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        indexes = []
        for row in rows:
            indexes.append(row[0])
        print(indexes)
        for index in indexes:
            sql = 'drop index ' + index + ';'
            print(sql)
            self.execute_sql(sql)

    def get_tables_info(self, schema):
        tables_sql = 'select tablename from pg_tables where schemaname=\''+schema+'\';'
        cur = self.conn.cursor()
        cur.execute(tables_sql)
        rows = cur.fetchall()
        table_names = list()
        for i, table_name in enumerate(rows):
            table_names.append(table_name[0])

        return table_names

    def get_columns_info(self, table_name, schema):
        attrs_sql = 'select column_name, data_type from information_schema.columns where table_schema=\''+schema+'\' and table_name=\''+table_name+'\''
        cur = self.conn.cursor()
        cur.execute(attrs_sql)
        rows = cur.fetchall()
        attrs = list()
        for i, attr in enumerate(rows):
            info = str(attr[0]) + "#" + str(attr[1])
            attrs.append(info)
        return attrs
    
    # def get_cardinality(self, table_name, column_name, schema):
    #     cardinality_sql = 'SELECT' + table_name + ' , ' + column_name + ', COUNT(DISTINCT' + column_name') as cardinality FROM information_schema.columns WHERE table_schema =\'' + schema +'\' GROUP BY table_name, column_name;'
    #     cur = self.conn.cursor()
    #     cur.execute(cardinality_sql)
    #     result = cur.fetchall()
    #     return result
    
    def get_key(self,schema):
        '''
        Get the constraint relationship between tables in the database
        '''
        key_sql = 'SELECT constraint_name, table_name, constraint_type FROM information_schema.table_constraints WHERE (constraint_type = \'PRIMARY KEY\' OR constraint_type = \'FOREIGN KEY\');;'
        cur = self.conn.cursor()
        cur.execute(key_sql)
        keys = cur.fetchall()
        return keys


if __name__ == "__main__":
    database = OG()
    # print(database.get_tables_info("public"))
    # print(database.get_columns_info("lineitem","public"))
    # print(database.get_key("public"))
    print(database.execute_sql("SELECT COUNT(DISTINCT l_orderkey) FROM lineitem;"))
    database.close()