# -*- coding: utf-8 -*- 
# @Time : 2022/10/27 9:12 
# @Author : Shuyu Guo
# @File : evaluate.py 
# @contact : guoshuyu225@gmail.com
import json
import sqlite3
import os

from simtester.utils.multiwoz.nlp import normalize
from simtester.config.multiwoz.config import ROOT_PATH

DATA_PATH = os.path.join(ROOT_PATH, 'data/multiwoz/dataset/')

class MultiWozDB(object):
    # loading databases
    domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital']  # , 'police']
    dbs = {}
    CUR_DIR = os.path.dirname(__file__)

    for domain in domains:
        db = os.path.join('db/{}-dbase.db'.format(domain))
        db = os.path.join(DATA_PATH, db)
        conn = sqlite3.connect(db, check_same_thread=False)
        c = conn.cursor()
        dbs[domain] = c

    hospital_db_path = os.path.join(DATA_PATH, 'db/hospital_db.json')
    with open(hospital_db_path, 'r') as f:
        hospital_db = json.load(f)
    all_items = {}
    # for domain in domains:
    #     if domain != 'taxi' and domain != 'hospital':
    #         sql_query = "select * from {}".format(domain)
    #         all_items[domain] = dbs[domain].execute(sql_query).fetchall()

    def queryResultVenues(self, domain, turn=None, bs=None, real_belief=False):
        # query the db
        sql_query = "select * from {}".format(domain)
        # import pdb
        # pdb.set_trace()
        if turn:
            if real_belief == True:
                items = turn.items()
            else:
                items = turn['metadata'][domain]['semi'].items()

        # if bs is None:
            # return []

        if bs == {}:
            try:  # "select * from attraction  where name = 'queens college'"
                return self.dbs[domain].execute(sql_query).fetchall()
            except:
                return []  # TODO test it
        elif bs is not None:
            items = bs.items()
        #     # print(bs, turn.items())
            if len(items) == 0:
                return []
            # import pdb
            # pdb.set_trace()
        # else:
            # items = []
            # if bs['domain'] == domain:
            #     items = bs.items()
            #     bs['domain'] = ''
            # else:
                # items = []
            # items_ = bs.items()
            # items_remains = {}
            # items_all = dict(items)

            # for k, v in items_:
            #     # try:
            #         # items_remains[k] = items_all[k]
            #     # except Exception:
            #         # continue

            #     if k in items_all.keys():
            #         # items_remains[k] = items_all[k]
            #         items_remains[k] = v
            # items = items_remains.items()
            # # print(items)
            # items = items_
            # import pdb

        if domain == 'hospital':
            ret = []
            for it in self.hospital_db:
                for key, val in items:
                    if val == "" or val == "dontcare" or val == 'not mentioned' or val == "don't care" or val == "dont care" or val == "do n't care" or val == "none":
                        pass
                    else:
                        if key in it.keys() and it[key] == val:
                            ret.append(it)
            return ret
        else:
            flag = True
            for key, val in items:
                if val == "" or val == "dontcare" or val == 'not mentioned' or val == "don't care" or val == "dont care" or val == "do n't care" or val == "none":
                    pass
                else:
                    if flag:
                        sql_query += " where "
                        val2 = val.replace("'", "''")
                        val2 = normalize(val2)
                        if key.lower() == 'leaveAt'.lower():
                            sql_query += r" " + key + " > " + r"'" + val2 + r"'"
                        elif key.lower() == 'arriveBy'.lower():
                            sql_query += r" " + key + " < " + r"'" + val2 + r"'"
                        elif key.lower() == 'name':
                            sql_query += r" " + key + " like " + r"'%" + val2 + r"%'"
                        else:
                            sql_query += r" " + key + "=" + r"'" + val2 + r"'"
                        flag = False
                    else:
                        val2 = val.replace("'", "''")
                        val2 = normalize(val2)
                        if key.lower() == 'leaveAt'.lower():
                            sql_query += r" and " + key + " > " + r"'" + val2 + r"'"
                        elif key.lower() == 'arriveBy'.lower():
                            sql_query += r" and " + key + " < " + r"'" + val2 + r"'"
                        elif key.lower() == 'name':
                            sql_query += r" and " + key + " like " + r"'%" + val2 + r"%'"
                        else:
                            sql_query += r" and " + key + "=" + r"'" + val2 + r"'"

            try:  # "select * from attraction  where name = 'queens college'"
                return self.dbs[domain].execute(sql_query).fetchall()
            except:
                return []  # TODO test it
