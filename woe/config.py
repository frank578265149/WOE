# -*- coding:utf-8 -*-
__author__ = 'lin.xiong@beibei.com'
import pandas as pd

class config:

    def __init__(self):
        self.config = None
        self.dataset_train = None
        self.variable_type = None
        self.bin_var_list = None
        self.discrete_var_list = None
        self.candidate_var_list = None
        self.dataset_len = None
        self.min_sample = None
        self.global_bt = None
        self.global_gt = None

    def load_file(self,config_path,data_path=False):
        self.config = pd.read_csv(config_path)
        # 指定变量的数据类型
        self.variable_type = self.config[['var_name', 'var_dtype']]
        self.variable_type = self.variable_type.rename(columns={'var_name': 'v_name', 'var_dtype': 'v_type'})
        self.variable_type = self.variable_type.set_index(['v_name'])

        # 指定需要分组的连续变量列表
        self.bin_var_list = self.config[self.config['is_tobe_bin'] == 1]['var_name']
        # 指定需要分组的离散变量列表
        self.discrete_var_list = self.config[(self.config['is_candidate'] == 1) & (self.config['var_dtype'] == 'object')]['var_name']

        # 指定需要分组的变量
        self.candidate_var_list = self.config[self.config['is_candidate'] == 1]['var_name']

        if data_path:
            data_path = data_path if isinstance(data_path, str) else None

            self.dataset_train = pd.read_csv(data_path)
            self.dataset_train.columns = [col.split('.')[-1] for col in self.dataset_train.columns]

            self.dataset_len = len(self.dataset_train)
            self.min_sample = int(self.dataset_len * 0.05)
            self.global_bt = sum(self.dataset_train['target'])
            self.global_gt = len(self.dataset_train) - sum(self.dataset_train['target'])

    def change_config_var_dtype(self,var_name,type,inplace_file=True):
        if type in ['object','string','int64','uint8','float64','bool1','bool2','dates','category']:
            self.variable_type.loc[var_name,'v_type'] = type
        else:
            raise KeyError("Invalid dtype specified! ")