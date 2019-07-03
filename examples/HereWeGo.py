# -*- coding:utf-8 -*-
import os
import sys
import numpy as np
print(os.getcwd())
sys.path.append("D:\OneDrive\Project\woe")
import woe.feature_process as fp
import woe.GridSearch as gs

if __name__ == '__main__':
    print("current workdir : {0}".format(os.getcwd()))
    config_path =os.path.join(os.getcwd(), 'examples\\config.csv') # 配置文件
    data_path =os.path.join( os.getcwd(), 'examples\\UCI_Credit_Card.csv') # 训练样本
    feature_detail_path =os.path.join(os.getcwd(), 'examples\\features_detail.csv') # 保存变量切分后的详细信息
    rst_pkl_path = os.path.join(os.getcwd(), 'examples\\woe_rule.pkl')
    # 
    feature_detail,rst = fp.process_train_woe(infile_path=data_path
                                           ,outfile_path=feature_detail_path
                                           ,rst_path=rst_pkl_path
                                           ,config_path=config_path)
    # proc woe transformation
    woe_train_path = os.path.join(os.getcwd(), 'examples\\dataset_train_woed.csv')
    fp.process_woe_trans(data_path,rst_pkl_path,woe_train_path,config_path)
    # # here i take the same dataset as test dataset
    # woe_test_path = os.path.join(os.getcwd(), 'examples\\dataset_test_woed.csv')
    # fp.process_woe_trans(data_path,rst_pkl_path,woe_test_path,config_path)

    # print('###TRAIN SCORECARD MODEL###')
    # params = {}
    # params['dataset_path'] = woe_train_path
    # params['validation_path'] = woe_test_path
    # params['config_path'] = config_path

    # params['df_coef_path'] = os.getcwd()+'\\df_model_coef_path.csv'
    # params['pic_coefpath'] = os.getcwd()+'\\model_coefpath.png'
    # params['pic_performance'] = os.getcwd()+'\\model_performance_path.png'
    # params['pic_coefpath_title'] = 'model_coefpath'
    # params['pic_performance_title'] = 'model_performance_path'

    # params['var_list_specfied'] = []
    # params['cs'] = np.logspace(-4, -1,40)
    # for key,value in params.items():
    #     print(key,': ',value)
    # gs.grid_search_lr_c_main(params)
