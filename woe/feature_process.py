# -*- coding:utf-8 -*-
__author__ = 'lin.xiong@beibei.com'
import numpy as np
import woe.config as config
import woe.eval as eval
import copy
import pickle
import time

class node:
    '''
        树节点类(因为是利用决策树来做分组的)
    '''
    def __init__(self,var_name=None,iv=0,split_point=None,right=None,left=None):
        self.var_name = var_name  # The column index value of the attributes that are used to split data sets
        self.iv = iv  # 树的iv值
        self.split_point = split_point # 分裂节点
        self.right = right  
        self.left = left  

class InfoValue(object):
    '''
        IV值(定义了计算iv值的好坏样本的数量，分组信息，分组iv值)
    '''
    def __init__(self):
        self.var_name = []
        self.split_list = []
        self.iv = 0
        self.woe_list = []
        self.iv_list = []
        self.is_discrete = 0
        self.sub_total_sample_num = []
        self.positive_sample_num = []
        self.negative_sample_num = []
        self.sub_total_num_percentage = []
        self.positive_rate_in_sub_total = []
        self.negative_rate_in_sub_total = []

    def init(self,civ):
        self.var_name = civ.var_name
        self.split_list = civ.split_list
        self.iv = civ.iv
        self.woe_list = civ.woe_list
        self.iv_list = civ.iv_list
        self.is_discrete = civ.is_discrete
        self.sub_total_sample_num = civ.sub_total_sample_num
        self.positive_sample_num = civ.positive_sample_num
        self.negative_sample_num = civ.negative_sample_num
        self.sub_total_num_percentage = civ.sub_total_num_percentage
        self.positive_rate_in_sub_total = civ.positive_rate_in_sub_total
        self.negative_rate_in_sub_total = civ.negative_rate_in_sub_total

class DisInfoValue(object):
    '''
        离散变量的woe转换信息
    '''
    def __init__(self):
        self.var_name = None
        self.origin_value = []
        self.woe_before = []


def change_feature_dtype(df,variable_type):
    '''
        改变特征数据类型
    '''
    s = 'Changing Feature Dtypes'
    print(s.center(60,'-'))
    for vname in df.columns:
        try:
            df[vname] = df[vname].astype(variable_type.loc[vname,'v_type'])
            print(vname,' '*(40-len(vname)),'{0: >10}'.format(variable_type.loc[vname,'v_type']))
        except Exception:
            print('[error]',vname)
            print('[original dtype] ',df.dtypes[vname],' [astype] ',variable_type.loc[vname,'v_type'])
            print('[unique value]',np.unique(df[vname]))

    s = 'Variable Dtypes Have Been Changed'
    print(s.center(60,'-'))

    return

def check_point(df,var,split,min_sample):
    """
    检查某个分裂点是否是的落入某个组的样本数量少于总样本量的5%, 如果少于5%那就和相邻的组合并
    Applies only to continuous values
    :param df: dataframe
    :param var: 变量
    :param split: 分裂点列表
    :param min_sample: 某个组最少样本值
    :return: 一组处理后的分裂点
    """
    new_split = []
    if split is not None and split.__len__()>0:
        new_split.append(split[0])
        pdf = df[df[var] <= split[0]]
        # 首先考察第一个分裂节点
        if (pdf.shape[0] < min_sample) or (len(np.unique(pdf['target']))<=1):
            # 叶子节点样本个数小于 Min_sample 或者 这个分裂区间只有一种标签
            # 则这个分裂点就是不合法的分裂点，不考虑
            new_split.pop()
        for i in range(0,split.__len__()-1):
            pdf = df[(df[var] > split[i]) & (df[var] <= split[i+1])]
            if (pdf.shape[0] < min_sample) or (np.unique(pdf['target']).__len__()<=1):
                continue
            else:
                new_split.append(split[i+1])

        # 考察最后一个节点的样本情况
        if new_split.__len__()>1 and len(df[df[var] >= new_split[new_split.__len__()-1]])<min_sample:
            new_split.pop()
        # 如果最后分裂点后的样本只有一种标签，也需要删除这个分裂节点
        if new_split.__len__()>1 and np.unique(df[df[var] >= new_split[new_split.__len__()-1]]['target']).__len__()<=1:
            new_split.pop()
        if new_split == []:
            new_split = split
    else:
        pass
    return new_split

def calulate_iv(df,var,global_bt,global_gt):
    '''
    计算woe和iv值
    '''
    # a = df.groupby(['target']).count()
    groupdetail = {}
    bt_sub = sum(df['target'])
    bri = (bt_sub + 0.0001)* 1.0 / global_bt
    gt_sub = df.shape[0] - bt_sub
    gri = (gt_sub + 0.0001)* 1.0 / global_gt

    groupdetail['woei'] = np.log(bri / gri)
    groupdetail['ivi'] = (bri - gri) * np.log(bri / gri)
    groupdetail['sub_total_num_percentage'] = df.shape[0]*1.0/(global_bt+global_gt)
    groupdetail['positive_sample_num'] = bt_sub
    groupdetail['negative_sample_num'] = gt_sub
    groupdetail['positive_rate_in_sub_total'] = bt_sub*1.0/df.shape[0]
    groupdetail['negative_rate_in_sub_total'] = gt_sub*1.0/df.shape[0]

    return groupdetail


def calculate_iv_split(df,var,split_point,global_bt,global_gt):
    """
        计算分类某个节点时的iv值,df要有target列，不然没法计算iv值
    """
    #split dataset
    dataset_r = df[df.loc[:,var] > split_point][[var,'target']]
    dataset_l = df[df.loc[:,var] <= split_point][[var,'target']]

    r1_cnt = sum(dataset_r['target'])
    r0_cnt = dataset_r.shape[0] - r1_cnt

    l1_cnt = sum(dataset_l['target'])
    l0_cnt = dataset_l.shape[0] - l1_cnt

    if r0_cnt == 0 or r1_cnt == 0 or l0_cnt == 0 or l1_cnt ==0:
        return 0,0,0,dataset_l,dataset_r,0,0

    lbr = (l1_cnt+ 0.0001)*1.0/global_bt
    lgr = (l0_cnt+ 0.0001)*1.0/global_gt
    woel = np.log(lbr/lgr)
    ivl = (lbr-lgr)*woel
    rbr = (r1_cnt+ 0.0001)*1.0/global_bt
    rgr = (r0_cnt+ 0.0001)*1.0/global_gt
    woer = np.log(rbr/rgr)
    ivr = (rbr-rgr)*woer
    iv = ivl+ivr

    return woel,woer,iv,dataset_l,dataset_r,ivl,ivr


def binning_data_split(df,var,global_bt,global_gt,min_sample,alpha=0.01):
    """
    Specify the data split level and return the split value list
    :return:
    """
    iv_var = InfoValue()
    # Calculates the IV of the current node before splitted
    gd = calulate_iv(df, var,global_bt,global_gt)

    woei, ivi = gd['woei'],gd['ivi']

    if np.unique(df[var]).__len__() <=8:
        # print('running into if')
        split = list(np.unique(df[var]))
        split.sort()
        # print('split:',split)
        #Segmentation point checking and processing
        split = check_point(df, var, split, min_sample)
        split.sort()
        # print('after check:',split)
        iv_var.split_list = split
        return node(split_point=split,iv=ivi)

    percent_value = list(np.unique(np.percentile(df[var], range(100))))
    percent_value.sort()

    if percent_value.__len__() <=2: 
        iv_var.split_list = list(np.unique(percent_value)).sort()
        return node(split_point=percent_value,iv=ivi)

    # A sentry that attempts to split the current node
    # Init bestSplit_iv with zero
    bestSplit_iv = 0
    bestSplit_woel = []
    bestSplit_woer = []
    bestSplit_ivl = 0
    bestSplit_ivr = 0
    bestSplit_point = []

    #remove max value and min value in case dataset_r  or dataset_l will be null
    for point in percent_value[0:percent_value.__len__()-1]:
        # If there is only a sample or a negative sample, skip
        if set(df[df[var] > point]['target']).__len__() == 1 or set(df[df[var] <= point]['target']).__len__() == 1 \
                or df[df[var] > point].shape[0] < min_sample or df[df[var] <= point].shape[0] < min_sample :
            continue

        woel, woer, iv, dataset_l, dataset_r, ivl, ivr = calculate_iv_split(df,var,point,global_bt,global_gt)

        if iv > bestSplit_iv:
            bestSplit_woel = woel
            bestSplit_woer = woer
            bestSplit_iv = iv
            bestSplit_point = point
            bestSplit_dataset_r = dataset_r
            bestSplit_dataset_l = dataset_l
            bestSplit_ivl = ivl
            bestSplit_ivr = ivr

    # If the IV after division is greater than the IV value before the current segmentation, the segmentation is valid and recursive
    # specified step learning rate 0.01
    if bestSplit_iv > ivi*(1+alpha) and bestSplit_dataset_r.shape[0] > min_sample and bestSplit_dataset_l.shape[0] > min_sample:
        presplit_right = node()
        presplit_left = node()

        # Determine whether the right node satisfies the segmentation prerequisite
        if bestSplit_dataset_r.shape[0] < min_sample or set(bestSplit_dataset_r['target']).__len__() == 1:
            presplit_right.iv = bestSplit_ivr
            right = presplit_right
        else:
            right = binning_data_split(bestSplit_dataset_r,var,global_bt,global_gt,min_sample,alpha=0.01)

        # Determine whether the left node satisfies the segmentation prerequisite
        if bestSplit_dataset_l.shape[0] < min_sample or np.unique(bestSplit_dataset_l['target']).__len__() == 1:
            presplit_left.iv = bestSplit_ivl
            left = presplit_left
        else:
            left = binning_data_split(bestSplit_dataset_l,var,global_bt,global_gt,min_sample,alpha=0.01)

        return node(var_name=var,split_point=bestSplit_point,iv=ivi,left=left,right=right)
    else:
        # Returns the current node as the final leaf node
        return node(var_name=var,iv=ivi)


def search(tree,split_list):
    '''
    找到树的分裂节点，返回 split_point list
    '''
    if isinstance(tree.split_point, list):
        split_list.extend(tree.split_point)
    else:
        split_list.append(tree.split_point)

    if tree.left is not None:
        search(tree.left,split_list)

    if tree.right is not None:
        search(tree.right,split_list)

    return split_list


def format_iv_split(df,var,split_list,global_bt,global_gt):
    '''
    给定一个数据集dataframe, 和一个分裂点列表, 返回一个信息值
    :param df:
    :param var:
    :param split_list:
    :param global_bt:
    :param global_gt:
    :return:
    '''
    civ = InfoValue()
    civ.var_name = var
    civ.split_list = split_list
    dfcp = df[:]

    civ.sub_total_sample_num = []
    civ.positive_sample_num = []
    civ.negative_sample_num = []
    civ.sub_total_num_percentage = []
    civ.positive_rate_in_sub_total = []

    for i in range(0, split_list.__len__()):
        dfi = dfcp[dfcp[var] <= split_list[i]]
        dfcp = dfcp[dfcp[var] > split_list[i]]
        gd = calulate_iv(dfi, var,global_bt,global_gt)
        woei, ivi = gd['woei'],gd['ivi']
        civ.woe_list.append(woei)
        civ.iv_list.append(ivi)
        civ.sub_total_sample_num.append(dfi.shape[0])
        civ.positive_sample_num.append(gd['positive_sample_num'])
        civ.negative_sample_num.append(gd['negative_sample_num'])
        civ.sub_total_num_percentage.append(gd['sub_total_num_percentage'])
        civ.positive_rate_in_sub_total.append(gd['positive_rate_in_sub_total'])
        civ.negative_rate_in_sub_total.append(gd['negative_rate_in_sub_total'])

    if dfcp.shape[0]>0:
        gd = calulate_iv(dfcp, var,global_bt,global_gt)
        woei, ivi = gd['woei'],gd['ivi']
        civ.woe_list.append(woei)
        civ.iv_list.append(ivi)
        civ.sub_total_sample_num.append(dfcp.shape[0])
        civ.positive_sample_num.append(gd['positive_sample_num'])
        civ.negative_sample_num.append(gd['negative_sample_num'])
        civ.sub_total_num_percentage.append(gd['sub_total_num_percentage'])
        civ.positive_rate_in_sub_total.append(gd['positive_rate_in_sub_total'])
        civ.negative_rate_in_sub_total.append(gd['negative_rate_in_sub_total'])

    civ.iv = sum(civ.iv_list)
    return civ


def woe_trans(dvar,civ):
    """
        使用woe值替换原始值
    """
    var = copy.deepcopy(dvar)
    if not civ.is_discrete:
        if civ.woe_list.__len__()>1:
            split_list = []
            split_list.append(float("-inf"))
            split_list.extend([i for i in civ.split_list])
            split_list.append(float("inf"))

            for i in range(civ.woe_list.__len__()):
                var[(dvar > split_list[i]) & (dvar <= split_list[i+1])] = civ.woe_list[i]
        else:
            var[:] = civ.woe_list[0]
    else:
        split_map = {}
        for i in range(civ.split_list.__len__()):
            for j in range(civ.split_list[i].__len__()):
                split_map[civ.split_list[i][j]] = civ.woe_list[i]

        var = var.map(split_map)

    return var

def proc_woe_discrete(df,var,global_bt,global_gt,min_sample,alpha=0.01):
    '''
    处理离散变量的woe转换
    :param df:
    :param var:
    :param global_bt:
    :param global_gt:
    :param min_sample:
    :return:
    '''
    s = 'process discrete variable:'+str(var)
    print(s.center(60, '-'))

    df = df[[var,'target']]
    div = DisInfoValue()
    div.var_name = var
    rdict = {}
    cpvar = df[var]
    # print('np.unique(df[var])：',np.unique(df[var]))
    for var_value in np.unique(df[var]):
        df_temp = df[df[var] == var_value]
        gd = calulate_iv(df_temp,var,global_bt,global_gt)
        woei, ivi = gd['woei'],gd['ivi']
        div.origin_value.append(var_value)
        div.woe_before.append(woei)
        rdict[var_value] = woei
        # print(var_value,woei,ivi)

    cpvar = cpvar.map(rdict)
    df[var] = cpvar

    iv_tree = binning_data_split(df,var,global_bt,global_gt,min_sample,alpha)

    # 得到所有的分裂点
    split_list = []
    search(iv_tree, split_list)
    split_list = list(np.unique([1.0 * x for x in split_list if x is not None]))
    split_list.sort()

    # 检查分裂点是否符合要求
    split_list = check_point(df, var, split_list, min_sample)
    split_list.sort()

    civ = format_iv_split(df, var, split_list,global_bt,global_gt)
    civ.is_discrete = 1

    split_list_temp = []
    split_list_temp.append(float("-inf"))
    split_list_temp.extend([i for i in split_list])
    split_list_temp.append(float("inf"))

    a = []
    for i in range(split_list_temp.__len__() - 1):
        temp = []
        for j in range(div.origin_value.__len__()):
            if (div.woe_before[j]>split_list_temp[i]) & (div.woe_before[j]<=split_list_temp[i+1]):
                temp.append(div.origin_value[j])

        if temp != [] :
            a.append(temp)

    civ.split_list = a

    return civ


def proc_woe_continuous(df,var,global_bt,global_gt,min_sample,alpha=0.01):
    '''
    处理连续变量的woe转换
    :param df:
    :param var:
    :param global_bt:
    :param global_gt:
    :param min_sample:
    :return:
    '''
    s = 'process continuous variable:'+str(var)
    print(s.center(60, '-'))
    df = df[[var,'target']]
    iv_tree = binning_data_split(df, var,global_bt,global_gt,min_sample,alpha)

    # 遍历整课树，获取最佳分裂点
    split_list = []
    search(iv_tree, split_list)
    split_list = list(np.unique([1.0 * x for x in split_list if x is not None]))
    split_list.sort()

    # 检查和处理分裂点
    split_list = check_point(df, var, split_list, min_sample)
    split_list.sort()

    civ = format_iv_split(df, var,split_list,global_bt,global_gt)

    return civ

def fillna(dataset,bin_var_list,discrete_var_list,continuous_filler=-1,discrete_filler='missing'):
    """
     填充null, 连续值用 -1 , 离散用 `missing`
    """
    for var in [tmp for tmp in bin_var_list if tmp in list(dataset.columns)]:
        dataset.loc[dataset[var].isnull(), (var)] = continuous_filler

    for var in [tmp for tmp in discrete_var_list if tmp in list(dataset.columns)]:
        dataset.loc[dataset[var].isnull(), (var)] = discrete_filler


def process_train_woe(infile_path=None,outfile_path=None,rst_path=None,config_path=None):
    print('run into process_train_woe: ',time.asctime(time.localtime(time.time())))
    data_path = infile_path
    cfg = config.config()
    cfg.load_file(config_path,data_path)
    bin_var_list = [tmp for tmp in cfg.bin_var_list if tmp in list(cfg.dataset_train.columns)]

    for var in bin_var_list:
        cfg.dataset_train.loc[cfg.dataset_train[var].isnull(), (var)] = -1

    change_feature_dtype(cfg.dataset_train, cfg.variable_type)
    rst = []

    
    print('process woe transformation of continuous variables: ',time.asctime(time.localtime(time.time())))
    print('cfg.global_bt',cfg.global_bt)
    print('cfg.global_gt', cfg.global_gt)
    # 处理连续变量
    for var in bin_var_list:
        rst.append(proc_woe_continuous(cfg.dataset_train,var,cfg.global_bt,cfg.global_gt,cfg.min_sample,alpha=0.05))

    # 处理离散变量
    print('process woe transformation of discrete variables: ',time.asctime(time.localtime(time.time())))
    for var in [tmp for tmp in cfg.discrete_var_list if tmp in list(cfg.dataset_train.columns)]:
        # fill null
        cfg.dataset_train.loc[cfg.dataset_train[var].isnull(), (var)] = 'missing'
        rst.append(proc_woe_discrete(cfg.dataset_train,var,cfg.global_bt,cfg.global_gt,cfg.min_sample,alpha=0.05))

    feature_detail = eval.eval_feature_detail(rst, outfile_path)

    print('save woe transformation rule into pickle: ',time.asctime(time.localtime(time.time())))
    output = open(rst_path, 'wb')
    pickle.dump(rst,output)
    output.close()

    return feature_detail,rst


def process_woe_trans(in_data_path=None,rst_path=None,out_path=None,config_path=None):
    cfg = config.config()
    cfg.load_file(config_path, in_data_path)

    for var in [tmp for tmp in cfg.bin_var_list if tmp in list(cfg.dataset_train.columns)]:
        # fill null
        cfg.dataset_train.loc[cfg.dataset_train[var].isnull(), (var)] = -1

    for var in [tmp for tmp in cfg.discrete_var_list if tmp in list(cfg.dataset_train.columns)]:
        # fill null
        cfg.dataset_train.loc[cfg.dataset_train[var].isnull(), (var)] = 'missing'

    change_feature_dtype(cfg.dataset_train, cfg.variable_type)

    output = open(rst_path, 'rb')
    rst = pickle.load(output)
    output.close()

    for r in rst:
        cfg.dataset_train[r.var_name] = woe_trans(cfg.dataset_train[r.var_name], r)

    cfg.dataset_train.to_csv(out_path,index=False)
