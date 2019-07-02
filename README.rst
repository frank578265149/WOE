woe
===

.. image:: https://travis-ci.org/justdoit0823/pywxclient.svg?branch=master
    :target: https://travis-ci.org/justdoit0823/pywxclient

version: 0.1.4

Tools for WoE Transformation mostly used in ScoreCard Model for credit rating

Installation
--------------------------------

We can simply use pip to install, as the following:

.. code-block:: bash

   $ pip install woe

or installing from git

.. code-block:: bash

   $ pip install git+https://github.com/boredbird/woe


Features
========

  * Split tree with IV criterion

  * Rich and plentiful model eval methods

  * Unified format and easy for output

  * Storage of IV tree for follow-up use

**woe** module function tree
============================

:: 

	|- __init__
	|- config.py 
	|   |-- config
	|   	|-- __init__
	|		|-- change_config_var_dtype()
	|		|-- load_file()
	|- eval.py 
	|   |-- compute_ks()
	|   |-- eval_data_summary()
	|   |-- eval_feature_detail()
	|   |-- eval_feature_stability()
	|   |-- eval_feature_summary()
	|   |-- eval_model_stability()
	|   |-- eval_model_summary()
	|   |-- eval_segment_metrics()
	|   |-- plot_ks()
	|   |-- proc_cor_eval()
	|   |-- proc_validation()
	|   |-- wald_test()
	|- feature_process.py 
	|   |-- binning_data_split()
	|   |-- calculate_iv_split()
	|   |-- calulate_iv()
	|   |-- change_feature_dtype()
	|   |-- check_point()
	|   |-- fillna()
	|   |-- format_iv_split()
	|   |-- proc_woe_continuous()
	|   |-- proc_woe_discrete()
	|   |-- process_train_woe()
	|   |-- process_woe_trans()
	|   |-- search()
	|   |-- woe_trans()
	|- ftrl.py 
	|   |-- FTRL()
	|   |-- LR()
	|- GridSearch.py 
	|   |-- fit_single_lr()
	|   |-- grid_search_lr_c()
	|   |-- grid_search_lr_c_main()
	|   |-- grid_search_lr_validation()


Examples
========

In the examples directory, there is a simple woe transformation program as tutorials.

Or you can write a more complex program with this `woe` package.
on branch feature
