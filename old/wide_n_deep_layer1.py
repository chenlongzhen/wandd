'''
Wide & deep model for training and evaluation.
Funcs:
    build_estimator: construct model structure and assign type/position of inputs.
    from_tfrecord_file: read HDFS tfrecord dataset directory.  from_csv_file: read HDFS csv dataset directory.
    train_and_eval: import input and run session.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import os
import tensorflow as tf
import yaml
from tensorflow.python import debug as tf_debug
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import nn
from tensorflow.contrib.layers.python.layers import optimizers
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.contrib.learn.python.learn.estimators import head as head_lib

import warnings
warnings.filterwarnings("ignore")

# set logging level
tf.logging.set_verbosity(tf.logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

flags = tf.app.flags
FLAGS = flags.FLAGS

# set parameters
# TODO: replace parameters as shell inputs.
flags.DEFINE_string("model_dir", "../../dis_with_wide/model_test", "Base directory for output models.")
flags.DEFINE_string("model_type", "wide", "Valid model types: {'wide', 'deep', 'wide_n_deep'}.")
flags.DEFINE_string("serving_model","../../dis_with_wide/serve_model3","serving model path") 
#flags.DEFINE_integer("train_steps", 50, "Number of training steps for each evaluation.")
flags.DEFINE_string("train_dir", "../../dis_with_wide/train", "Path to the training data.")
flags.DEFINE_string("test_dir", "../../dis_with_wide/test", "Path to the test data.")
#flags.DEFINE_string("train_dir", "/data/CidV_ImgV/training", "Path to the training data.")
#flags.DEFINE_string("test_dir", "/data/CidV_ImgV/testing", "Path to the test data.")
flags.DEFINE_boolean("is_test", True, "whether to be tested.")
flags.DEFINE_string("feature_yaml", "../../dis_with_wide/features.yaml", "Path to the feature config file.")
flags.DEFINE_boolean("is_from_hdfs", False, "whether reading HDFS data.")

# load config yaml
yaml_config = yaml.load(open(FLAGS.feature_yaml))
COLUMNS = yaml_config['using_features_dl']
FEATURE_CONF = yaml_config['features_conf']
CARTESIAN_CROSS = yaml_config['cartesian_cross']
LABEL_COLUMN = "label"
POSTERIOR_PREFIX = ["poPV", "poCLK", "poCTR", "poCOEC_st", "poCOEC_em"]
CARTESIAN_CROSS_DICT = {}

def _append_element(feat_name, feat_conf = None):
    global COLUMNS
    global FEATURE_CONF
    COLUMNS.append(feat_name)
    FEATURE_CONF[feat_name] = {
        "feature_type": 'sparse', 
        "model_type": 'deep' 
    } if feat_conf is None else feat_conf
    
def _replace_element(feat_name, new_name = None):
    global COLUMNS
    global FEATURE_CONF
    if new_name:
        COLUMNS.append(new_name)
        FEATURE_CONF[new_name] = FEATURE_CONF[feat_name]
    COLUMNS.remove(feat_name)
    FEATURE_CONF.pop(feat_name)

def append_posterior():
    posterior_days = [3]
    if yaml_config['if_posteriori_pv'] == 1:
        feats = [k for k, v in yaml_config['features_conf'].items() if v.get('post_pv', 0) == 1 and k in COLUMNS]
        for i in feats:
            for j in posterior_days:
                _append_element("poPV_" + i + "_" + str(j), yaml_config['features_conf']['poPV'])
    if yaml_config['if_posteriori_clk'] == 1:
        feats = [k for k, v in yaml_config['features_conf'].items() if v.get('post_clk', 0) == 1 and k in COLUMNS]
        for i in feats:
            for j in posterior_days:
                _append_element("poCLK_" + i + "_" + str(j), yaml_config['features_conf']['poCLK'])
    if yaml_config['if_posteriori_ctr'] == 1:
        feats = [k for k, v in yaml_config['features_conf'].items() if v.get('post_ctr', 0) == 1 and k in COLUMNS]
        for i in feats:
            for j in posterior_days:
                _append_element("poCTR_" + i + "_" + str(j), yaml_config['features_conf']['poCTR'])
    if yaml_config['if_posteriori_coec'] == 1:
        feats = [k for k, v in yaml_config['features_conf'].items() if v.get('post_coec', 0) == 1 and k in COLUMNS]
        for i in feats:
            for j in posterior_days:
                if yaml_config['bias_method'] == 'STAT':
                    _append_element("poCOEC_st_" + i + "_" + str(j), yaml_config['features_conf']['poCOEC_st'])
                elif yaml_config['bias_method'] == 'EM':
                    _append_element("poCOEC_em_" + i + "_" + str(j), yaml_config['features_conf']['poCOEC_em'])
                        
def append_rounding():
    tmp_d = [fea for fea in COLUMNS if fea in FEATURE_CONF and 'use_rounding' in FEATURE_CONF[fea] and FEATURE_CONF[fea]['use_rounding'] == 1]
    print (COLUMNS)
    print (FEATURE_CONF)
    print (tmp_d)
    for item in tmp_d:
        if not item.startswith('dl_'):
            _replace_element(item, 'dl_' + item)
        

def append_cartesian_cross():
    for item in CARTESIAN_CROSS:
        d = sorted(item.split("&"))
        if len(d) == 2 and d[0] in COLUMNS and d[1] in COLUMNS:
            _append_element(d[0] + "&" + d[1])
            
'''
def _append_cartesian_cross_element(feat_name):
    COLUMNS.append(feat_name)
    FEATURE_CONF[feat_name] = {"feature_type": "sparse", "model_type": "deep", "model_feed_type": "embedding", "bucket_size": 100000, "dimension": 64}

def append_campaign_id():
    campaign_list = []
    for line in open('campaign.txt'):
        try:
            campaign_id = int(line)
            if campaign_id not in campaign_list:
                campaign_list.append(campaign_id)
        except ValueError:
            pass
    CAMPAIGN_DICT = zip(campaign_list, range(len(campaign_list)))
'''

def _check_config(source, attr, legal_list = None, default = None):
    if source is None or attr not in source or (legal_list is not None and source[attr] not in legal_list):
        return default
    return source[attr]


# build feature columns to their suitable types, according to yaml config
def build_columns():
    wide_columns = []
    deep_columns = []
    psid_columns = []
    # TODO: use columns_dict to record cross_column origins.
    columns_dict = {}
    # wide 
    iid_clked_layer = None
    iid_imp_layer = None
    
    
    # column types:
    # tf.contrib.layers.sparse_column_with_keys(column_name, keys, default_value=-1, combiner=None)
    # tf.contrib.layers.sparse_column_with_hash_bucket(column_name, hash_bucket_size, combiner=None, dtype=tf.string)
    # tf.contrib.layers.real_valued_column(column_name, dimension=1, default_value=None, dtype=tf.float32, normalizer=None)
    # tf.contrib.layers.bucketized_column(source_column, boundaries)
    # tf.contrib.layers.crossed_column(columns, hash_bucket_size, combiner=None, ckpt_to_load_from=None, tensor_name_in_ckpt=None, hash_key=None)
    # tf.contrib.layers.embedding_column(sparse_id_column, dimension, combiner=None, initializer=None, ckpt_to_load_from=None, tensor_name_in_ckpt=None, max_norm=None)

    # input assignment
    for fea in COLUMNS:
        fea_config = _check_config(FEATURE_CONF, fea)
        if fea_config is None:
            print("[build_estimator] incorrect input %s: no feature_conf." % (str(fea)))
            continue
        feature_type = _check_config(fea_config, 'feature_type', legal_list = ['sparse','multi_sparse','real'])
        model_type = _check_config(fea_config, 'model_type', legal_list = ['wide','deep'])
        layer = None
        # assign input type.
        if feature_type == 'sparse' or feature_type == 'multi_sparse':
            if model_type == 'wide':
                feature_sparse = _check_config(fea_config, 'feature_sparse', legal_list = ['key','hash','integer'], default = 'hash')
                if feature_sparse == 'key':
                    sparse_key = _check_config(fea_config, 'sparse_key')
                    if sparse_key is None:
                        print("[build_estimator] incorrect input %s: no sparse_key." % (str(fea)))
                        continue
                    print("[build_estimator] add sparse_column_with_keys, fea = %s, keys = %s" % (str(fea), str(sparse_key)))
                    columns_dict[fea] = tf.feature_column.categorical_column_with_vocabulary_list(key=fea, vocabulary_list=sparse_key)
                elif feature_sparse == 'hash':
                    bucket_size = _check_config(fea_config, 'bucket_size', default = 1024)
                    print("[build_estimator] add sparse_column_with_hash_bucket, fea = %s, hash_bucket_size = %d" % (str(fea), bucket_size))
                    columns_dict[fea] = tf.feature_column.categorical_column_with_hash_bucket(key=fea, hash_bucket_size=bucket_size)
                elif feature_sparse == 'integer':
                    bucket_size = _check_config(fea_config, 'bucket_size')
                    if bucket_size is None:
                        print("[build_estimator] incorrect input %s: no bucket_size." % (str(fea)))
                        continue
                    print("[build_estimator] add sparse_column_with_integerized_feature, fea = %s, bucket_size = %d" % (str(fea), bucket_size))
                    columns_dict[fea] = tf.feature_column.categorical_column_with_identity(key=fea, num_buckets=bucket_size)
                else:
                    print("[build_estimator] incorrect input %s: illegal feature_sparse." % (str(fea)))
                layer = columns_dict[fea]
            elif model_type == 'deep':
                model_feed_type = _check_config(fea_config, 'model_feed_type', legal_list = ['embedding','onehot'], default = 'onehot')
                bucket_size = _check_config(fea_config, 'bucket_size', default = 1024)
                columns_dict[fea] = tf.feature_column.categorical_column_with_hash_bucket(key=fea, hash_bucket_size=bucket_size)
                if model_feed_type == 'embedding':
                    dimension = _check_config(fea_config, 'dimension', default = 32)
                    print("[build_estimator] add embedding_column, fea = %s, hash_bucket_size = %d, dimension = %d" % (str(fea), bucket_size, dimension))
                    layer = tf.feature_column.embedding_column(categorical_column = columns_dict[fea], dimension=dimension, combiner="mean")
                elif model_feed_type == 'onehot':
                    print("[build_estimator] add one_hot_column, fea = %s, hash_bucket_size = %d" % (str(fea), bucket_size))
                    layer = tf.feature_column.indicator_column(columns_dict[fea])
                else:
                    print("[build_estimator] incorrect input %s: illegal model_feed_type." % (str(fea)))
            else:
                print("[build_estimator] incorrect input %s: illegal model_type" % (str(fea)))
        elif feature_type == 'real':
            if model_type == 'wide':
                boundaries = _check_config(fea_config, 'boundaries')
                if boundaries is None:
                    print("[build_estimator] incorrect input %s: no boundaries." % (str(fea)))
                    continue
                print("[build_estimator] add bucketized_column, fea = %s, boundaries = %s" % (str(fea), str(boundaries)))
                columns_dict[fea] = tf.feature_column.bucketized_column(source_column = fea, boundaries=boundaries)
            elif model_type == 'deep':
                fea_dim = _check_config(fea_config, 'feature_dimension', default = 1)
                print("[build_estimator] add real_valued_column, fea = %s, dimension = %d" % (str(fea), fea_dim))
                columns_dict[fea] = tf.feature_column.numeric_column(key = fea, shape=(fea_dim,))
            else:
                print("[build_estimator] incorrect input %s: illegal model_type." % (str(fea)))
            layer = columns_dict[fea]        
        else:
            print("[build_estimator] incorrect input %s: illegal feature_type." % (str(fea)))
        # add layer to columns
        if layer is not None:
            if model_type == 'wide':
                wide_columns.append(layer)
            elif model_type == 'deep':
                if fea == 'psid_abs':
                    psid_columns.append(layer)
                else:
                    deep_columns.append(layer)               
            # clk item id and pv item id.
            if fea == "iid_clked":
                iid_clked_layer = layer
            if fea == "iid_imp":
                iid_imp_layer = layer

    '''
    for item in CARTESIAN_CROSS:
        d = sorted(item.split("&"))
        if len(d) == 2 and d[0] in COLUMNS and d[1] in COLUMNS:
            print("[build_estimator] add crossed_column, fea=" + str(d[0]) + ", " + str(d[1]) + ", hash_bucket_size=10000")
            layer = tf.feature_column.crossed_column([columns_dict[d[0]], columns_dict[d[1]]], hash_bucket_size = 10000)
            wide_columns.append(layer)
    '''

    #layer_wc = tf.contrib.layers.sparse_column_with_integerized_feature('wide_clk', bucket_size = 2, combiner = "sqrtn")
    #wide_columns.append(layer_wc)
    #layer_wp = tf.contrib.layers.sparse_column_with_integerized_feature('wide_pv', bucket_size = 2, combiner = "sqrtn")
    #wide_columns.append(layer_wp)

    if iid_clked_layer is not None and iid_imp_layer is not None:
        bucket_size = 50000
        print("[build_estimator] add crossed_column for wide crossing, bucket_size = %d" % (bucket_size))
        layer_wa = tf.feature_column.crossed_column([iid_clked_layer, iid_imp_layer], bucket_size)
        wide_columns.append(layer_wa)

    # print some info for the columns registered to different types
    print("[build_estimator] wide columns: %d" % (len(wide_columns)))
    print(wide_columns)
    print("[build estimator] deep columns: %d" % (len(deep_columns)))
    print(deep_columns)
    print("[build estimator] psid columns: %d" % (len(psid_columns)))
    print(psid_columns)
    print("[build estimator] COLUMNS: %d" % (len(COLUMNS)))
    print(columns_dict)

    return wide_columns, deep_columns, psid_columns

# the function discribing what the model looks like
def build_model_fn(features, labels, mode, params):
    wide_columns = params.get("wide_columns")
    deep_columns = params.get("deep_columns")
    psid_columns = params.get("psid_columns")
    print("[build_model_fn] features: %d" % (len(features)))
    print(features)

    # logic to do the following:
    # 1. Configure the model via TensorFlow operations
    # partitioner = partitioned_variables.min_max_variable_partitioner(
    #     max_partitions = 0)
    input_layer_partitioner = (
        partitioned_variables.min_max_variable_partitioner(
            max_partitions = 0,
            min_slice_size = 64<<20))

    deep_input = tf.feature_column.input_layer(
        features = features, 
        feature_columns = deep_columns + psid_columns, 
        weight_collections = ['deep_embedding'])
    
#    with tf.variable_scope('dnn_input_psid', values = features.values(), partitioner = input_layer_partitioner) as dnn_input_psid_scope:
#        deep_input_psid = tf.feature_column.input_layer(
#            features = features, 
#            feature_columns = psid_columns, 
#            weight_collections = ['deep'])
#    
#    deep_input = tf.concat([deep_input, deep_input_psid], 1)

    with tf.variable_scope('deep_model', values = (deep_input,)) as dnn_hidden_scope:
        with tf.variable_scope('dnn_hidden_1', values = (deep_input,)) as dnn_hidden_scope:
            deep_hidden_1 = tf.layers.dense(
                deep_input, 1024, activation = nn.relu)
            deep_hidden_1 = tf.layers.dropout(
                deep_hidden_1, rate = 0.9, 
                training = mode == tf.estimator.ModeKeys.TRAIN)
            if mode == tf.estimator.ModeKeys.TRAIN:
                tf.summary.scalar("%s/fraction_of_zero_values" % dnn_hidden_scope.name, nn.zero_fraction(deep_hidden_1))
                tf.summary.histogram("%s/activation" % dnn_hidden_scope.name, deep_hidden_1)

        with tf.variable_scope('dnn_hidden_2', values = (deep_hidden_1,)) as dnn_hidden_scope:
            #deep_hidden_2 = tf.contrib.layers.fully_connected(
            #    deep_hidden_1, 512, activation_fn = nn.relu, variables_collections = ['deep'])
            deep_hidden_2 = tf.layers.dense(
                deep_hidden_1, 512, activation = nn.relu)
            deep_hidden_2 = tf.layers.dropout(
                deep_hidden_2, rate = 0.9, 
                training = mode == tf.estimator.ModeKeys.TRAIN)
            if mode == tf.estimator.ModeKeys.TRAIN:
                tf.summary.scalar("%s/fraction_of_zero_values" % dnn_hidden_scope.name, nn.zero_fraction(deep_hidden_2))
                tf.summary.histogram("%s/activation" % dnn_hidden_scope.name, deep_hidden_2)
        # deep_hidden_2_merge = tf.concat([deep_hidden_2, deep_input_psid], 1)
        
        with tf.variable_scope('dnn_hidden_3', values = (deep_hidden_2,)) as dnn_hidden_scope:
            deep_hidden_3 = tf.layers.dense(
                deep_hidden_2, 256, activation = nn.relu)
            deep_hidden_3 = tf.layers.dropout(
                deep_hidden_3, rate = 0.9, 
                training = mode == tf.estimator.ModeKeys.TRAIN)
            if mode == tf.estimator.ModeKeys.TRAIN:
                tf.summary.scalar("%s/fraction_of_zero_values" % dnn_hidden_scope.name, nn.zero_fraction(deep_hidden_3))
                tf.summary.histogram("%s/activation" % dnn_hidden_scope.name, deep_hidden_3)
        deep_hidden_3_con = tf.concat([deep_hidden_3,deep_input],1)
        with tf.variable_scope('dnn_logits', values = (deep_hidden_3_con,)) as dnn_logits_scope:
            deep_logits = tf.layers.dense(
                deep_hidden_3_con, 1, activation = None, bias_initializer = None)
            tf.summary.scalar("%s/fraction_of_zero_values" % dnn_logits_scope.name, nn.zero_fraction(deep_logits))
            tf.summary.histogram("%s/activation" % dnn_logits_scope.name, deep_logits)

    # wide
#    wide_input = tf.feature_column.input_layer(
#        features = features, 
#        feature_columns = wide_columns, 
#        weight_collections = ['wide'])
#
#    with tf.variable_scope('wide_dense', values = (deep_hidden_3,)) as dnn_logits_scope:
#        deep_logits = tf.layers.dense(
#            wide_input, 1, activation = None, bias_initializer = None)

    with tf.variable_scope('linear_model', values = features.values()) as linear_scope:
        wide_logits = tf.feature_column.linear_model(features = features, feature_columns = wide_columns, units = 1, weight_collections = ['wide'])
        tf.summary.scalar("%s/fraction_of_zero_values" % linear_scope.name, nn.zero_fraction(wide_logits))
        tf.summary.histogram("%s/activation" % linear_scope.name,wide_logits)

    # logit setting
    if FLAGS.model_type == "deep":
        print("[build_model_fn] logits use deep.")
        logits = tf.reshape(deep_logits,[-1])

    elif FLAGS.model_type == "wide":
        print("[build_model_fn] logits use wide.")
        logits = tf.reshape(wide_logits,[-1]) 

    else:
        print("[build_model_fn] logits use wide and deep.")
        logits = tf.reshape(deep_logits,[-1]) + tf.reshape(wide_logits,[-1])


    predictions = tf.sigmoid(logits,name='output')

    # 2. Define the loss function for training/evaluation
    training_loss = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        # training_loss = tf.losses.log_loss(labels, predictions)
        training_loss = tf.losses.sigmoid_cross_entropy(labels, logits)

    # 3. Define the training operation/optimizer
    train_op = None

    if mode == tf.estimator.ModeKeys.TRAIN:

       if "deep" in FLAGS.model_type:
           deep_opt = tf.train.AdagradOptimizer(
               learning_rate = 0.5, name = "Adagrad"
                       )
       
       #deep_opt = tf.train.AdamOptimizer(
       #        learning_rate = 0.001, name = "Adam"
       #                )

           train_op_deep = deep_opt.minimize(
               loss = training_loss,
	       global_step=tf.train.get_global_step(),
               var_list = tf.get_collection('deep_embedding') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"deep_model*")
               )

       if "wide" in FLAGS.model_type:
           wide_opt = tf.train.FtrlOptimizer(
                   learning_rate = min(0.01, 1.0 / len(wide_columns)),
                   learning_rate_power=-0.5,
                   initial_accumulator_value=0.1,
                   l1_regularization_strength=0.0,
                   l2_regularization_strength=0.0,
                   name='Ftrl',
                   accum_name=None,
                   linear_name=None,
                   l2_shrinkage_regularization_strength=0.0 
               )
           train_op_wide = wide_opt.minimize(
		    training_loss,
		    global_step=tf.train.get_global_step(),
                    var_list = tf.get_collection('wide'),
		    aggregation_method=None,
		    colocate_gradients_with_ops=False,
		    name=None,
		    grad_loss=None 
		)

       if FLAGS.model_type == "deep":
           print("[build_model_fn] logits use deep train op.")
           train_op = train_op_deep
       elif FLAGS.model_type == "wide":
           print("[build_model_fn] logits use wide train op.")
           train_op = train_op_wide 
       else:
           print("[build_model_fn] logits use deep and wide op.")
           train_ops = [train_op_deep, train_op_wide]
           train_op = control_flow_ops.group(*train_ops)
#          train_op = train_op_wide
           with ops.control_dependencies([train_op]):
               with ops.colocate_with(tf.train.get_global_step()):
                   train_op = state_ops.assign_add(tf.train.get_global_step(),1).op

    # 4. Generate predictions
    predictions_dict = {
            "prediction": predictions}

    eval_metric_ops = None
    if mode != tf.estimator.ModeKeys.PREDICT:
        eval_metric_ops = {
                "AUC": tf.metrics.auc(labels, predictions, name = "auc"),
                "mean_labels": tf.metrics.mean(labels, name = "mean_labels"),
                "mean_preds": tf.metrics.mean(predictions, name = "mean_preds")}

    # print 
    #print("tf.get_collection('deep')")
    #print(tf.get_collection('deep_embedding'))
    #print("tf.get_collection('wide')")
    #print(tf.get_collection('wide'))
    #print("tf.GraphKeys.GLOBAL_VARIABLES")
    #print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    #print("ops.GraphKeys.MODEL_VARIABLES")
    #print(tf.get_collection(ops.GraphKeys.MODEL_VARIABLES))
    #print("ops.GraphKeys.TRAINABLE_VARIABLES")
    #print(tf.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES))

    #print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"deep_model*"))
    #exit(1)

    # 5. Return predictions/loss/train_op/eval_metric_ops in ModelFnOps object
    return tf.estimator.EstimatorSpec(
            mode = mode,
            predictions = predictions_dict,
            loss = training_loss,
            train_op = train_op,
            eval_metric_ops = eval_metric_ops)
    '''
    head = head_lib._multi_class_head(n_classes = 2)
    return head.head_ops(features, labels, mode, _make_training_op, logits = logits)
    
    head = head_lib.multi_class_head(n_classes = 2)
    return head.create_model_fn_ops(
            mode = mode,
            features = features,
            labels = labels,
            train_op_fn = _make_training_op,
            logits_input = logits)
    '''

# call this function to return an estimator which can 'fit', 'evaluate' or 'predict_proba' data.
def build_estimator(model_dir):
    wide_columns, deep_columns, psid_columns = build_columns()
    m = tf.estimator.Estimator(
            model_fn = build_model_fn,
            model_dir = model_dir,
            config = tf.estimator.RunConfig().replace(
                save_checkpoints_steps = 200,
                save_summary_steps =25
            ),
            params = {
                "wide_columns": tuple(wide_columns or []),
                "deep_columns": tuple(deep_columns or []), 
                "psid_columns": tuple(psid_columns or [])})
    return m, wide_columns, deep_columns, psid_columns

#def _process_list_column(list_column, vocab_size):
#    sparse_strings = tf.string_split(list_column, delimiter='##')
#    sparse_ints = tf.SparseTensor(
#            indices = sparse_strings.indices,
#            values = tf.string_to_hash_bucket_fast(sparse_strings.values,vocab_size),
#            dense_shape = sparse_strings.dense_shape)
#    return tf.cast(tf.sparse_to_indicator(sparse_ints, vocab_size = vocab_size), tf.int64)

#def _process_list_column(list_column, vocab_size):
#    sparse_strings = tf.string_split(list_column, delimiter='##')
#    #print sess.run(sparse_strings.values)
#    values = tf.string_to_hash_bucket_fast(sparse_strings.values,vocab_size)       
#    return values

def _process_list_column(list_column, vocab_size):
    sparse_strings = tf.string_split(list_column, delimiter='##')
    sparse_ints = tf.SparseTensor(
            indices = sparse_strings.indices,
            values = tf.string_to_hash_bucket_fast(sparse_strings.values, vocab_size),
            dense_shape = sparse_strings.dense_shape)
    return sparse_ints

def read_csv_file(example):
    #reader = tf.TextLineReader()
    #_, example = reader.read(filename_queue)
    # label
    record_defaults = [tf.constant([], dtype = tf.int64)]

    # the dtype of each column is determined by the corresponding element of the record_defaults argument.
    for fea in sorted(COLUMNS):
        feature_type = FEATURE_CONF[fea]['feature_type']
        # wide side setting String !
        if FEATURE_CONF[fea]['model_type'] == 'wide':
            record_defaults.append(tf.constant(['missing'], dtype = tf.string))
            #print ('[read_csv_file] append record = %s, type = %s' % (fea, 'tf.string'))
            continue      
        if feature_type == 'sparse' or feature_type == 'multi_sparse':
            if 'feature_sparse' in FEATURE_CONF[fea] and FEATURE_CONF[fea]['feature_sparse'] == 'integer':
                #print ('[read_csv_file] append record = %s, type = %s' % (fea, 'tf.int64'))
                record_defaults.append(tf.constant([0], dtype = tf.int64))
            else:
                #print ('[read_csv_file] append record = %s, type = %s' % (fea, 'tf.string'))
                record_defaults.append(tf.constant(["missing"], dtype = tf.string))
        elif feature_type == 'real':
            fea_dim = _check_config(FEATURE_CONF[fea], 'feature_dimension', default = 1)
            #print ('[read_csv_file] append record = %s, type = %s, dim = %d' % (fea, 'tf.float32', fea_dim))
            for i in range(fea_dim):
                record_defaults.append(tf.constant([0], dtype = tf.float32))

    # wide_clk
    # record_defaults.append(tf.constant([''], dtype = tf.string))
    # wide_pv
    # record_defaults.append(tf.constant([''], dtype = tf.string))
    # line EOF = '\x03'
    # record_defaults.append(tf.constant([], dtype = tf.string))
    # print ('record defaults')
    # print (record_defaults)
    features = tf.decode_csv(example, record_defaults = record_defaults,name = "decode_csv")
    print ("features")
    print (features)
    #return features_cvt, features_batch[0]
    # clz
    # map features with the same order in to_csv_mapper.py.
    features_cvt = {}
    sorted_columns = sorted(COLUMNS)
    i = 1
    for c in sorted_columns:
        if FEATURE_CONF[c]['model_type'] == 'wide':
            features_cvt[c] = _process_list_column(features[i], 20000)
            i += 1
            # features_cvt[c].set_shape([batch_size, ])
        else:
            fea_dim = _check_config(FEATURE_CONF[c], 'feature_dimension', default = 1)
            if fea_dim > 1:
                features_cvt[c] = tf.stack(features[i:i+fea_dim], axis = 1)
            else:
                features_cvt[c] = features[i]
            i += fea_dim

    print ('features_cvt')
    print (features_cvt)
    print ('features[0]')
    print (features[0])
    return features_cvt,features[0]
 
def from_csv_queue(filename_queue, batch_size, read_threads, shuffle = True, allow_smaller_final_batch = False):
    # shuffle_batch from multiple files.
    if shuffle:
        features = [read_csv_file(filename_queue) for _ in range(read_threads)] 
        features_batch = tf.train.shuffle_batch_join(features, batch_size = batch_size, min_after_dequeue = batch_size * 3, capacity = batch_size * 9, allow_smaller_final_batch = allow_smaller_final_batch)
    else:
        features = read_csv_file(filename_queue)
        features_batch = tf.train.batch(features, batch_size = batch_size, capacity = batch_size * 5, allow_smaller_final_batch = allow_smaller_final_batch)
        
    # map features with the same order in to_csv_mapper.py.
    features_cvt = {}
    sorted_columns = sorted(COLUMNS)
    i = 1
    for c in sorted_columns:
        if FEATURE_CONF[c]['model_type'] == 'wide':
            features_cvt[c] = _process_list_column(features_batch[i], 20000)
            i += 1
            # features_cvt[c].set_shape([batch_size, ])
        else:
            fea_dim = _check_config(FEATURE_CONF[c], 'feature_dimension', default = 1)
            if fea_dim > 1:
                features_cvt[c] = tf.stack(features_batch[i:i+fea_dim], axis = 1)
            else:
                features_cvt[c] = features_batch[i]
            i += fea_dim

    # print ('features_cvt')
    # print (features_cvt)
    # print ('features_batch[0]')
    # print (features_batch[0])
    return features_cvt, features_batch[0]

def from_csv_file(ifile, batch_size, read_threads, shuffle = True, allow_smaller_final_batch = False):
    filename_queue = tf.train.string_input_producer(ifile, num_epochs = 1, shuffle = shuffle)
    return from_csv_queue(filename_queue, batch_size, read_threads, shuffle, allow_smaller_final_batch)

# main 
def train_and_eval():
    # load training/testing data from HDFS
    if FLAGS.is_from_hdfs:
        ''' HDFS directory parsing '''
        train_dir = FLAGS.train_dir + ("/" if FLAGS.train_dir[-1] != "/" else "")
        rst = os.system('hadoop fs -test -d ' + train_dir)
        if (rst != 0):
            print ("[main] train_dir not exists: " + train_dir)
            return
        cmd = os.popen('hadoop fs -ls ' + train_dir)
        train_files = [i.split()[-1] for i in cmd.read().strip().split('\n')[2:]]
        is_test = FLAGS.is_test
        if is_test:
            test_dir = FLAGS.test_dir + ("/" if FLAGS.test_dir[-1] != "/" else "")
            rst = os.system('hadoop fs -test -d ' + test_dir)
            if (rst != 0):
                print ("[main] test_dir not exists: " + test_dir)
                is_test = False
            else:
                cmd = os.popen('hadoop fs -ls ' + test_dir)
                test_files = [i.split()[-1] for i in cmd.read().strip().split('\n')[2:]]
    # load training/testing data from local directory
    else:
        train_dir = FLAGS.train_dir + ("/" if FLAGS.train_dir[-1] != "/" else "")
        rst = os.system('ls ' + train_dir)
        if (rst != 0):
            print ("[main] train_dir not exists: " + train_dir)
            return
        cmd = os.popen('ls ' + train_dir)
        train_files = [train_dir + i.split()[-1] for i in cmd.read().strip().split('\n')]
        is_test = FLAGS.is_test
        if is_test:
            test_dir = FLAGS.test_dir + ("/" if FLAGS.test_dir[-1] != "/" else "")
            rst = os.system('ls ' + test_dir)
            if (rst != 0):
                print ("[main] test_dir not exists: " + test_dir)
                is_test = False
            else:
                cmd = os.popen('ls ' + test_dir)
                test_files = [test_dir + i.split()[-1] for i in cmd.read().strip().split('\n')]
    # tell Tensorflow where to save/restore model files
    model_dir = tempfile.mkdtemp() if not FLAGS.model_dir else FLAGS.model_dir
    print("model directory = %s" % model_dir)

    # place holder 
    #inputPH = tf.placeholder(tf.string,shape=(1,1))     

    #inputs = "1,Chrome,,1-905234,,,1,unknown,0,Android,,13,,11,0,8833b578-130c-4aad-bdfb-5540ab4489f8-1502682390492,,0"
    

    # main graph
    with tf.Graph().as_default():
        value = ["1,Chrome,,1-905234,,,1,unknown,0,Android,,13,,11,0,8833b578-130c-4aad-bdfb-5540ab4489f8-150268239049    2,,0","1,Chrome,,1-905234,,,1,unknown,0,Android,,13,,11,0,8833b578-130c-4aad-bdfb-5540ab4489f8-150268239049        2,,0"]
        #inputs = tf.placeholder(tf.string, shape=(None,))
        #readCsv = read_csv_file(inputs)

        # build estimator
        m, wide_columns, deep_columns, psid_columns = build_estimator(model_dir)
        # set Supervisor for continious training
        sv = tf.train.Supervisor(logdir = model_dir + "/supervisor", save_summaries_secs = 3600)

        # GPU allocation  
        config = tf.ConfigProto()  
        config.gpu_options.allow_growth = False

        with sv.managed_session(config = config) as sess:
            #print("sess.run")
            #print(sess.run(readCsv, feed_dict= {inputs: value}))

            # set input_fn from input data files
            input_fn_fit = lambda: read_csv_file(value)

            input_fn_evaluate = lambda: read_csv_file(value)
            for step in range(100):
                print ("current step: %d" % (step))
                if sv.should_stop():
                    print ("should_stop: %d" % (step + 1))
                    break
                m.train(input_fn = input_fn_fit, steps = 1)

                results = m.evaluate(input_fn = input_fn_fit, steps = 1)
                for key in sorted(results):
                    print("train_%d | %s: %s" % ((step + 1), key, results[key]))

              #  # testing
              #  if is_test:
              #      results = m.evaluate(input_fn = input_fn_evaluate, steps = 10) 
              #      for key in sorted(results):
              #          print("test_%d | %s: %s" % (step + 50, key, results[key]))
        
#    # for serving
#    feature_columns = wide_columns + deep_columns + psid_columns
#    print(feature_columns)
#
#    from tensorflow.contrib.layers import create_feature_spec_for_parsing
#    feature_spec = create_feature_spec_for_parsing(feature_columns)
#
#    from tensorflow.contrib.learn.python.learn.utils import input_fn_utils
#    serving_input_fn = input_fn_utils.build_parsing_serving_input_fn(feature_spec)
#    servable_model_dir = FLAGS.serving_model
#    servable_model_path = m.export_savedmodel(servable_model_dir, serving_input_fn)

        

def main(_):
    append_posterior()
#    append_cartesian_cross()
#    append_campaign_id()
    append_rounding()
    train_and_eval()

if __name__ == "__main__":
    tf.app.run()

