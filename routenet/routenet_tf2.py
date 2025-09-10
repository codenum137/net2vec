# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import argparse
import os
from tqdm import tqdm
import datetime

# ==============================================================================
# KAN (Kolmogorov-Arnold Networks) Implementation
# ==============================================================================

class KANLayer(tf.keras.layers.Layer):
    """
    KAN (Kolmogorov-Arnold Networks) Layer implementation
    简化版本，使用可学习的样条函数替代传统激活函数
    """
    
    def __init__(self, units, grid_size=5, spline_order=3, **kwargs):
        super(KANLayer, self).__init__(**kwargs)
        self.units = units
        self.grid_size = grid_size
        self.spline_order = spline_order
        
    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        
        # 基础权重矩阵（线性部分）
        self.base_weight = self.add_weight(
            name='base_weight',
            shape=(input_dim, self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # 偏置项
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        
        # 样条函数的权重参数
        # 为每个输入-输出连接学习多项式系数
        self.spline_weights = self.add_weight(
            name='spline_weights',
            shape=(input_dim, self.units, 4),  # 3次多项式，4个系数
            initializer='glorot_uniform',
            trainable=True
        )
        
        # 门控参数：控制线性部分和非线性部分的权重
        self.gate_weights = self.add_weight(
            name='gate_weights',
            shape=(input_dim, self.units),
            initializer='ones',
            trainable=True
        )
        
        super(KANLayer, self).build(input_shape)
    
    def call(self, inputs, training=None):
        # 基础线性变换
        linear_output = tf.matmul(inputs, self.base_weight) + self.bias  # [batch_size, units]
        
        # 非线性样条变换
        # 将输入标准化到[-1, 1]范围
        x_normalized = tf.tanh(inputs)  # [batch_size, input_dim]
        
        # 计算多项式基函数：1, x, x^2, x^3
        x_powers = tf.stack([
            tf.ones_like(x_normalized),
            x_normalized,
            tf.square(x_normalized),
            tf.pow(x_normalized, 3)
        ], axis=-1)  # [batch_size, input_dim, 4]
        
        # 样条输出计算：[batch_size, input_dim, 4] @ [input_dim, units, 4] -> [batch_size, input_dim, units]
        # 使用 einsum 进行高效的张量乘法
        spline_contributions = tf.einsum('bid,ijd->bij', x_powers, self.spline_weights)  # [batch_size, input_dim, units]
        
        # 应用门控权重并求和
        gated_splines = spline_contributions * tf.expand_dims(self.gate_weights, 0)  # [batch_size, input_dim, units]
        spline_output = tf.reduce_sum(gated_splines, axis=1)  # [batch_size, units]
        
        # 组合线性和非线性部分
        output = linear_output + spline_output
        
        # 应用激活函数
        output = tf.nn.selu(output)
        
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)
    
    def get_config(self):
        config = super(KANLayer, self).get_config()
        config.update({
            'units': self.units,
            'grid_size': self.grid_size,
            'spline_order': self.spline_order,
        })
        return config

# ==============================================================================
# 1. 数据加载与预处理 (无变化)
# ==============================================================================

def scale_fn(k, val):
    if k == 'traffic':
        return (val - 0.18) / 0.15
    if k == 'capacities':
        return val / 10.0
    return val

def parse_fn(serialized):
    feature_spec = {
        'traffic': tf.io.VarLenFeature(tf.float32),
        'delay': tf.io.VarLenFeature(tf.float32),
        'jitter': tf.io.VarLenFeature(tf.float32),
        'drops': tf.io.VarLenFeature(tf.float32),
        'packets': tf.io.VarLenFeature(tf.float32),
        'capacities': tf.io.VarLenFeature(tf.float32),
        'links': tf.io.VarLenFeature(tf.int64),
        'paths': tf.io.VarLenFeature(tf.int64),
        'sequences': tf.io.VarLenFeature(tf.int64),
        'n_links': tf.io.FixedLenFeature([], tf.int64),
        'n_paths': tf.io.FixedLenFeature([], tf.int64),
    }
    
    features = tf.io.parse_single_example(serialized, features=feature_spec)
    
    # 转换稀疏张量并标准化
    for k, v in features.items():
        if isinstance(v, tf.SparseTensor):
            v = tf.sparse.to_dense(v)
        if k in ['traffic', 'capacities']:
            v = scale_fn(k, v)
        features[k] = v
            
    labels = {
        'delay': features.pop('delay'),
        'jitter': features.pop('jitter'),
        'drops': features.pop('drops'),
        'packets': features['packets']
    }
    return features, labels

def transformation_func(features_batch, labels_batch):
    """正确的图合并批处理函数，修复形状不匹配问题"""
    
    # 获取批次大小
    batch_size = tf.shape(features_batch['n_links'])[0]
    
    # 计算累积偏移量
    n_links_cumsum = tf.cumsum(features_batch['n_links'])
    n_paths_cumsum = tf.cumsum(features_batch['n_paths'])
    
    link_offsets = tf.concat([[0], n_links_cumsum[:-1]], axis=0)
    path_offsets = tf.concat([[0], n_paths_cumsum[:-1]], axis=0)
    
    # 使用 tf.map_fn 来正确处理每个样本的偏移量
    def apply_link_offset(args):
        i, links = args
        return links + link_offsets[i]
    
    def apply_path_offset(args):
        i, paths = args
        return paths + path_offsets[i]
    
    # 为每个样本应用对应的偏移量
    batch_indices = tf.range(batch_size)
    
    adjusted_links = tf.map_fn(
        apply_link_offset,
        (batch_indices, features_batch['links']),
        fn_output_signature=tf.RaggedTensorSpec(shape=[None], dtype=tf.int64),
        parallel_iterations=10
    )
    
    adjusted_paths = tf.map_fn(
        apply_path_offset,
        (batch_indices, features_batch['paths']),
        fn_output_signature=tf.RaggedTensorSpec(shape=[None], dtype=tf.int64),
        parallel_iterations=10
    )
    
    # 合并所有特征 - 使用 flat_values 获取展平的数据
    merged_features = {
        'traffic': features_batch['traffic'].flat_values,
        'capacities': features_batch['capacities'].flat_values,
        'packets': features_batch['packets'].flat_values,
        'links': adjusted_links.flat_values,
        'paths': adjusted_paths.flat_values,
        'sequences': features_batch['sequences'].flat_values,
        'n_links': tf.reduce_sum(features_batch['n_links']),
        'n_paths': tf.reduce_sum(features_batch['n_paths']),
    }
    
    merged_labels = {
        'delay': labels_batch['delay'].flat_values,
        'jitter': labels_batch['jitter'].flat_values,
        'drops': labels_batch['drops'].flat_values,
        'packets': labels_batch['packets'].flat_values,
    }
    
    return merged_features, merged_labels

def create_dataset(filenames, batch_size, is_training=True):
    ds = tf.data.TFRecordDataset(filenames)
    if is_training:
        ds = ds.shuffle(1000)
    
    ds = ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    # 使用 ragged_batch 来处理可变长度的张量
    ds = ds.ragged_batch(batch_size)
    
    # 然后应用图合并函数
    ds = ds.map(transformation_func, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# ==============================================================================
# 2. RouteNet 模型 (支持不同的输出配置)
# ==============================================================================

class RouteNet(tf.keras.Model):
    def __init__(self, config, output_units=2, final_activation=None, use_kan=False):
        super().__init__()
        self.config = config
        self.use_kan = use_kan
        
        # 消息传递层（与原版保持一致的命名）
        self.link_update = tf.keras.layers.GRUCell(config['link_state_dim'])  # 原版中叫 edge_update
        self.path_update = tf.keras.layers.GRUCell(config['path_state_dim'])
        
        # 【修复1: 在__init__中创建RNN层，避免在循环中重复创建】
        self.rnn_layer = tf.keras.layers.RNN(
            self.path_update, 
            return_sequences=True, 
            return_state=True
        )

        # 读出网络 - 根据use_kan参数选择MLP或KAN
        if use_kan:
            print("Using KAN (Kolmogorov-Arnold Networks) for readout layers")
            self.readout = tf.keras.Sequential([
                KANLayer(
                    config['readout_units'],
                    grid_size=5,
                    spline_order=3,
                    name='kan_layer_{}'.format(i)
                ) for i in range(config['readout_layers'])
            ])
        else:
            print("Using traditional MLP for readout layers")
            # 【修复2: 在MLP版本中添加Dropout层】
            readout_layers = []
            for _ in range(config['readout_layers']):
                readout_layers.append(
                    tf.keras.layers.Dense(
                        config['readout_units'],
                        activation='selu',
                        kernel_regularizer=tf.keras.regularizers.l2(config['l2'])
                    )
                )
                # 添加Dropout层，使用与原版一致的dropout_rate
                readout_layers.append(tf.keras.layers.Dropout(rate=0.5))  # 原版默认dropout_rate=0.5
                
            self.readout = tf.keras.Sequential(readout_layers)
        
        # 最终输出层，支持不同的激活函数
        self.final_layer = tf.keras.layers.Dense(
            output_units,
            activation=final_activation,
            kernel_regularizer=tf.keras.regularizers.l2(config['l2_2'])
        )

    def call(self, inputs, training=False):
        # 初始化状态
        link_state = tf.concat([
            tf.expand_dims(inputs['capacities'], axis=1),
            tf.zeros([inputs['n_links'], self.config['link_state_dim'] - 1])
        ], axis=1)
        
        path_state = tf.concat([
            tf.expand_dims(inputs['traffic'], axis=1),
            tf.zeros([inputs['n_paths'], self.config['path_state_dim'] - 1])
        ], axis=1)

        links = inputs['links']
        paths = inputs['paths']
        seqs = inputs['sequences']
        
        # T 轮消息传递（使用与原版相同的 RNN 处理）
        for _ in range(self.config['T']):
            # 收集每条边上的链路状态
            h_ = tf.gather(link_state, links)
            
            # 构建路径的序列输入 - 与原版完全一致
            ids = tf.stack([paths, seqs], axis=1)
            max_len = tf.reduce_max(seqs) + 1
            shape = tf.stack([inputs['n_paths'], max_len, self.config['link_state_dim']])
            
            # 计算每条路径的长度
            # 注意：segment_sum 要求 segment_ids 是排序的
            unique_paths, _ = tf.unique(paths)
            lens = tf.math.unsorted_segment_sum(
                data=tf.ones_like(paths, dtype=tf.int32),
                segment_ids=paths, 
                num_segments=inputs['n_paths']
            )
            
            # 将链路状态散布到序列格式 [n_paths, max_len, link_state_dim]
            link_inputs = tf.scatter_nd(ids, h_, shape)
            
            # 使用 masking 来处理变长序列
            # 创建 mask: True 表示有效位置，False 表示 padding
            mask = tf.sequence_mask(lens, maxlen=max_len, dtype=tf.bool)
            
            # 【修复: 使用预先创建的RNN层，而不是在循环中重复创建】
            # RNN 前向传播
            outputs, path_state = self.rnn_layer(
                link_inputs, 
                initial_state=path_state, 
                mask=mask,
                training=training
            )
            
            # 从 RNN 输出中提取对应路径位置的结果
            m = tf.gather_nd(outputs, ids)
            
            # 按链路聚合所有路径的消息
            m = tf.math.unsorted_segment_sum(m, links, inputs['n_links'])
            
            # 更新链路状态
            link_state, _ = self.link_update(m, [link_state])

        # 读出阶段
        readout_output = self.readout(path_state, training=training)
        final_input = tf.concat([readout_output, path_state], axis=1)
        return self.final_layer(final_input)

# ==============================================================================
# 3. 损失函数 (支持延迟和丢包两种任务 + 物理约束)
# ==============================================================================

def gradient_constraint_loss(model, features, predictions, use_hard_constraint=True):
    """
    梯度约束损失函数 - 支持软约束和硬约束
    强制模型学习符合物理直觉的"流量-延迟"正相关关系
    
    Args:
        model: 模型实例
        features: 输入特征
        predictions: 模型预测
        use_hard_constraint: True为硬约束(逐样本), False为软约束(批次平均)
    
    软约束公式: L_gradient = ReLU(-E_batch[gk]) = max(0, -1/|batch| * sum(gk))
    硬约束公式: L_gradient = E_batch[ReLU(-gk)] = mean(max(0, -gk))
    """
    traffic = features['traffic']
    
    if predictions.shape[1] != 2:
        return tf.constant(0.0, dtype=tf.float32)
    
    with tf.GradientTape() as grad_tape:
        grad_tape.watch(traffic)
        pred_with_grad = model(features, training=True)
        loc = pred_with_grad[:, 0]
    
    gradients = grad_tape.gradient(loc, traffic)
    
    if gradients is None:
        return tf.constant(0.0, dtype=tf.float32)
    
    if use_hard_constraint:
        # 硬约束：先对每个样本的负梯度应用ReLU，再求平均值
        # L_gradient = E_batch[ReLU(-gk)] = mean(max(0, -gk))
        gradient_penalties = tf.nn.relu(-gradients)
        return tf.reduce_mean(gradient_penalties)
    else:
        # 软约束：计算批次内的平均梯度，然后应用ReLU
        # L_gradient = ReLU(-E_batch[gk]) = max(0, -1/|batch| * sum(gk))
        batch_mean_gradient = tf.reduce_mean(gradients)
        gradient_penalty = tf.nn.relu(-batch_mean_gradient)
        return gradient_penalty

def heteroscedastic_loss(y_true, y_pred):
    """异方差损失函数 - 用于延迟预测
    
    这个实现与原始routenet.py中的实现保持一致：
    - 使用相同的scale计算方式（包含c偏移常数）
    - 使用相同的_2sigma计算方式
    """
    loc = y_pred[:, 0]
    
    # 与原版保持一致的scale计算，包含重要的c偏移常数
    c = tf.math.log(tf.math.expm1(tf.constant(0.098, dtype=tf.float32)))
    scale = tf.nn.softplus(c + y_pred[:, 1]) + 1e-9
    
    delay_true = y_true['delay']
    jitter_true = y_true['jitter']
    packets_true = y_true['packets'] 
    drops_true = y_true['drops']
    
    n = packets_true - drops_true
    # 与原版保持一致的_2sigma计算
    _2sigma = tf.constant(2.0, dtype=tf.float32) * tf.square(scale)
    
    nll = (n * jitter_true / _2sigma + 
           n * tf.square(delay_true - loc) / _2sigma + 
           n * tf.math.log(scale))
           
    return tf.reduce_sum(nll) / 1e6

def binomial_loss(y_true, y_pred):
    """二项分布损失函数 - 用于丢包预测
    
    这个实现与原始routenet.py中的实现保持一致：
    - 使用 sigmoid_cross_entropy_with_logits 来计算交叉熵
    - 使用 packets 作为权重
    - 使用相同的缩放因子 1e5
    """
    # y_pred 是 logits（未经过 sigmoid）
    logits = y_pred[:, 0]
    
    packets_true = y_true['packets']
    drops_true = y_true['drops']
    
    # 计算真实丢包率 (这是标签)
    loss_ratio = drops_true / (packets_true + 1e-9)
    
    # 使用与原版相同的公式：
    # tf.reduce_sum(packets * sigmoid_cross_entropy_with_logits(labels=loss_ratio, logits=logits)) / 1e5
    loss = tf.reduce_sum(
        packets_true * tf.nn.sigmoid_cross_entropy_with_logits(
            labels=loss_ratio,
            logits=logits
        )
    ) / 1e5  # 使用与原版相同的缩放因子
    
    return loss

def physics_informed_loss(y_true, y_pred, model, features, lambda_physics=0.1, use_hard_constraint=True):
    """
    物理约束损失函数 (Physics-Informed Loss Function)
    
    总损失函数: L_total = L_hetero + λ * L_gradient
    - L_hetero: 异方差损失函数（数据拟合项）
    - L_gradient: 梯度约束损失（物理约束项）
    - λ: 平衡超参数
    
    Args:
        y_true: 真实标签
        y_pred: 模型预测
        model: 模型实例（用于计算梯度）
        features: 输入特征（包含traffic）
        lambda_physics: 物理约束权重系数
        use_hard_constraint: True为硬约束，False为软约束
    """
    # 数据拟合项：异方差损失
    l_hetero = heteroscedastic_loss(y_true, y_pred)
    
    # 物理约束项：梯度约束损失
    l_gradient = gradient_constraint_loss(model, features, y_pred, use_hard_constraint)
    
    # 总损失
    l_total = l_hetero + lambda_physics * l_gradient
    
    return l_total, l_hetero, l_gradient

def create_model_and_loss_fn(config, target, use_kan=False, use_physics_loss=False, use_hard_constraint=True, lambda_physics=0.1):
    """根据target参数创建相应的模型和损失函数
    
    Args:
        config: 模型配置
        target: 预测目标 ('delay' 或 'drops')
        use_kan: 是否使用KAN架构
        use_physics_loss: 是否使用物理约束损失函数
        use_hard_constraint: True为硬约束(逐样本)，False为软约束(批次平均)
        lambda_physics: 物理约束权重系数
    """
    model_type = "KAN-based" if use_kan else "MLP-based"
    
    # 确定约束类型
    if use_physics_loss:
        constraint_type = "hard-constraint" if use_hard_constraint else "soft-constraint"
    else:
        constraint_type = "standard"
    
    if target == 'delay':
        # 延迟预测模型
        model = RouteNet(config, output_units=2, final_activation=None, use_kan=use_kan)
        
        if use_physics_loss:
            # 使用物理约束损失函数
            def loss_fn(labels, predictions, model=model, features=None):
                if features is None:
                    # 如果没有features，退回到标准异方差损失
                    return heteroscedastic_loss(labels, predictions)
                return physics_informed_loss(labels, predictions, model, features, lambda_physics, use_hard_constraint)
            print("Created {} delay prediction model with {} (λ={})".format(
                model_type, constraint_type, lambda_physics))
        else:
            # 使用标准异方差损失
            loss_fn = heteroscedastic_loss
            print("Created {} delay prediction model with {} loss".format(
                model_type, constraint_type))
            
    elif target == 'drops':
        # 丢包预测模型
        model = RouteNet(config, output_units=1, final_activation=None, use_kan=use_kan)
        loss_fn = binomial_loss
        if use_physics_loss:
            print("Warning: Physics constraints are not implemented for drops prediction. Using standard binomial loss.")
        print("Created {} drop prediction model with binomial loss".format(model_type))
    else:
        raise ValueError("Unsupported target: {}. Choose 'delay' or 'drops'".format(target))
    
    return model, loss_fn

@tf.function
def train_step(model, optimizer, features, labels, loss_fn, use_physics_loss=False):
    """训练步骤 - 支持物理约束损失"""
    with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        
        if use_physics_loss:
            # 物理约束损失函数需要额外参数
            if hasattr(loss_fn, '__call__'):
                try:
                    # 尝试调用物理约束损失函数
                    loss_result = loss_fn(labels, predictions, model, features)
                    if isinstance(loss_result, tuple):
                        # 返回 (total_loss, hetero_loss, gradient_loss)
                        loss, l_hetero, l_gradient = loss_result
                    else:
                        loss = loss_result
                        l_hetero = tf.constant(0.0)
                        l_gradient = tf.constant(0.0)
                except:
                    # 退回到标准损失
                    loss = loss_fn(labels, predictions)
                    l_hetero = loss
                    l_gradient = tf.constant(0.0)
            else:
                loss = loss_fn(labels, predictions)
                l_hetero = loss
                l_gradient = tf.constant(0.0)
        else:
            # 标准损失函数
            loss = loss_fn(labels, predictions)
            l_hetero = loss
            l_gradient = tf.constant(0.0)
            
        # 添加模型正则化损失
        loss += sum(model.losses)
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss, predictions, l_hetero, l_gradient

@tf.function  
def eval_step(model, features, labels, loss_fn):
    predictions = model(features, training=False)
    loss = loss_fn(labels, predictions)
    loss += sum(model.losses)
    return loss, predictions

# ==============================================================================
# 4. 主执行逻辑
# ==============================================================================

def main(args):
    config = {
        'link_state_dim': 4,
        'path_state_dim': 2, 
        'T': 3,
        'readout_units': 8,
        'readout_layers': 2,
        'l2': 0.1,
        'l2_2': 0.01,
    }

    # 设置 TensorBoard 日志目录，包含target和KAN信息
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_type = "kan" if args.kan else "mlp"
    log_dir = os.path.join(args.model_dir, 'logs', '{}_{}_{}'.format(args.target, model_type, current_time))
    train_log_dir = os.path.join(log_dir, 'train')
    val_log_dir = os.path.join(log_dir, 'validation')
    
    # 创建 TensorBoard 写入器
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    train_files = tf.io.gfile.glob(os.path.join(args.train_dir, '*.tfrecords'))
    train_dataset = create_dataset(train_files, args.batch_size, is_training=True)
    print("Found {} training files.".format(len(train_files)))

    eval_files = tf.io.gfile.glob(os.path.join(args.eval_dir, '*.tfrecords'))
    eval_dataset = create_dataset(eval_files, args.batch_size, is_training=False)
    print("Found {} evaluation files.".format(len(eval_files)))

    # 根据target创建模型和损失函数
    model, loss_fn = create_model_and_loss_fn(config, args.target, 
                                             use_kan=args.kan, 
                                             use_physics_loss=args.physics_loss,
                                             use_hard_constraint=args.hard_physics,
                                             lambda_physics=args.lambda_physics)
    
    # 创建动态学习率调度器
    if args.lr_schedule == 'exponential':
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=args.learning_rate,
            decay_steps=args.decay_steps,
            decay_rate=args.decay_rate,
            staircase=True
        )
        print("Using exponential decay learning rate schedule")
    elif args.lr_schedule == 'cosine':
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=args.learning_rate,
            decay_steps=args.epochs * args.steps_per_epoch,
            alpha=0.01  # 最小学习率是初始学习率的1%
        )
        print("Using cosine decay learning rate schedule")
    elif args.lr_schedule == 'polynomial':
        lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=args.learning_rate,
            decay_steps=args.epochs * args.steps_per_epoch,
            end_learning_rate=args.learning_rate * 0.01,
            power=1.0
        )
        print("Using polynomial decay learning rate schedule")
    elif args.lr_schedule == 'plateau':
        # 用于ReduceLROnPlateau的初始学习率
        lr_schedule = args.learning_rate
        print("Using ReduceLROnPlateau learning rate schedule")
    else:
        # 固定学习率
        lr_schedule = args.learning_rate
        print("Using fixed learning rate: {}".format(args.learning_rate))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # 如果使用plateau调度，创建ReduceLROnPlateau callback
    reduce_lr_callback = None
    if args.lr_schedule == 'plateau':
        reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            min_lr=args.learning_rate * 0.001,
            verbose=1
        )
    
    print("Starting training for target: {}".format(args.target))
    print("TensorBoard logs will be saved to: {}".format(log_dir))
    print("Run 'tensorboard --logdir {}' to view training progress".format(log_dir))
    
    best_eval_loss = float('inf')
    global_step = 0
    
    for epoch in range(args.epochs):
        print("\nEpoch {}/{}".format(epoch + 1, args.epochs))
        
        # 训练
        total_train_loss = 0.0
        total_hetero_loss = 0.0
        total_gradient_loss = 0.0
        train_step_count = 0
        pbar = tqdm(train_dataset, desc="Training Epoch {}".format(epoch+1))
        
        for features, labels in pbar:
            loss, predictions, l_hetero, l_gradient = train_step(model, optimizer, features, labels, loss_fn, use_physics_loss=args.physics_loss)
            total_train_loss += loss
            total_hetero_loss += l_hetero
            total_gradient_loss += l_gradient
            train_step_count += 1
            global_step += 1
            
            # 记录每个批次的损失到 TensorBoard (每10步记录一次)
            if global_step % 10 == 0:
                with train_summary_writer.as_default():
                    tf.summary.scalar('batch_loss', loss, step=global_step)
                    if args.physics_loss:
                        tf.summary.scalar('batch_hetero_loss', l_hetero, step=global_step)
                        tf.summary.scalar('batch_gradient_loss', l_gradient, step=global_step)
                    # 记录当前学习率
                    current_lr = optimizer.learning_rate
                    if hasattr(current_lr, 'numpy'):
                        current_lr = current_lr.numpy()
                    tf.summary.scalar('learning_rate', current_lr, step=global_step)
            
            # 更新进度条显示
            if args.physics_loss:
                pbar.set_postfix({
                    'total': '{:.4f}'.format(loss), 
                    'hetero': '{:.4f}'.format(l_hetero),
                    'grad': '{:.4f}'.format(l_gradient),
                    'step': global_step
                })
            else:
                pbar.set_postfix({'loss': '{:.4f}'.format(loss), 'step': global_step})
                
        avg_train_loss = total_train_loss / train_step_count
        avg_hetero_loss = total_hetero_loss / train_step_count 
        avg_gradient_loss = total_gradient_loss / train_step_count

        # 记录训练的平均损失
        with train_summary_writer.as_default():
            tf.summary.scalar('epoch_loss', avg_train_loss, step=epoch + 1)
            if args.physics_loss:
                tf.summary.scalar('epoch_hetero_loss', avg_hetero_loss, step=epoch + 1)
                tf.summary.scalar('epoch_gradient_loss', avg_gradient_loss, step=epoch + 1)

        # 评估
        total_eval_loss = 0.0
        eval_step_count = 0
        pbar_eval = tqdm(eval_dataset, desc="Evaluating Epoch {}".format(epoch+1))
        
        for features, labels in pbar_eval:
            loss, predictions = eval_step(model, features, labels, loss_fn)
            total_eval_loss += loss
            eval_step_count += 1
            pbar_eval.set_postfix({'loss': '{:.4f}'.format(loss)})
            
        avg_eval_loss = total_eval_loss / eval_step_count

        # 记录验证损失
        with val_summary_writer.as_default():
            tf.summary.scalar('epoch_loss', avg_eval_loss, step=epoch + 1)
            # 记录每个epoch结束时的学习率
            current_lr = optimizer.learning_rate
            if hasattr(current_lr, 'numpy'):
                current_lr = current_lr.numpy()
            tf.summary.scalar('learning_rate_epoch', current_lr, step=epoch + 1)

        # 如果使用ReduceLROnPlateau，手动调整学习率
        if reduce_lr_callback is not None:
            # 模拟callback行为
            if not hasattr(reduce_lr_callback, 'best'):
                reduce_lr_callback.best = avg_eval_loss
                reduce_lr_callback.wait = 0
            else:
                if avg_eval_loss < reduce_lr_callback.best:
                    reduce_lr_callback.best = avg_eval_loss
                    reduce_lr_callback.wait = 0
                else:
                    reduce_lr_callback.wait += 1
                    if reduce_lr_callback.wait >= args.plateau_patience:
                        old_lr = optimizer.learning_rate.numpy()
                        new_lr = old_lr * args.plateau_factor
                        if new_lr >= args.learning_rate * 0.001:
                            optimizer.learning_rate.assign(new_lr)
                            print("Reducing learning rate from {:.6f} to {:.6f}".format(old_lr, new_lr))
                            reduce_lr_callback.wait = 0

        # 输出epoch结果
        if args.physics_loss:
            print("Epoch {} finished. Total Loss: {:.4f} (Hetero: {:.4f}, Gradient: {:.4f}), Eval Loss: {:.4f}, LR: {:.6f}".format(
                epoch + 1, avg_train_loss, avg_hetero_loss, avg_gradient_loss, avg_eval_loss,
                optimizer.learning_rate.numpy() if hasattr(optimizer.learning_rate, 'numpy') else optimizer.learning_rate))
        else:
            print("Epoch {} finished. Avg Train Loss: {:.4f}, Avg Eval Loss: {:.4f}, LR: {:.6f}".format(
                epoch + 1, avg_train_loss, avg_eval_loss, 
                optimizer.learning_rate.numpy() if hasattr(optimizer.learning_rate, 'numpy') else optimizer.learning_rate))

        # 保存最佳模型
        if avg_eval_loss < best_eval_loss:
            print("Evaluation loss improved from {:.4f} to {:.4f}. Saving model...".format(
                best_eval_loss, avg_eval_loss))
            best_eval_loss = avg_eval_loss
            
            # 根据是否使用KAN来命名模型文件
            model_suffix = "kan_model" if args.kan else "model"
            save_path = os.path.join(args.model_dir, "best_{}_{}.weights.h5".format(args.target, model_suffix))
            model.save_weights(save_path)
            
            # 记录最佳模型的信息
            with val_summary_writer.as_default():
                tf.summary.scalar('best_loss', best_eval_loss, step=epoch + 1)
        else:
            print("Evaluation loss did not improve from {:.4f}.".format(best_eval_loss))
    
    # 训练结束后，关闭 summary writers
    train_summary_writer.close()
    val_summary_writer.close()
    print("\nTraining completed! TensorBoard logs saved to: {}".format(log_dir))
    print("To view the results, run: tensorboard --logdir {}".format(log_dir))
    
    # 根据是否使用KAN来显示模型文件名
    model_suffix = "kan_model" if args.kan else "model"
    print("Model weights saved as: {}".format(
        os.path.join(args.model_dir, "best_{}_{}.weights.h5".format(args.target, model_suffix))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RouteNet TF2 Implementation')
    parser.add_argument('--train_dir', type=str, required=True, 
                      help='Directory containing training TFRecord files')
    parser.add_argument('--eval_dir', type=str, required=True,
                      help='Directory containing evaluation TFRecord files') 
    parser.add_argument('--model_dir', type=str, required=True,
                      help='Directory to save model checkpoints and logs')
    parser.add_argument('--target', type=str, choices=['delay', 'drops'], default='delay',
                      help='Training target: "delay" for delay prediction, "drops" for packet drop prediction')
    parser.add_argument('--kan', action='store_true', 
                      help='Use KAN (Kolmogorov-Arnold Networks) instead of traditional MLP for readout layers')
    parser.add_argument('--epochs', type=int, default=10,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate for Adam optimizer')
    parser.add_argument('--lr_schedule', type=str, choices=['fixed', 'exponential', 'cosine', 'polynomial', 'plateau'], 
                      default='fixed', help='Learning rate schedule strategy')
    
    # 指数衰减参数
    parser.add_argument('--decay_steps', type=int, default=1000,
                      help='Steps for exponential decay (only for exponential schedule)')
    parser.add_argument('--decay_rate', type=float, default=0.96,
                      help='Decay rate for exponential decay (only for exponential schedule)')
    
    # Plateau调度参数
    parser.add_argument('--plateau_factor', type=float, default=0.5,
                      help='Factor to reduce learning rate on plateau (only for plateau schedule)')
    parser.add_argument('--plateau_patience', type=int, default=3,
                      help='Number of epochs to wait before reducing LR on plateau (only for plateau schedule)')
    
    # 物理约束损失参数
    parser.add_argument('--physics_loss', action='store_true',
                      help='Enable physics-informed loss function for delay prediction')
    parser.add_argument('--hard_physics', action='store_true',
                      help='Use hard constraint (per-sample) instead of soft constraint (batch-average). Only effective when --physics_loss is enabled.')
    parser.add_argument('--lambda_physics', type=float, default=0.1,
                      help='Weight coefficient for physics constraint term (default: 0.1)')
    
    # 用于cosine和polynomial调度的steps_per_epoch估计
    parser.add_argument('--steps_per_epoch', type=int, default=100,
                      help='Estimated steps per epoch for cosine/polynomial schedules')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
        
    main(args)
