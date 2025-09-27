# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import argparse
import os
from tqdm import tqdm
import datetime
import random
from tensorboard.plugins.hparams import api as hp

# ==============================================================================
# KAN (Kolmogorov-Arnold Networks) Implementation
# ==============================================================================

class KANLayer(tf.keras.layers.Layer):
    """
    KAN (Kolmogorov-Arnold Networks) Layer implementation
    简化版本，默认使用多项式基（1, x, x^2, x^3）作为可学习非线性；
    可选启用 B 样条基函数（bspline）。
    """
    
    def __init__(self, units, grid_size=5, spline_order=3, basis_type='poly', **kwargs):
        super(KANLayer, self).__init__(**kwargs)
        self.units = units
        self.grid_size = grid_size              # B样条网格间隔数量（区间数）
        self.spline_order = spline_order        # B样条阶次（degree），常用3表示三次
        self.basis_type = basis_type            # 'poly' 或 'bspline'
        # 以下在 build() 中根据 basis_type 动态确定
        self._basis_dim = None
        self._knots = None
        self._n_basis = None
        
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
        
        # 确定基函数维度，并在需要时准备B样条结点
        if self.basis_type == 'bspline':
            # Open uniform B-spline knots on [0, 1]
            import numpy as _np
            degree = int(self.spline_order)
            n_intervals = int(self.grid_size)
            # 选择基函数数量（控制点数）：n_basis = n_intervals + degree
            self._n_basis = n_intervals + degree
            # 构造 open-uniform knot 向量: [0]*(degree+1), internal uniform, [1]*(degree+1)
            if n_intervals > 1:
                internal = _np.linspace(0.0, 1.0, n_intervals + 1, dtype=_np.float32)[1:-1]
            else:
                internal = _np.array([], dtype=_np.float32)
            start = _np.zeros((degree + 1,), dtype=_np.float32)
            end = _np.ones((degree + 1,), dtype=_np.float32)
            np_knots = _np.concatenate([start, internal, end], axis=0)  # len = n_basis + degree + 1
            # 将 knots 注册为不可训练的变量，避免 tf.function 图作用域问题
            self._knots = self.add_weight(
                name='bspline_knots',
                shape=(np_knots.shape[0],),
                dtype=tf.float32,
                initializer=tf.keras.initializers.Constant(np_knots),
                trainable=False
            )
            self._basis_dim = self._n_basis
        else:
            # 多项式基：1, x, x^2, x^3
            self._basis_dim = 4

        # 样条/基函数的权重参数：每个输入-输出连接对应 basis_dim 个系数
        self.spline_weights = self.add_weight(
            name='spline_weights',
            shape=(input_dim, self.units, self._basis_dim),
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
    
    def _bspline_basis(self, u):
        """
        计算 B 样条基函数值。
        输入:
          u: [batch_size, input_dim]，定义域假设在 [0, 1]
        返回:
          basis: [batch_size, input_dim, n_basis]
        """
        degree = int(self.spline_order)
        knots = self._knots  # [n_basis + degree + 1]
        n_basis = int(self._n_basis)

        # 基础的0次基函数 B_{i,0}
        # 对每个 i (0..n_basis-1): 1 if knots[i] <= u < knots[i+1] else 0
        # 处理 u==1 的边界（归属到最后一个基函数）
        u_exp = tf.expand_dims(u, axis=-1)  # [B, D, 1]
        t_i = tf.reshape(knots[:n_basis], [1, 1, n_basis])
        t_ip1 = tf.reshape(knots[1:n_basis+1], [1, 1, n_basis])

        left = tf.cast(u_exp >= t_i, tf.float32)
        right = tf.cast(u_exp < t_ip1, tf.float32)
        B = left * right  # [B, D, n_basis]

        # 特殊处理 u==1.0: 令最后一个基函数为1
        is_one = tf.equal(u_exp, 1.0)
        any_one = tf.cast(is_one, tf.float32)
        last_hot = tf.one_hot(n_basis - 1, n_basis, dtype=tf.float32)  # [n_basis]
        last_hot = tf.reshape(last_hot, [1, 1, n_basis])
        B = tf.where(tf.reduce_any(is_one, axis=-1, keepdims=True), last_hot, B)

        # 递推计算高阶基函数
        for k in range(1, degree + 1):
            # denom1: t_{i+k} - t_i,  i=0..n_basis-1
            t_i_k = tf.reshape(knots[k:n_basis + k], [1, 1, n_basis])
            denom1 = t_i_k - t_i
            # denom2: t_{i+k+1} - t_{i+1}
            t_ip1_k1 = tf.reshape(knots[k+1:n_basis + k + 1], [1, 1, n_basis])
            denom2 = t_ip1_k1 - t_ip1

            # term1 = ((u - t_i)/denom1) * B_{i,k-1}
            numer1 = u_exp - t_i
            coef1 = tf.where(denom1 > 0, numer1 / (denom1 + 1e-12), tf.zeros_like(denom1))
            term1 = coef1 * B

            # term2 = ((t_{i+k+1} - u)/denom2) * B_{i+1,k-1}
            numer2 = t_ip1_k1 - u_exp
            coef2 = tf.where(denom2 > 0, numer2 / (denom2 + 1e-12), tf.zeros_like(denom2))
            # B shifted: B_{i+1,k-1}
            B_shift = tf.concat([B[..., 1:], tf.zeros_like(B[..., :1])], axis=-1)
            term2 = coef2 * B_shift

            B = term1 + term2

        return B  # [B, D, n_basis]

    def call(self, inputs, training=None):
        # 基础线性变换
        linear_output = tf.matmul(inputs, self.base_weight) + self.bias  # [batch_size, units]
        
        # 非线性变换：多项式或B样条
        # 将输入标准化
        x_tanh = tf.tanh(inputs)  # [-1, 1]
        if self.basis_type == 'bspline':
            # remap to [0,1]
            u = (x_tanh + 1.0) * 0.5
            basis = self._bspline_basis(u)  # [B, D, n_basis]
        else:
            # 多项式基函数：1, x, x^2, x^3
            basis = tf.stack([
                tf.ones_like(x_tanh),
                x_tanh,
                tf.square(x_tanh),
                tf.pow(x_tanh, 3)
            ], axis=-1)  # [B, D, 4]
        
        # 样条/基函数输出：[B, D, basis] @ [D, U, basis] -> [B, D, U]
        spline_contributions = tf.einsum('bid,ijd->bij', basis, self.spline_weights)  # [batch_size, input_dim, units]
        
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
            'basis_type': self.basis_type,
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

def create_dataset(filenames, batch_size, is_training=True, seed=None):
    ds = tf.data.TFRecordDataset(filenames)
    if is_training:
        # 使用提供 seed 以实现可复现 shuffle；reshuffle_each_iteration 使 epoch 变化仍受同一全局种子控制
        ds = ds.shuffle(1000, seed=seed, reshuffle_each_iteration=True)
    
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
        
        # 读出网络（支持KAN和传统MLP）
        if use_kan:
            readout_layers = []
            for i in range(config['readout_layers']):
                readout_layers.append(KANLayer(
                    config['readout_units'],
                    grid_size=config.get('kan_grid_size', 5),
                    spline_order=config.get('kan_spline_order', 3),
                    basis_type=config.get('kan_basis', 'poly')
                ))
                # 仅在中间层添加 Dropout，且需开启 use_dropout
                if (
                    i < config['readout_layers'] - 1
                    and config.get('use_dropout', True)
                ):
                    readout_layers.append(
                        tf.keras.layers.Dropout(config.get('dropout_rate', 0.1))
                    )
            self.readout = tf.keras.Sequential(readout_layers)
        else:
            # 传统 MLP 读出网络
            readout_layers = []
            for _ in range(config['readout_layers']):
                readout_layers.append(tf.keras.layers.Dense(
                    config['readout_units'], 
                    activation='selu',
                    kernel_regularizer=tf.keras.regularizers.l2(config['l2'])
                ))
                if config.get('use_dropout', True):
                    readout_layers.append(
                        tf.keras.layers.Dropout(config.get('dropout_rate', 0.1))
                    )
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

# 已移除梯度约束损失

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

def create_model_and_loss_fn(config, target, use_kan=False):
    """根据target参数创建相应的模型和损失函数（无物理/梯度约束）。"""
    model_type = "KAN-based" if use_kan else "MLP-based"

    if target == 'delay':
        model = RouteNet(config, output_units=2, final_activation=None, use_kan=use_kan)
        loss_fn = heteroscedastic_loss
        print("Created {} delay prediction model with standard heteroscedastic loss".format(model_type))
    elif target == 'drops':
        model = RouteNet(config, output_units=1, final_activation=None, use_kan=use_kan)
        loss_fn = binomial_loss
        print("Created {} drop prediction model with binomial loss".format(model_type))
    else:
        raise ValueError("Unsupported target: {}. Choose 'delay' or 'drops'".format(target))

    return model, loss_fn

@tf.function
def train_step(model, optimizer, features, labels, loss_fn):
    """标准训练步骤（无物理/梯度约束）。"""
    with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        loss = loss_fn(labels, predictions)
        # 添加模型正则化损失
        loss += sum(model.losses)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, predictions

@tf.function  
def eval_step(model, features, labels, loss_fn):
    predictions = model(features, training=False)
    loss = loss_fn(labels, predictions)
    loss += sum(model.losses)
    return loss, predictions

# ==============================================================================
# 4. 主执行逻辑
# ==============================================================================

def set_global_seed(seed: int, deterministic: bool = True):
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        if deterministic:
            tf.config.experimental.enable_op_determinism()
    except Exception:  # pragma: no cover
        pass
    print(f"[SEED] Global seed set to {seed} deterministic={deterministic}")


def main(args):
    # 设置全局随机种子，保证 shuffle、权重初始化、TF 算子一致
    set_global_seed(args.seed)
    config = {
        'link_state_dim': 4,
        'path_state_dim': 2, 
        'T': 3,
        'readout_units': 8,
        'readout_layers': 2,
        'l2': 0.1,
        'l2_2': 0.01,
        # KAN 参数默认值（当使用KAN时生效）
        'kan_basis': 'poly',        # 'poly' 或 'bspline'
        'kan_grid_size': 5,         # B样条间隔数，只有在bspline时使用
        'kan_spline_order': 3,      # B样条阶次（degree），典型为3
        # Dropout 默认开启，率为 0.1
        'use_dropout': True,
        'dropout_rate': 0.1,
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

    # 记录 HParams（供 TensorBoard HParams Plugin 使用）
    hparams = {
        'target': args.target,
        'kan': int(args.kan),
        'kan_basis': args.kan_basis or 'none',
        'kan_grid_size': args.kan_grid_size if args.kan_grid_size is not None else -1,
        'kan_spline_order': args.kan_spline_order if args.kan_spline_order is not None else -1,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'lr_schedule': args.lr_schedule,
        'dropout_rate': args.dropout_rate,
        'use_dropout': int(not args.no_dropout),
        'decay_steps': args.decay_steps,
        'decay_rate': args.decay_rate,
        'plateau_factor': args.plateau_factor,
        'plateau_patience': args.plateau_patience,
        'early_stopping': int(args.early_stopping),
        'early_stopping_patience': args.early_stopping_patience,
        'seed': args.seed,
    }
    with train_summary_writer.as_default():
        hp.hparams(hparams)

    train_files = tf.io.gfile.glob(os.path.join(args.train_dir, '*.tfrecords'))
    train_dataset = create_dataset(train_files, args.batch_size, is_training=True, seed=args.seed)
    print("Found {} training files.".format(len(train_files)))

    eval_files = tf.io.gfile.glob(os.path.join(args.eval_dir, '*.tfrecords'))
    eval_dataset = create_dataset(eval_files, args.batch_size, is_training=False)
    print("Found {} evaluation files.".format(len(eval_files)))

    # 根据target创建模型和损失函数
    # 将 CLI 的 KAN 参数写入 config（仅当用户传入时覆盖默认值）
    if args.kan_basis is not None:
        config['kan_basis'] = args.kan_basis
    if args.kan_grid_size is not None:
        config['kan_grid_size'] = args.kan_grid_size
    if args.kan_spline_order is not None:
        config['kan_spline_order'] = args.kan_spline_order

    # 覆盖 Dropout 配置
    if args.no_dropout:
        config['use_dropout'] = False
    if args.dropout_rate is not None:
        config['dropout_rate'] = float(args.dropout_rate)

    model, loss_fn = create_model_and_loss_fn(config, args.target, use_kan=args.kan)
    
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
    
    # Plateau 调度自定义实现（避免使用回调对象导致 best=None 比较出错）
    plateau_best = None
    plateau_wait = 0
    
    print("Starting training for target: {}".format(args.target))
    print("TensorBoard logs will be saved to: {}".format(log_dir))
    print("Run 'tensorboard --logdir {}' to view training progress".format(log_dir))
    
    best_eval_loss = float('inf')
    global_step = 0
    
    # 早停变量初始化
    if args.early_stopping:
        early_stopping_patience = args.early_stopping_patience
        early_stopping_counter = 0
        early_stopping_min_delta = args.early_stopping_min_delta
        best_weights = None
        print(f"🛑 Early stopping enabled: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")
    else:
        early_stopping_min_delta = 0.0  # 为非早停模式设置默认值
    
    for epoch in range(args.epochs):
        print("\nEpoch {}/{}".format(epoch + 1, args.epochs))
        
        # 训练
        total_train_loss = 0.0
        train_step_count = 0
        pbar = tqdm(train_dataset, desc="Training Epoch {}".format(epoch+1))
        
        for features, labels in pbar:
            # 标准训练步骤
            loss, predictions = train_step(model, optimizer, features, labels, loss_fn)
            
            total_train_loss += loss
            
            train_step_count += 1
            global_step += 1
            
            # 记录每个批次的损失到 TensorBoard (每10步记录一次)
            if global_step % 10 == 0:
                with train_summary_writer.as_default():
                    tf.summary.scalar('batch_loss', loss, step=global_step)
                    # 记录当前学习率
                    current_lr = optimizer.learning_rate
                    if hasattr(current_lr, 'numpy'):
                        current_lr = current_lr.numpy()
                    tf.summary.scalar('learning_rate', current_lr, step=global_step)
            
            # 更新进度条显示
            pbar.set_postfix({'loss': '{:.4f}'.format(loss), 'step': global_step})
                
        avg_train_loss = total_train_loss / train_step_count

        # 记录训练的平均损失
        with train_summary_writer.as_default():
            tf.summary.scalar('epoch_loss', avg_train_loss, step=epoch + 1)

        # 评估
        total_eval_loss = 0.0
        eval_step_count = 0
        pbar_eval = tqdm(eval_dataset, desc="Evaluating Epoch {}".format(epoch+1))
        
        for features, labels in pbar_eval:
            # 评估步骤
            loss, predictions = eval_step(model, features, labels, loss_fn)
            
            total_eval_loss += loss
            eval_step_count += 1
            pbar_eval.set_postfix({'loss': '{:.4f}'.format(loss)})
            
        avg_eval_loss = total_eval_loss / eval_step_count

        # 记录验证损失 (val writer)
        with val_summary_writer.as_default():
            tf.summary.scalar('epoch_loss', avg_eval_loss, step=epoch + 1)
            current_lr = optimizer.learning_rate
            if hasattr(current_lr, 'numpy'):
                current_lr = current_lr.numpy()
            tf.summary.scalar('learning_rate_epoch', current_lr, step=epoch + 1)
        # 为 HParams 插件在同一 writer 下再记录一份验证指标
        with train_summary_writer.as_default():
            tf.summary.scalar('val_epoch_loss', avg_eval_loss, step=epoch + 1)

        # Plateau 学习率调度逻辑
        if args.lr_schedule == 'plateau':
            cur_val = float(avg_eval_loss.numpy() if hasattr(avg_eval_loss, 'numpy') else avg_eval_loss)
            if plateau_best is None or cur_val < plateau_best - 0.0:  # 不额外使用 min_delta，这里简单判断
                plateau_best = cur_val
                plateau_wait = 0
            else:
                plateau_wait += 1
                if plateau_wait >= args.plateau_patience:
                    old_lr = optimizer.learning_rate.numpy() if hasattr(optimizer.learning_rate, 'numpy') else float(optimizer.learning_rate)
                    new_lr = max(old_lr * args.plateau_factor, args.learning_rate * 0.001)
                    if new_lr < old_lr:
                        optimizer.learning_rate.assign(new_lr)
                        print("[Plateau] Reducing learning rate from {:.6f} to {:.6f}".format(old_lr, new_lr))
                    plateau_wait = 0

        # 输出epoch结果
        print("Epoch {} finished. Avg Train Loss: {:.4f}, Avg Eval Loss: {:.4f}, LR: {:.6f}".format(
            epoch + 1, avg_train_loss, avg_eval_loss, 
            optimizer.learning_rate.numpy() if hasattr(optimizer.learning_rate, 'numpy') else optimizer.learning_rate))

        # 保存最佳模型
        if avg_eval_loss < best_eval_loss - early_stopping_min_delta if args.early_stopping else avg_eval_loss < best_eval_loss:
            print("Evaluation loss improved from {:.4f} to {:.4f}. Saving model...".format(
                best_eval_loss, avg_eval_loss))
            best_eval_loss = avg_eval_loss
            
            # 根据是否使用KAN来命名模型文件
            model_suffix = "kan_model" if args.kan else "model"
            save_path = os.path.join(args.model_dir, "best_{}_{}.weights.h5".format(args.target, model_suffix))
            model.save_weights(save_path)
            
            # 早停：重置计数器并保存最佳权重
            if args.early_stopping:
                early_stopping_counter = 0
                if args.early_stopping_restore_best:
                    best_weights = model.get_weights()
                    print("🔄 Early stopping: best weights saved")
            
            # 记录最佳模型的信息
            with val_summary_writer.as_default():
                tf.summary.scalar('best_loss', best_eval_loss, step=epoch + 1)
        else:
            print("Evaluation loss did not improve from {:.4f}.".format(best_eval_loss))
            
            # 早停：增加计数器
            if args.early_stopping:
                early_stopping_counter += 1
                print(f"🛑 Early stopping: {early_stopping_counter}/{early_stopping_patience}")
                
                # 检查是否需要早停
                if early_stopping_counter >= early_stopping_patience:
                    print(f"🛑 Early stopping triggered after {epoch + 1} epochs!")
                    print(f"🛑 No improvement for {early_stopping_patience} consecutive epochs")
                    
                    # 恢复最佳权重
                    if args.early_stopping_restore_best and best_weights is not None:
                        model.set_weights(best_weights)
                        print("🔄 Restored best model weights")
                    
                    break
    
    # 训练结束后，关闭 summary writers
    train_summary_writer.close()
    val_summary_writer.close()
    
    # 训练完成统计
    if args.early_stopping:
        if early_stopping_counter >= early_stopping_patience:
            print(f"\n🛑 Training stopped early after {epoch + 1}/{args.epochs} epochs")
            print(f"🛑 Best validation loss: {best_eval_loss:.6f}")
        else:
            print(f"\n✅ Training completed normally after {args.epochs} epochs")
            print(f"✅ Best validation loss: {best_eval_loss:.6f}")
    else:
        print(f"\n✅ Training completed after {args.epochs} epochs")
        print(f"✅ Best validation loss: {best_eval_loss:.6f}")
    
    print("TensorBoard logs saved to: {}".format(log_dir))
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
    # KAN 参数
    parser.add_argument('--kan_basis', type=str, choices=['poly', 'bspline'], default=None,
                      help='KAN basis type for readout: poly (default) or bspline')
    parser.add_argument('--kan_grid_size', type=int, default=None,
                      help='Number of intervals for B-spline grid (only for bspline basis)')
    parser.add_argument('--kan_spline_order', type=int, default=None,
                      help='Degree/order of B-spline basis (only for bspline basis)')
    parser.add_argument('--epochs', type=int, default=10,
                      help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate for Adam optimizer')
    parser.add_argument('--lr_schedule', type=str, choices=['fixed', 'exponential', 'cosine', 'polynomial', 'plateau'], 
                      default='fixed', help='Learning rate schedule strategy')
    # Dropout 开关与比例
    parser.add_argument('--no_dropout', action='store_true',
                      help='Disable dropout layers in readout (default: enabled)')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                      help='Dropout rate when enabled (default: 0.1)')
    
    # 指数衰减参数
    parser.add_argument('--decay_steps', type=int, default=1000,
                      help='Steps for exponential decay (only for exponential schedule)')
    parser.add_argument('--decay_rate', type=float, default=0.96,
                      help='Decay rate for exponential decay (only for exponential schedule)')
    
    # Plateau调度参数
    parser.add_argument('--plateau_factor', type=float, default=0.5,
                      help='Factor to reduce learning rate on plateau (only for plateau schedule)')
    parser.add_argument('--plateau_patience', type=int, default=8,
                      help='Number of epochs to wait before reducing LR on plateau (only for plateau schedule)')
    
    # 早停参数
    parser.add_argument('--early_stopping', action='store_true',
                      help='Enable early stopping based on validation loss')
    parser.add_argument('--early_stopping_patience', type=int, default=8,
                      help='Number of epochs to wait before early stopping (default: 5)')
    parser.add_argument('--early_stopping_min_delta', type=float, default=1e-6,
                      help='Minimum change in monitored quantity to qualify as an improvement (default: 1e-6)')
    parser.add_argument('--early_stopping_restore_best', action='store_true',
                      help='Restore model weights from the epoch with the best value of the monitored quantity')
    
    # 已移除物理约束与课程学习相关参数
    
    # 用于cosine和polynomial调度的steps_per_epoch估计
    parser.add_argument('--steps_per_epoch', type=int, default=100,
                      help='Estimated steps per epoch for cosine/polynomial schedules')
    parser.add_argument('--seed', type=int, default=137, help='Global random seed for reproducibility')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
        
    main(args)
