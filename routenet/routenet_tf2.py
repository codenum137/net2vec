# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import argparse
import os
from tqdm import tqdm
import datetime
import random


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
        # 使用提供的 seed 以实现可复现性
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
    def __init__(self, config, output_units=2, final_activation=None):
        super().__init__()
        self.config = config
        # SAE 设置
        self.sae_enabled = bool(config.get('sae_enabled', False))

        # 消息传递层
        self.link_update = tf.keras.layers.GRUCell(config['link_state_dim'])
        self.path_update = tf.keras.layers.GRUCell(config['path_state_dim'])

        # 预创建 RNN 层
        self.rnn_layer = tf.keras.layers.RNN(
            self.path_update,
            return_sequences=True,
            return_state=True
        )

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

        # 最终输出层
        self.final_layer = tf.keras.layers.Dense(
            output_units,
            activation=final_activation,
            kernel_regularizer=tf.keras.regularizers.l2(config['l2_2'])
        )

        # 稀疏自编码器 (SAE)
        if self.sae_enabled:
            sae_hidden = int(config.get('sae_hidden_dim', 64))
            sae_latent = int(config.get('sae_latent_dim', 16))
            sae_act = config.get('sae_activation', 'relu')
            self.sae_encoder = tf.keras.Sequential([
                tf.keras.layers.Dense(sae_hidden, activation=sae_act, name='sae_enc_hidden'),
                tf.keras.layers.Dense(sae_latent, activation=sae_act, name='sae_enc_latent')
            ], name='sae_encoder')
            self.sae_decoder = tf.keras.Sequential([
                tf.keras.layers.Dense(sae_hidden, activation=sae_act, name='sae_dec_hidden'),
                tf.keras.layers.Dense(config['link_state_dim'], activation=None, name='sae_dec_out')
            ], name='sae_decoder')

    def call(self, inputs, training=False, return_aux=False):
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
            
            # 【使用预先创建的RNN层，而不是在循环中重复创建】
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
        pred = self.final_layer(final_input)

        if self.sae_enabled:
            # SAE 使用最终的 link_state 作为重构目标
            original = link_state  # [n_links, link_state_dim]
            latent = self.sae_encoder(original, training=training)  # [n_links, latent]
            recon = self.sae_decoder(latent, training=training)     # [n_links, link_state_dim]
            if return_aux:
                return pred, { 'original_link_state': original, 'recon': recon, 'latent': latent }
        if return_aux:
            return pred, None
        return pred


# ==============================================================================
# 3. 损失函数
# ==============================================================================


def make_delay_loss(config):
    """创建延迟预测损失函数（支持 legacy 与 standard 两种形式）。

    不再在 loss_fn 上维护跨图属性，所有诊断指标由 train_step / eval_step 内部计算后返回。
    """
    variant = config.get('loss_variant', 'standard')

    def legacy_loss(y_true, y_pred):
        loc = y_pred[:, 0]
        c = tf.math.log(tf.math.expm1(tf.constant(0.098, dtype=tf.float32)))
        scale = tf.nn.softplus(c + y_pred[:, 1]) + 1e-9
        delay_true = y_true['delay']
        jitter_true = y_true['jitter']
        packets_true = y_true['packets']
        drops_true = y_true['drops']
        n = packets_true - drops_true
        _2sigma = tf.constant(2.0, dtype=tf.float32) * tf.square(scale)
        nll = (n * jitter_true / _2sigma + n * tf.square(delay_true - loc) / _2sigma + n * tf.math.log(scale))
        loss = tf.reduce_sum(nll) / 1e6
        return loss

    def standard_loss(y_true, y_pred):
        loc = y_pred[:, 0]
        raw_scale = y_pred[:, 1]
        sigma = tf.nn.softplus(raw_scale) + 1e-6  # 保证 >0
        delay_true = y_true['delay']
        packets_true = y_true['packets']
        drops_true = y_true['drops']
        n = packets_true - drops_true
        res = delay_true - loc
        nll = 0.5 * (tf.square(res) / tf.square(sigma) + 2.0 * tf.math.log(sigma) + tf.math.log(2.0 * tf.constant(3.141592653589793, dtype=tf.float32)))
        weighted = n * nll
        loss = tf.reduce_sum(weighted) / (tf.reduce_sum(n) + 1e-9)
        return loss

    def loss_fn(y_true, y_pred):
        if variant == 'legacy':
            return legacy_loss(y_true, y_pred)
        else:
            return standard_loss(y_true, y_pred)

    loss_fn.variant = variant
    return loss_fn

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

def create_model_and_loss_fn(config, target):
    """根据 target 创建相应模型与损失函数 (仅 MLP)。"""
    if target == 'delay':
        model = RouteNet(config, output_units=2, final_activation=None)
        loss_fn = make_delay_loss(config)
        print("Created MLP-based delay prediction model with {} loss variant".format(loss_fn.variant))
    elif target == 'drops':
        model = RouteNet(config, output_units=1, final_activation=None)
        loss_fn = binomial_loss
        print("Created MLP-based drop prediction model with binomial loss")
    else:
        raise ValueError("Unsupported target: {}. Choose 'delay' or 'drops'".format(target))
    return model, loss_fn

@tf.function(experimental_relax_shapes=True)
def train_step(model, optimizer, features, labels, loss_fn, sae_alpha=1.0, sae_beta=1e-4):
    """标准训练步骤 + 可选SAE联合训练。

    返回:
      total_loss, metrics_dict, predictions
    metrics_dict: {'delay_loss':..., 'recon_loss':..., 'sparsity_loss':...}
    """
    with tf.GradientTape() as tape:
        outputs = model(features, training=True, return_aux=model.sae_enabled)
        if model.sae_enabled:
            predictions, aux = outputs
        else:
            predictions = outputs
            aux = None
        delay_loss = loss_fn(labels, predictions)
        recon_loss = tf.constant(0.0, dtype=tf.float32)
        sparsity_loss = tf.constant(0.0, dtype=tf.float32)
        if model.sae_enabled and aux is not None:
            recon_loss = tf.reduce_mean(tf.square(aux['recon'] - aux['original_link_state']))
            sparsity_loss = tf.reduce_mean(tf.abs(aux['latent']))
        total_loss = delay_loss + sae_alpha * recon_loss + sae_beta * sparsity_loss
        total_loss += tf.add_n(model.losses) if model.losses else 0.0

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # 诊断指标（在图内计算并返回）
    loc = predictions[:, 0]
    raw_scale = predictions[:, 1]
    if loss_fn.variant == 'legacy':
        c = tf.math.log(tf.math.expm1(tf.constant(0.098, dtype=tf.float32)))
        scale = tf.nn.softplus(c + raw_scale) + 1e-9
    else:
        scale = tf.nn.softplus(raw_scale) + 1e-6
    delay_true = labels['delay']
    res = delay_true - loc
    rmse = tf.sqrt(tf.reduce_mean(tf.square(res)))
    mean_scale = tf.reduce_mean(scale)
    ratio = rmse / (mean_scale + 1e-9)
    return total_loss, delay_loss, recon_loss, sparsity_loss, mean_scale, rmse, ratio, predictions

@tf.function(experimental_relax_shapes=True)
def eval_step(model, features, labels, loss_fn, sae_alpha=1.0, sae_beta=1e-4):
    outputs = model(features, training=False, return_aux=model.sae_enabled)
    if model.sae_enabled:
        predictions, aux = outputs
    else:
        predictions = outputs
        aux = None
    delay_loss = loss_fn(labels, predictions)
    recon_loss = tf.constant(0.0, dtype=tf.float32)
    sparsity_loss = tf.constant(0.0, dtype=tf.float32)
    if model.sae_enabled and aux is not None:
        recon_loss = tf.reduce_mean(tf.square(aux['recon'] - aux['original_link_state']))
        sparsity_loss = tf.reduce_mean(tf.abs(aux['latent']))
    total_loss = delay_loss + sae_alpha * recon_loss + sae_beta * sparsity_loss
    total_loss += tf.add_n(model.losses) if model.losses else 0.0
    loc = predictions[:, 0]
    raw_scale = predictions[:, 1]
    if loss_fn.variant == 'legacy':
        c = tf.math.log(tf.math.expm1(tf.constant(0.098, dtype=tf.float32)))
        scale = tf.nn.softplus(c + raw_scale) + 1e-9
    else:
        scale = tf.nn.softplus(raw_scale) + 1e-6
    delay_true = labels['delay']
    res = delay_true - loc
    rmse = tf.sqrt(tf.reduce_mean(tf.square(res)))
    mean_scale = tf.reduce_mean(scale)
    ratio = rmse / (mean_scale + 1e-9)
    return total_loss, delay_loss, recon_loss, sparsity_loss, mean_scale, rmse, ratio, predictions

# ==============================================================================
# 4. 主执行逻辑
# ==============================================================================

def set_global_determinism(seed: int):
    """设置全局随机种子与确定性选项以最大化复现性。

    注意：某些 GPU 算子仍可能存在非确定性实现；如果需要完全确定性，
    可在必要时切换到 CPU 或升级 / 固定特定 TF 与 CUDA 版本。
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # 触发部分算子的确定性实现
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # 新版本 TF (>=2.13) 支持的 API，若不存在则忽略
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:  # pragma: no cover - 兼容旧版本
        pass


def main(args):
    # 在执行任何需要随机性的逻辑之前设置随机种子
    set_global_determinism(args.seed)
    print(f"[SEED] Global seed set to {args.seed} (PYTHON/NumPy/TF). Deterministic ops requested.")
    config = {
        'link_state_dim': 4,
        'path_state_dim': 2,
        'T': 3,
        'readout_units': 8,
        'readout_layers': 2,
        'l2': 0.1,
        'l2_2': 0.01,
        # Dropout 默认开启
        'use_dropout': True,
        'dropout_rate': 0.1,
        # SAE 默认关闭
        'sae_enabled': False,
        'sae_hidden_dim': 64,
        'sae_latent_dim': 16,
        'sae_alpha': 1.0,
        'sae_beta': 1e-4,
        'sae_activation': 'relu',
    }

    # 设置 TensorBoard 日志目录
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.model_dir, 'logs', '{}_{}'.format(args.target, current_time))
    train_log_dir = os.path.join(log_dir, 'train')
    val_log_dir = os.path.join(log_dir, 'validation')
    
    # 创建 TensorBoard 写入器
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    train_files = tf.io.gfile.glob(os.path.join(args.train_dir, '*.tfrecords'))
    train_dataset = create_dataset(train_files, args.batch_size, is_training=True, seed=args.seed)
    print("Found {} training files.".format(len(train_files)))

    eval_files = tf.io.gfile.glob(os.path.join(args.eval_dir, '*.tfrecords'))
    eval_dataset = create_dataset(eval_files, args.batch_size, is_training=False, seed=args.seed)
    print("Found {} evaluation files.".format(len(eval_files)))

    # 覆盖 Dropout 配置
    if args.no_dropout:
        config['use_dropout'] = False
    if args.dropout_rate is not None:
        config['dropout_rate'] = float(args.dropout_rate)

    # 覆盖 SAE 配置（仅 MLP 生效）
    if args.enable_sae:
        config['sae_enabled'] = True
    if args.sae_hidden_dim is not None:
        config['sae_hidden_dim'] = args.sae_hidden_dim
    if args.sae_latent_dim is not None:
        config['sae_latent_dim'] = args.sae_latent_dim
    if args.sae_alpha is not None:
        config['sae_alpha'] = args.sae_alpha
    if args.sae_beta is not None:
        config['sae_beta'] = args.sae_beta
    if args.sae_activation is not None:
        config['sae_activation'] = args.sae_activation

    # 记录异方差延迟损失变体
    config['loss_variant'] = args.loss_variant if hasattr(args, 'loss_variant') else 'standard'
    print(f"[LOSS] Using heteroscedastic delay loss variant: {config['loss_variant']}")

    model, loss_fn = create_model_and_loss_fn(config, args.target)
    
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
            total_loss_t, delay_loss_t, recon_loss_t, sparsity_loss_t, mean_scale_t, rmse_t, ratio_t, _ = train_step(
                model, optimizer, features, labels, loss_fn,
                sae_alpha=tf.constant(config['sae_alpha'], dtype=tf.float32),
                sae_beta=tf.constant(config['sae_beta'], dtype=tf.float32)
            )
            total_train_loss += total_loss_t
            train_step_count += 1
            global_step += 1
            if global_step % 10 == 0:
                with train_summary_writer.as_default():
                    tf.summary.scalar('batch_total_loss', total_loss_t, step=global_step)
                    tf.summary.scalar('batch_delay_loss', delay_loss_t, step=global_step)
                    if model.sae_enabled:
                        tf.summary.scalar('batch_recon_loss', recon_loss_t, step=global_step)
                        tf.summary.scalar('batch_sparsity_loss', sparsity_loss_t, step=global_step)
                    tf.summary.scalar('delay_mean_scale', mean_scale_t, step=global_step)
                    tf.summary.scalar('delay_rmse', rmse_t, step=global_step)
                    tf.summary.scalar('delay_rmse_over_mean_sigma', ratio_t, step=global_step)
                    current_lr = optimizer.learning_rate
                    if hasattr(current_lr, 'numpy'):
                        current_lr = current_lr.numpy()
                    tf.summary.scalar('learning_rate', current_lr, step=global_step)
            pbar.set_postfix({'total': '{:.4f}'.format(float(total_loss_t.numpy())), 'delay': '{:.4f}'.format(float(delay_loss_t.numpy()))})
                
        avg_train_loss = total_train_loss / train_step_count

        # 记录训练的平均损失
        with train_summary_writer.as_default():
            tf.summary.scalar('epoch_total_loss', avg_train_loss, step=epoch + 1)

        # 评估
        total_eval_loss = 0.0
        eval_step_count = 0
        pbar_eval = tqdm(eval_dataset, desc="Evaluating Epoch {}".format(epoch+1))
        
        last_mean_scale = last_rmse = last_ratio = None
        for features, labels in pbar_eval:
            total_loss_t, delay_loss_t, recon_loss_t, sparsity_loss_t, mean_scale_t, rmse_t, ratio_t, _ = eval_step(
                model, features, labels, loss_fn,
                sae_alpha=tf.constant(config['sae_alpha'], dtype=tf.float32),
                sae_beta=tf.constant(config['sae_beta'], dtype=tf.float32)
            )
            total_eval_loss += total_loss_t
            eval_step_count += 1
            pbar_eval.set_postfix({'total': '{:.4f}'.format(float(total_loss_t.numpy())), 'delay':'{:.4f}'.format(float(delay_loss_t.numpy()))})
            last_mean_scale = mean_scale_t
            last_rmse = rmse_t
            last_ratio = ratio_t
            
        avg_eval_loss = total_eval_loss / eval_step_count

        # 记录验证损失
        with val_summary_writer.as_default():
            tf.summary.scalar('epoch_total_loss', avg_eval_loss, step=epoch + 1)
            current_lr = optimizer.learning_rate
            if hasattr(current_lr, 'numpy'):
                current_lr = current_lr.numpy()
            tf.summary.scalar('learning_rate_epoch', current_lr, step=epoch + 1)
            # 记录最终一批的延迟模型诊断指标
            if last_mean_scale is not None:
                tf.summary.scalar('epoch_delay_mean_scale', last_mean_scale, step=epoch + 1)
                tf.summary.scalar('epoch_delay_rmse', last_rmse, step=epoch + 1)
                tf.summary.scalar('epoch_delay_rmse_over_mean_sigma', last_ratio, step=epoch + 1)

        # ================= SAE 统计探针（可选） =================
        if model.sae_enabled and args.sae_probe_batches > 0:
            probe_latents = []
            taken = 0
            for f_probe, l_probe in eval_dataset:
                preds_aux = model(f_probe, training=False, return_aux=True)
                _, aux_probe = preds_aux
                if aux_probe is not None:
                    probe_latents.append(aux_probe['latent'])
                taken += 1
                if taken >= args.sae_probe_batches:
                    break
            if probe_latents:
                lat = tf.concat(probe_latents, axis=0)
                abs_lat = tf.abs(lat)
                threshold = tf.constant(args.sae_sparsity_threshold, dtype=lat.dtype)
                sparsity_ratio = tf.reduce_mean(tf.cast(abs_lat < threshold, tf.float32))
                mean_abs = tf.reduce_mean(abs_lat)
                with val_summary_writer.as_default():
                    tf.summary.scalar('sae_sparsity_ratio', sparsity_ratio, step=epoch + 1)
                    tf.summary.scalar('sae_latent_mean_abs', mean_abs, step=epoch + 1)
                    tf.summary.histogram('sae_latent', lat, step=epoch + 1)
                print(f"[SAE] epoch {epoch+1}: sparsity_ratio={sparsity_ratio:.4f} (|z|<{args.sae_sparsity_threshold}), mean|z|={mean_abs:.4e}")

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
        print("Epoch {} finished. Avg Train Total Loss: {:.4f}, Avg Eval Total Loss: {:.4f}, LR: {:.6f}".format(
            epoch + 1, avg_train_loss, avg_eval_loss,
            optimizer.learning_rate.numpy() if hasattr(optimizer.learning_rate, 'numpy') else optimizer.learning_rate))

        # 保存最佳模型
        if avg_eval_loss < best_eval_loss - early_stopping_min_delta if args.early_stopping else avg_eval_loss < best_eval_loss:
            print("Evaluation loss improved from {:.4f} to {:.4f}. Saving model...".format(
                best_eval_loss, avg_eval_loss))
            best_eval_loss = avg_eval_loss
            
            # 根据是否使用KAN来命名模型文件
            save_path = os.path.join(args.model_dir, "best_{}_model.weights.h5".format(args.target))
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
    print("Model weights saved as: {}".format(
        os.path.join(args.model_dir, "best_{}_model.weights.h5".format(args.target))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RouteNet TF2 Implementation')
    parser.add_argument('--train_dir', type=str,default='./data/routenet/nsfnetbw/tfrecords/train', 
                      help='Directory containing training TFRecord files')
    parser.add_argument('--eval_dir', type=str, default='./data/routenet/nsfnetbw/tfrecords/evaluate',
                      help='Directory containing evaluation TFRecord files') 
    parser.add_argument('--model_dir', type=str, required=True,
                      help='Directory to save model checkpoints and logs')
    parser.add_argument('--target', type=str, choices=['delay', 'drops'], default='delay',
                      help='Training target: "delay" for delay prediction, "drops" for packet drop prediction')
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
    # SAE 联合训练配置（仅 MLP 模式有效）
    parser.add_argument('--enable_sae', action='store_true', help='Enable joint training with Sparse Autoencoder (only MLP)')
    parser.add_argument('--sae_hidden_dim', type=int, default=64, help='Hidden dimension of SAE encoder/decoder')
    parser.add_argument('--sae_latent_dim', type=int, default=16, help='Latent dimension of SAE')
    parser.add_argument('--sae_alpha', type=float, default=1.0, help='Weight α for SAE reconstruction loss')
    parser.add_argument('--sae_beta', type=float, default=1e-4, help='Weight β for SAE sparsity (L1 on latent)')
    parser.add_argument('--sae_activation', type=str, default='relu', choices=['relu','selu','tanh','gelu'], help='Activation for SAE layers')
    parser.add_argument('--sae_sparsity_threshold', type=float, default=1e-3, help='Threshold for counting a latent unit as inactive')
    parser.add_argument('--sae_probe_batches', type=int, default=1, help='Number of eval batches to probe SAE latent each epoch (0 to disable)')
    parser.add_argument('--loss_variant', type=str, choices=['standard','legacy'], default='standard', help='Heteroscedastic delay loss variant: standard (Gaussian NLL) or legacy (original)')
    
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
    
    
    # 用于cosine和polynomial调度的steps_per_epoch估计
    parser.add_argument('--steps_per_epoch', type=int, default=100,
                      help='Estimated steps per epoch for cosine/polynomial schedules')
    # 复现性
    parser.add_argument('--seed', type=int, default=137, help='Global random seed for Python/NumPy/TensorFlow')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
        
    main(args)
