# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import argparse
import os
from tqdm import tqdm
import datetime

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
    def __init__(self, config, output_units=2, final_activation=None):
        super().__init__()
        self.config = config
        
        # 消息传递层
        self.path_update = tf.keras.layers.GRUCell(config['path_state_dim'])
        self.link_update = tf.keras.layers.GRUCell(config['link_state_dim'])

        # 读出网络
        self.readout = tf.keras.Sequential([
            tf.keras.layers.Dense(
                config['readout_units'],
                activation='selu',
                kernel_regularizer=tf.keras.regularizers.l2(config['l2'])
            ) for _ in range(config['readout_layers'])
        ])
        
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

        # T 轮消息传递
        for t in range(self.config['T']):
            # 路径更新：收集每条路径经过的链路状态
            link_gather = tf.gather(link_state, inputs['links'])
            
            # 简化的路径更新 - 使用 segment_sum 来聚合每条路径的链路状态
            # 而不是使用复杂的序列处理
            path_link_agg = tf.math.unsorted_segment_sum(
                link_gather, inputs['paths'], inputs['n_paths']
            )
            
            # 使用 GRU 更新路径状态
            path_state, _ = self.path_update(path_link_agg, [path_state])
            
            # 链路更新：聚合所有经过每条链路的路径信息
            path_messages = tf.gather(path_state, inputs['paths'])
            link_agg_messages = tf.math.unsorted_segment_sum(
                path_messages, inputs['links'], inputs['n_links']
            )
            
            # 更新链路状态
            link_state, _ = self.link_update(link_agg_messages, [link_state])

        # 读出阶段
        readout_output = self.readout(path_state, training=training)
        final_input = tf.concat([readout_output, path_state], axis=1)
        return self.final_layer(final_input)

# ==============================================================================
# 3. 损失函数 (支持延迟和丢包两种任务)
# ==============================================================================

def heteroscedastic_loss(y_true, y_pred):
    """异方差损失函数 - 用于延迟预测"""
    loc = y_pred[:, 0]
    scale = tf.nn.softplus(y_pred[:, 1]) + 1e-9
    
    delay_true = y_true['delay']
    jitter_true = y_true['jitter']
    packets_true = y_true['packets'] 
    drops_true = y_true['drops']
    
    n = packets_true - drops_true
    _2sigma = 2.0 * tf.square(scale)
    
    nll = (n * jitter_true / _2sigma + 
           n * tf.square(delay_true - loc) / _2sigma + 
           n * tf.math.log(scale))
           
    return tf.reduce_sum(nll) / 1e6

def binomial_loss(y_true, y_pred):
    """二项分布损失函数 - 用于丢包预测"""
    # y_pred 是 logits（未经过 sigmoid）
    logits = y_pred[:, 0]
    
    packets_true = y_true['packets']
    drops_true = y_true['drops']
    
    # 计算真实丢包率
    loss_ratio = drops_true / (packets_true + 1e-9)
    
    # 使用二项分布负对数似然
    # 这里使用与原版相同的公式
    predictions = tf.nn.sigmoid(logits)
    
    # 二项分布损失：-sum(log(p^k * (1-p)^(n-k)))
    # 简化为: -sum(k*log(p) + (n-k)*log(1-p))
    loss = -(drops_true * tf.math.log(predictions + 1e-9) + 
             (packets_true - drops_true) * tf.math.log(1 - predictions + 1e-9))
    
    return tf.reduce_sum(loss) / 1e6

def create_model_and_loss_fn(config, target):
    """根据target参数创建相应的模型和损失函数"""
    if target == 'delay':
        # 延迟预测模型
        model = RouteNet(config, output_units=2, final_activation=None)
        loss_fn = heteroscedastic_loss
        print("Created delay prediction model with heteroscedastic loss")
    elif target == 'drops':
        # 丢包预测模型
        model = RouteNet(config, output_units=1, final_activation=None)
        loss_fn = binomial_loss
        print("Created drop prediction model with binomial loss")
    else:
        raise ValueError("Unsupported target: {}. Choose 'delay' or 'drops'".format(target))
    
    return model, loss_fn

@tf.function(reduce_retracing=True)
def train_step(model, optimizer, features, labels, loss_fn):
    with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        loss = loss_fn(labels, predictions)
        loss += sum(model.losses)
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, predictions

@tf.function(reduce_retracing=True)
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

    # 设置 TensorBoard 日志目录，包含target信息
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.model_dir, 'logs', '{}_{}'.format(args.target, current_time))
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
    
    for epoch in range(args.epochs):
        print("\nEpoch {}/{}".format(epoch + 1, args.epochs))
        
        # 训练
        total_train_loss = 0.0
        train_step_count = 0
        pbar = tqdm(train_dataset, desc="Training Epoch {}".format(epoch+1))
        
        for features, labels in pbar:
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

        print("Epoch {} finished. Avg Train Loss: {:.4f}, Avg Eval Loss: {:.4f}, LR: {:.6f}".format(
            epoch + 1, avg_train_loss, avg_eval_loss, 
            optimizer.learning_rate.numpy() if hasattr(optimizer.learning_rate, 'numpy') else optimizer.learning_rate))

        # 保存最佳模型
        if avg_eval_loss < best_eval_loss:
            print("Evaluation loss improved from {:.4f} to {:.4f}. Saving model...".format(
                best_eval_loss, avg_eval_loss))
            best_eval_loss = avg_eval_loss
            save_path = os.path.join(args.model_dir, "best_{}_model.weights.h5".format(args.target))
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
    print("Model weights saved as: {}".format(
        os.path.join(args.model_dir, "best_{}_model.weights.h5".format(args.target))))

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
    
    # 用于cosine和polynomial调度的steps_per_epoch估计
    parser.add_argument('--steps_per_epoch', type=int, default=100,
                      help='Estimated steps per epoch for cosine/polynomial schedules')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
        
    main(args)
