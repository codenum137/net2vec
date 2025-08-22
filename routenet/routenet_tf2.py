# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import argparse
import os
from tqdm import tqdm
import datetime

# ==============================================================================
# 1. 数据加载与预处理 (修正版)
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
# 2. RouteNet 模型 (简化版本，修复维度问题)
# ==============================================================================

class RouteNet(tf.keras.Model):
    def __init__(self, config, output_units=2):
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
        
        self.final_layer = tf.keras.layers.Dense(
            output_units,
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
# 3. 损失函数 
# ==============================================================================

def heteroscedastic_loss(y_true, y_pred):
    """与原版完全一致的异方差损失函数"""
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

@tf.function
def train_step(model, optimizer, features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        loss = heteroscedastic_loss(labels, predictions)
        loss += sum(model.losses)
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

@tf.function  
def eval_step(model, features, labels):
    predictions = model(features, training=False)
    loss = heteroscedastic_loss(labels, predictions)
    loss += sum(model.losses)
    return loss

# ==============================================================================
# 4. 主执行逻辑 (无变化)
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

    # 设置 TensorBoard 日志目录
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.model_dir, 'logs', current_time)
    train_log_dir = os.path.join(log_dir, 'train')
    val_log_dir = os.path.join(log_dir, 'validation')
    
    # 创建 TensorBoard 写入器
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    train_files = tf.io.gfile.glob(os.path.join(args.train_dir, '*.tfrecords'))
    train_dataset = create_dataset(train_files, args.batch_size, is_training=True)
    print(f"Found {len(train_files)} training files.")

    eval_files = tf.io.gfile.glob(os.path.join(args.eval_dir, '*.tfrecords'))
    eval_dataset = create_dataset(eval_files, args.batch_size, is_training=False)
    print(f"Found {len(eval_files)} evaluation files.")

    model = RouteNet(config, output_units=2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    print("Starting training...")
    print(f"TensorBoard logs will be saved to: {log_dir}")
    print(f"Run 'tensorboard --logdir {log_dir}' to view training progress")
    
    best_eval_loss = float('inf')
    global_step = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # 训练
        total_train_loss = 0.0
        train_step_count = 0
        pbar = tqdm(train_dataset, desc=f"Training Epoch {epoch+1}")
        
        for features, labels in pbar:
            loss = train_step(model, optimizer, features, labels)
            total_train_loss += loss
            train_step_count += 1
            global_step += 1
            
            # 记录每个批次的损失到 TensorBoard (每10步记录一次，避免日志过多)
            if global_step % 10 == 0:
                with train_summary_writer.as_default():
                    tf.summary.scalar('batch_loss', loss, step=global_step)
                    tf.summary.scalar('learning_rate', optimizer.learning_rate, step=global_step)
            
            pbar.set_postfix({'loss': f'{loss:.4f}', 'step': global_step})
        avg_train_loss = total_train_loss / train_step_count

        # 记录训练的平均损失
        with train_summary_writer.as_default():
            tf.summary.scalar('epoch_loss', avg_train_loss, step=epoch + 1)

        # 评估
        total_eval_loss = 0.0
        eval_step_count = 0
        pbar_eval = tqdm(eval_dataset, desc=f"Evaluating Epoch {epoch+1}")
        
        for features, labels in pbar_eval:
            loss = eval_step(model, features, labels)
            total_eval_loss += loss
            eval_step_count += 1
            pbar_eval.set_postfix({'loss': f'{loss:.4f}'})
            
        avg_eval_loss = total_eval_loss / eval_step_count

        # 记录验证损失
        with val_summary_writer.as_default():
            tf.summary.scalar('epoch_loss', avg_eval_loss, step=epoch + 1)

        print(f"Epoch {epoch + 1} finished. Avg Train Loss: {avg_train_loss:.4f}, Avg Eval Loss: {avg_eval_loss:.4f}")

        # 保存最佳模型
        if avg_eval_loss < best_eval_loss:
            print(f"Evaluation loss improved from {best_eval_loss:.4f} to {avg_eval_loss:.4f}. Saving model...")
            best_eval_loss = avg_eval_loss
            save_path = os.path.join(args.model_dir, "best_model.weights.h5")
            model.save_weights(save_path)
            
            # 记录最佳模型的信息
            with val_summary_writer.as_default():
                tf.summary.scalar('best_loss', best_eval_loss, step=epoch + 1)
        else:
            print(f"Evaluation loss did not improve from {best_eval_loss:.4f}.")
    
    # 训练结束后，关闭 summary writers
    train_summary_writer.close()
    val_summary_writer.close()
    print(f"\nTraining completed! TensorBoard logs saved to: {log_dir}")
    print(f"To view the results, run: tensorboard --logdir {log_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RouteNet TF2 Implementation')
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--eval_dir', type=str, required=True) 
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
        
    main(args)