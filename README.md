# net2vec

This repository is a collection of machine learning models for computer networks with comprehensive training automation and experiment management.

Currently, the following models are implemented:

1. [Message Passing](mpnn) - vanilla Graph Neural Network
1. [RouteNet](routenet) - A new neural architecture designed for neural understanding of routing in the network.
1. **RouteNet TensorFlow 2** - Modern TensorFlow 2.x implementation with physics constraints, KAN architectures, and automated training

## 🌟 Key Features

- **🧠 Multiple Architectures**: Traditional MLP vs Kolmogorov-Arnold Networks (KAN)
- **⚡ Physics Constraints**: Soft and hard physics constraint implementations  
- **🤖 Automated Training**: Systematic training across 14 different model configurations
- **📊 Comprehensive Evaluation**: Automated experiment management and comparison
- **📈 Real-time Monitoring**: Live training progress and TensorBoard integration
- **🔧 Modern TF2**: Keras APIs, eager execution, and best practices

**If you decide to apply the concepts presented or base on the provided code, please do refer our related paper**

## 🚀 Quick Start

**For automated training of all model configurations:**
```bash
# Install dependencies
pip install tensorflow tqdm numpy matplotlib seaborn

# Train all 14 model configurations (MLP/KAN × No/Soft/Hard physics)
python train_models.py

# Models will be saved to fixed_model/ with descriptive names
```

**For single model training:**
```bash
# Train a specific RouteNet configuration
python routenet/routenet_tf2.py \
    --train_dir data/routenet/nsfnetbw/tfrecords/train/ \
    --eval_dir data/routenet/nsfnetbw/tfrecords/evaluate/ \
    --model_dir models/custom_model \
    --target delay \
    --epochs 20
```

## RouteNet TensorFlow 2 Implementation

### 📋 Prerequisites

```bash
# Create conda environment
conda create --name routenet-tf2-env python=3.9 -y
conda activate routenet-tf2-env

# Install dependencies
pip install tensorflow tqdm numpy matplotlib seaborn
```

### 🚀 Training the Model

Use `routenet_tf2.py` to train the RouteNet model with modern TensorFlow 2.x:

```bash
python routenet/routenet_tf2.py     --train_dir data/routenet/nsfnetbw/tfrecords/train/     --eval_dir data/routenet/nsfnetbw/tfrecords/evaluate/     --model_dir models/routenet_tf2_model     --target drops     --epochs 20     --batch_size 32  --lr_schedule plateau   --learning_rate 0.001 --plateau_patience 5 --plateau_factor 0.5
```

**Training Parameters:**
- `--train_dir`: Directory containing training TFRecord files
- `--eval_dir`: Directory containing evaluation TFRecord files  
- `--model_dir`: Directory to save model checkpoints and logs
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size for training (default: 16)
- `--learning_rate`: Learning rate for optimization (default: 0.001)

**Features:**
- ✅ **TensorBoard Integration**: Automatic logging of training/validation losses
- ✅ **Best Model Saving**: Automatically saves the best performing model
- ✅ **Modern TF2 APIs**: Uses Keras, `@tf.function`, and eager execution
- ✅ **Progress Tracking**: Real-time training progress with tqdm

### 📊 Monitoring Training Progress

The training script automatically creates TensorBoard logs. To visualize training progress:

```bash
# View current training logs
tensorboard --logdir models/routenet_tf2_model/logs/

# Or specify a particular training run
tensorboard --logdir models/routenet_tf2_model/logs/20250822-030524
```

Then open http://localhost:6006 in your browser to see:
- Training and validation loss curves
- Learning rate schedules
- Batch-level loss tracking
- Best model performance tracking

### 📈 Model Evaluation and Visualization

Use `evaluate_routenet.py` to load trained models and generate comprehensive analysis:

```bash
# Basic evaluation
python evaluate_routenet.py --delay_model_dir models/routenet_tf2_model/nsfnetbw_delay --drops_model_dir models/routenet_tf2_model/nsfnetbw_drops --nsfnet_test_dir data/routenet/nsfnetbw/tfrecords/evaluate --gbn_test_dir data/routenet/gbnbw/tfrecords/evaluate --output_dir evaluation_results --num_samples 1000

# Quick evaluation with limited samples
python evaluate_routenet.py \
    --model_dir models/routenet_tf2_model \
    --test_dir data/routenet/nsfnetbw/tfrecords/evaluate \
    --output_dir evaluation_results \
    --num_samples 1000 \
    --batch_size 16
```

**Evaluation Parameters:**
- `--model_dir`: Directory containing trained model weights
- `--test_dir`: Directory containing test TFRecord files
- `--output_dir`: Directory to save evaluation results (default: evaluation_results)
- `--num_samples`: Number of samples to evaluate (None for all)
- `--batch_size`: Batch size for evaluation (default: 32)

**Generated Outputs:**
1. **`relative_error_cdf.png`** - Cumulative Distribution Function of relative errors
2. **`detailed_analysis.png`** - Scatter plots and error distributions
3. **Console metrics** - MAE, RMSE, MAPE, and statistical summaries

### 📊 Evaluation Metrics

The evaluation script provides comprehensive metrics:

- **MAE (Mean Absolute Error)**: Average absolute prediction error
- **RMSE (Root Mean Square Error)**: Square root of mean squared errors
- **MAPE (Mean Absolute Percentage Error)**: Average relative error percentage
- **Relative Error Statistics**: Mean, standard deviation, and 95th percentile

### 🔄 Typical Workflows

#### **Option 1: Single Model Training**
1. **Prepare Data**: Ensure TFRecord files are in the correct format
2. **Train Model**: Use `routenet_tf2.py` to train a specific configuration
3. **Monitor Progress**: Use TensorBoard to track training
4. **Evaluate Results**: Use `evaluate_routenet.py` to assess performance

#### **Option 2: Comprehensive Model Comparison**
1. **Prepare Data**: Ensure TFRecord files are in the correct format
2. **Automated Training**: Use `train_models.py` to train all 14 model configurations
3. **Monitor Progress**: Real-time epoch progress for each model
4. **Systematic Evaluation**: Use experiment automation for comprehensive comparison
5. **Analysis**: Review comparative results across architectures and constraints

#### **Option 3: Custom Experiment Design**
1. **Configure Experiments**: Modify `experiment_config.yaml` for specific comparisons
2. **Selective Training**: Train specific model subsets using `train_models.py`
3. **Automated Evaluation**: Run experiment automation for systematic analysis
4. **Results Analysis**: Generate comparative reports and visualizations

## 🤖 Automated Training System

### Systematic Model Training with `train_models.py`

The `train_models.py` script provides automated training for comprehensive model comparison across different architectures and physics constraints:

```bash
python train_models.py
```

**Training Configurations (14 models total):**

1. **MLP-based models** (7 configurations):
   - No physics constraints with λ ∈ {0}
   - Soft physics constraints with λ ∈ {0.001, 0.01, 0.1}
   - Hard physics constraints with λ ∈ {0.001, 0.01, 0.1}

2. **KAN-based models** (7 configurations):
   - No physics constraints with λ ∈ {0}
   - Soft physics constraints with λ ∈ {0.001, 0.01, 0.1}
   - Hard physics constraints with λ ∈ {0.001, 0.01, 0.1}

**Features:**
- ✅ **Epoch Progress Tracking**: Shows current epoch and percentage progress only
- ✅ **Detailed Logging**: Full training output saved to individual log files
- ✅ **Organized Storage**: Models saved to `fixed_model/` with descriptive names
- ✅ **Complete Coverage**: Tests all architecture and constraint combinations
- ✅ **Robust Training**: Handles interruptions and errors gracefully

**Model Directory Structure:**
```
fixed_model/
├── mlp_no_physics_lambda_0/
│   ├── training.log              # 完整训练日志
│   ├── best_delay_model.weights.h5
│   └── logs/                     # TensorBoard日志
├── mlp_soft_physics_lambda_0.001/
│   ├── training.log
│   ├── best_delay_model.weights.h5
│   └── logs/
├── mlp_soft_physics_lambda_0.01/
├── mlp_soft_physics_lambda_0.1/
├── mlp_hard_physics_lambda_0.001/
├── mlp_hard_physics_lambda_0.01/
├── mlp_hard_physics_lambda_0.1/
├── kan_no_physics_lambda_0/
├── kan_soft_physics_lambda_0.001/
├── kan_soft_physics_lambda_0.01/
├── kan_soft_physics_lambda_0.1/
├── kan_hard_physics_lambda_0.001/
├── kan_hard_physics_lambda_0.01/
└── kan_hard_physics_lambda_0.1/
```

### Training Logs

Each model generates a detailed `training.log` file containing:

- **📋 Training Configuration**: Model type, physics constraints, parameters
- **⚡ Execution Command**: Full command used for training
- **📊 Complete Output**: All TensorFlow/Keras training output
- **📈 Progress Markers**: Epoch progress with timestamps
- **✅ Final Results**: Success/failure status and timing
- **🔍 Error Details**: Exception information if training fails

**Log File Format:**
```log
================================================================================
训练开始时间: 2025-09-10 12:14:56
模型配置: mlp_soft_physics_lambda_0.001
执行命令: python routenet/routenet_tf2.py --train_dir ...
================================================================================

Loading training data...
Epoch 1/20
100/100 [==============================] - 15s - loss: 0.1234 - val_loss: 0.0987

[PROGRESS] 📈 训练进度: Epoch 1/20 (5.0%)
...

[SUCCESS] 训练成功完成!
```

### Physics Constraints Explained

**Physics Constraint Types:**
- **No Physics**: Traditional neural network training (λ = 0)
- **Soft Physics**: Batch-averaged gradient constraints (gradual enforcement)
- **Hard Physics**: Per-sample gradient constraints (strict enforcement)

**Lambda Values:**
- `λ = 0.001`: Weak constraint influence
- `λ = 0.01`: Moderate constraint influence  
- `λ = 0.1`: Strong constraint influence

## 🔬 Experiment Automation with YAML Configuration

### Automated Evaluation with `experiment_config.yaml`

The experiment automation system enables systematic model evaluation across multiple configurations:

```bash
# Run automated experiment evaluation
python run_experiments.py experiment_config.yaml
```

**Configuration Structure:**
```yaml
experiments:
  mlp_soft_0001:
    model_dir: "fixed_model/mlp_soft_physics_lambda_0.001"
    architecture: "mlp"
    physics_constraint: "soft"
    lambda_value: 0.001
    
  kan_hard_01:
    model_dir: "fixed_model/kan_hard_physics_lambda_0.1" 
    architecture: "kan"
    physics_constraint: "hard"
    lambda_value: 0.1
```

**Evaluation Outputs:**
- Performance comparison across all model variants
- Statistical significance testing
- Comprehensive visualization of results
- Automated report generation

### 💡 Tips and Best Practices

- **Training Duration**: Each model trains for 20 epochs, total ~4-6 hours for all 14 models
- **Batch Size**: Start with smaller batch sizes (8-16) to reduce memory usage
- **Epochs**: Monitor validation loss to avoid overfitting
- **Data Quality**: Ensure TFRecord files contain all required features
- **GPU Memory**: The model will use GPU automatically if available
- **Retracing Warnings**: Some TF function retracing is normal due to variable graph sizes
- **Model Comparison**: Use experiment automation for systematic performance analysis
- **Physics Constraints**: Start with soft constraints (easier convergence) before hard constraints

