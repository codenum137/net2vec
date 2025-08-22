# net2vec

This repository is a collection of machine learning models for computer networks.
Currently, the following models are implemented:

1. [Message Passing](mpnn) - vanilla Graph Neural Network
1. [RouteNet](routenet) - A new neural architecture designed for neural understanding of routing in the network.
1. **RouteNet TensorFlow 2** - Modern TensorFlow 2.x implementation with TensorBoard support and evaluation tools

**If you decide to apply the concepts presented or base on the provided code, please do refer our related paper**

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
python routenet/routenet_tf2.py \
    --train_dir data/routenet/nsfnetbw/tfrecords/train/ \
    --eval_dir data/routenet/nsfnetbw/tfrecords/evaluate/ \
    --model_dir models/routenet_tf2_model \
    --target delay \
    --epochs 20 \
    --batch_size 32 \
    --learning_rate 0.001
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
python evaluate_routenet.py \
    --model_dir models/routenet_tf2_model \
    --test_dir data/routenet/nsfnetbw/tfrecords/evaluate \
    --output_dir evaluation_results

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

### 🔄 Typical Workflow

1. **Prepare Data**: Ensure TFRecord files are in the correct format
2. **Train Model**: Use `routenet_tf2.py` to train the network
3. **Monitor Progress**: Use TensorBoard to track training
4. **Evaluate Results**: Use `evaluate_routenet.py` to assess performance
5. **Analyze Results**: Review generated plots and metrics

### 💡 Tips and Best Practices

- **Batch Size**: Start with smaller batch sizes (8-16) to reduce memory usage
- **Epochs**: Monitor validation loss to avoid overfitting
- **Data Quality**: Ensure TFRecord files contain all required features
- **GPU Memory**: The model will use GPU automatically if available
- **Retracing Warnings**: Some TF function retracing is normal due to variable graph sizes

