# 神经网络与深度学习作业：CIFAR-10 图像分类

本项目实现了一个多层神经网络/多层感知机（MLP）模型，用于对 CIFAR-10 数据集进行分类。代码包括数据加载、模型定义、训练、测试以及超参数搜索等功能。

---

## 目录结构

```
nnassignment/
├── dataset.py              # 数据加载和预处理
├── model.py                # MLP 模型定义
├── train.py                # 模型训练、评估和可视化
├── hyperparam_search.py    # 超参数搜索
├── main.py                 # 主程序入口，训练和测试模型
```

---

## 文件说明

### 1. dataset.py
- **功能**：负责下载、解压和加载 CIFAR-10 数据集。
- **主要函数**：
  - `download_cifar10(download_dir)`: 下载并解压 CIFAR-10 数据集。
  - `load_cifar10_data(data_dir)`: 加载训练集和测试集，并进行归一化处理。

### 2. model.py
- **功能**：定义多层感知机（MLP）模型。
- **主要类**：
  - `MLP`：
    - `__init__(input_size, hidden_sizes, output_size, l2_lambda, seed)`: 初始化模型参数。
    - `forward(X)`: 前向传播。
    - `backward(cache, y)`: 反向传播，计算梯度。
    - `update_params(grads, lr, optimizer)`: 更新模型参数。
    - `predict(X)`: 预测输入数据的类别。
    - `cross_entropy_loss(A2, y)`: 计算交叉熵损失。

### 3. train.py
- **功能**：实现模型的训练、验证和评估。
- **主要函数**：
  - `train_model(X_train, y_train, X_val, y_val, ...)`：
    - 训练模型，支持保存最佳权重。
    - 返回训练好的模型和训练历史。
  - `load_weights(model, path)`：加载保存的模型权重。
  - `plot_training_history(history)`：绘制训练和验证的损失与准确率曲线。
  - `save_training_plot(history, save_path)`：保存训练曲线图。

### 4. hyperparam_search.py
- **功能**：实现超参数搜索。
- **主要函数**：
  - `hyperparam_search()`：
    - 对隐藏层大小、学习率、批量大小和 L2 正则化系数进行网格搜索。
    - 保存每组超参数的训练历史和最佳权重。

### 5. `main.py`
- **功能**：主程序入口，加载数据、初始化模型、训练模型并测试性能。

---

## 使用说明

### 1. 环境准备
确保安装以下依赖：
- Python 3.8+
- NumPy
- Matplotlib

安装依赖：
```bash
pip install numpy matplotlib
```

### 2. 数据加载
运行以下代码下载并加载 CIFAR-10 数据集：
```python
from dataset import load_cifar10_data

train_data, train_labels, test_data, test_labels = load_cifar10_data("./cifar-10-data")
print("训练集大小:", train_data.shape)
print("测试集大小:", test_data.shape)
```

### 3. 模型训练
在 `main.py` 中调用 `train_model` 函数训练模型：
```python
from model import MLP
from train import train_model

# 初始化模型
model = MLP(input_size=3072, hidden_sizes=[128, 256], output_size=10, l2_lambda=0.001)

# 训练模型
trained_model, history = train_model(
    X_train, y_train, X_val, y_val,
    model=model,
    lr=0.001,
    batch_size=64,
    epochs=15,
    optimizer="adam",
    save_best=True,
    save_path="best_model.npz"
)
```

### 4. 测试模型
加载训练好的模型权重并测试性能：
```python
from train import load_weights

# 加载权重
load_weights(trained_model, "best_model.npz")

# 测试模型
A2_test, _ = trained_model.forward(test_data)
test_accuracy = np.mean(np.argmax(A2_test, axis=1) == test_labels)
print(f"测试集准确率: {test_accuracy:.4f}")
```

### 5. 绘制训练曲线
使用 `plot_training_history` 函数绘制训练和验证的损失与准确率曲线：
```python
from train import plot_training_history

plot_training_history(history)
```

### 6. 超参数搜索
运行 hyperparam_search.py 进行超参数搜索：
```bash
python hyperparam_search.py
```

---

## 注意事项
1. 调整超参数（如隐藏层大小、学习率、批量大小等）以优化模型性能。
2. 如果需要保存训练曲线图，请确保指定有效的保存路径。
