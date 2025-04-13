import numpy as np
import pickle
import os
import urllib.request
import tarfile


# 下载并解压CIFAR-10数据集
def download_cifar10(download_dir="cifar-10-data"):
    os.makedirs(download_dir, exist_ok=True)
    filename = os.path.join(download_dir, "cifar-10-python.tar.gz")
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

    if not os.path.exists(filename):
        print("Downloading CIFAR-10 dataset...")
        urllib.request.urlretrieve(url, filename)

    extracted_dir = os.path.join(download_dir, "cifar-10-batches-py")
    if not os.path.exists(extracted_dir):
        print("Extracting files...")
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall(path=download_dir)

    return extracted_dir

# 加载数据集
def load_cifar10_data(data_dir="cifar-10-data"):
    extracted_dir = os.path.join(data_dir, "cifar-10-batches-py")
    if not os.path.exists(extracted_dir):
        extracted_dir = download_cifar10(data_dir)

    def unpickle(file_path):
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f, encoding='latin1')
        return data_dict

    # 加载训练数据
    def load_data_batches():
        data_batches = []
        labels = []
        for i in range(1, 6):
            batch_path = os.path.join(extracted_dir, f'data_batch_{i}')
            batch = unpickle(batch_path)
            data_batches.append(batch['data'])
            labels += batch['labels']
        data = np.concatenate(data_batches)
        labels = np.array(labels)
        return data, labels

    # 加载测试数据
    def load_test_batch():
        test_path = os.path.join(extracted_dir, 'test_batch')
        test_batch = unpickle(test_path)
        test_data = test_batch['data']
        test_labels = np.array(test_batch['labels'])
        return test_data, test_labels

    train_data, train_labels = load_data_batches()
    test_data, test_labels = load_test_batch()

    # 归一化
    train_data = train_data.astype(np.float32) / 255.0
    test_data = test_data.astype(np.float32) / 255.0

    return train_data, train_labels, test_data, test_labels