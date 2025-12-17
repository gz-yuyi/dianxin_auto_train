import argparse
import pickle
import warnings
import numpy as np
import pandas as pd
import os
import torch
import random
from transformers import BertTokenizer
from torch import nn
from transformers import BertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--bert-path", default="bert-base-chinese")
parser.add_argument("--model-filename", default="wenchong.pt")
parser.add_argument("--save-path", default=r"C:\Users\admin\Desktop\蚊虫")
parser.add_argument("--data-filename", default=r"C:\Users\admin\Desktop\蚊虫\登革热.xlsx")
parser.add_argument("--sheetname", default="原始数据")
parser.add_argument("--column-name", default="内容合并")
parser.add_argument("--label-name", default="标签")
args = parser.parse_args()

# 下载的预训练文件路径
BERT_PATH = args.bert_path
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
#忽视警告
warnings.filterwarnings("ignore", message="1Torch was not compiled with flash attention")

#存储映射
LABEL2ID = {}
ID2LABEL = {}

#文件路径
model_filename = args.model_filename
save_path = args.save_path  #文件夹
filename = os.path.join(save_path, args.data_filename)      #工作表
sheetname = args.sheetname         # 工作表名
column_name = args.column_name      #文本列
label_name = args.label_name        #标签列

# 定义一个函数来读取文本文件并返回DataFrame
def read_text_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            label, text = line.strip().split(' ', 1)  # 分割标签和文本
            data.append((LABEL2ID[label], text))
    df = pd.DataFrame(data, columns=[label_name, column_name])
    return df

def dataprocess(filename, sheetname, text_column, label_column, test_size=0.2, random_state=42):
    # 读取Excel文件的指定表单和列
    file_path = filename
    sheet_name = sheetname

    # 读取文本数据和标签数据
    df_text = pd.read_excel(file_path, sheet_name=sheet_name, usecols=[text_column], skiprows=0)
    df_label = pd.read_excel(file_path, sheet_name=sheet_name, usecols=[label_column], skiprows=0)

    # 假设列名分别为'text'和'label'，如果列名不同，请相应修改
    df_text.columns = [column_name]
    df_label.columns = [label_name]

    df = pd.concat([df_text, df_label], axis=1)

    # 确保“文本内容”列是字符串类型，并将空值填充为空字符串
    df[column_name] = df[column_name].astype(str)
    df[column_name] = df[column_name].fillna('')

    # 自动创建标签到ID的映射
    labels = df[label_name].unique()
    label_number=len(labels)
    LABEL2ID = {label: idx for idx, label in enumerate(labels)}
    ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
    df[label_name] = df[label_name].map(LABEL2ID)
    return df, LABEL2ID, ID2LABEL,label_number

# 调用函数读取数据
data_df,LABEL2ID,ID2LABEL,out_features = dataprocess(filename,sheetname,column_name,label_name)

# 保存到文件
filename = os.path.join(save_path, f'label_mappings_{model_filename}.pkl')
with open(filename, 'wb') as f:
    pickle.dump((LABEL2ID, ID2LABEL), f)


# 划分数据集，80%用于训练，10%用于验证，10%用于测试
train_df, temp_df = train_test_split(data_df, test_size=0.2, stratify=None, random_state=42)
dev_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=None, random_state=42)


class MyDataset(Dataset):
    def __init__(self, df):
        # tokenizer分词后可以被自动汇聚
        self.texts = [tokenizer(text,
                                padding='max_length',  # 填充到最大长度
                                max_length=512,  # 经过数据分析，最大长度为35
                                truncation=True,
                                return_tensors="pt")
                      for text in df[column_name]]
        # Dataset会自动返回Tensor
        self.labels = [label for label in df[label_name]]

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


class BertClassifier(nn.Module):
    def __init__(self,out_features):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_PATH)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(768, out_features)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer

train_dataset = MyDataset(train_df)
dev_dataset = MyDataset(dev_df)
test_dataset = MyDataset(test_df)

# 训练参数
lr = 3e-5
epoch = 5
batch_size = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
random_seed = 1999

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(random_seed)


def save_model(save_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, save_name))


# 定义模型
model = BertClassifier(out_features)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=lr)
model = model.to(device)
criterion = criterion.to(device)

# 构建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size)

# 训练
best_dev_acc = 0
for epoch_num in range(epoch):
    total_acc_train = 0
    total_loss_train = 0
    for inputs, labels in tqdm(train_loader,desc=f"Epoch {epoch_num + 1}/{epoch}"):
        input_ids = inputs['input_ids'].squeeze(1).to(device)  # torch.Size([32, 35])
        masks = inputs['attention_mask'].squeeze(1).to(device)  # torch.Size([32, 1, 35])
        labels = labels.to(device)
        output = model(input_ids, masks)

        batch_loss = criterion(output, labels)
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        acc = (output.argmax(dim=1) == labels).sum().item()
        total_acc_train += acc
        total_loss_train += batch_loss.item()

    # ----------- 验证模型 -----------
    model.eval()
    total_acc_val = 0
    total_loss_val = 0
    # 不需要计算梯度
    with torch.no_grad():
        # 循环获取数据集，并用训练好的模型进行验证
        for inputs, labels in dev_loader:
            input_ids = inputs['input_ids'].squeeze(1).to(device)  # torch.Size([32, 35])
            masks = inputs['attention_mask'].squeeze(1).to(device)  # torch.Size([32, 1, 35])
            labels = labels.to(device)
            output = model(input_ids, masks)

            batch_loss = criterion(output, labels)
            acc = (output.argmax(dim=1) == labels).sum().item()
            total_acc_val += acc
            total_loss_val += batch_loss.item()

        print(f'''Epochs: {epoch_num + 1} 
          | Train Loss: {total_loss_train / len(train_dataset): .3f} 
          | Train Accuracy: {total_acc_train / len(train_dataset): .3f} 
          | Val Loss: {total_loss_val / len(dev_dataset): .3f} 
          | Val Accuracy: {total_acc_val / len(dev_dataset): .3f}''')

        # 保存最优的模型
        if total_acc_val / len(dev_dataset) > best_dev_acc:
            best_dev_acc = total_acc_val / len(dev_dataset)
            save_model(model_filename)

    model.train()

# 保存最后的模型，以便继续训练
#save_model('pingtailast.pt')
