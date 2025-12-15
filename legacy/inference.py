import datetime
import pickle
import warnings

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

# 忽略特定警告
warnings.filterwarnings(
    "ignore", message="1Torch was not compiled with flash attention"
)

# 全局配置
BERT_PATH = "bert-base-chinese"
SAVE_PATH = "/path/to/model"  # 模型保存路径
MODEL_FILENAME = "yangyan_testwwm.pt"  # 模型文件名

# 加载标签映射
with open(f"{SAVE_PATH}\\{MODEL_FILENAME}.pkl", "rb") as f:
    LABEL2ID, ID2LABEL = pickle.load(f)

# 加载分词器
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BertClassifier(nn.Module):
    """BERT 分类器模型"""

    def __init__(self, out_features):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_PATH)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(768, out_features)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False
        )
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer


def load_model(model_path, out_features):
    """加载预训练模型"""
    model = BertClassifier(out_features)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model


def predict(text, model, top_n):
    """预测文本的类别及其概率"""
    bert_input = tokenizer(
        text, padding="max_length", max_length=512, truncation=True, return_tensors="pt"
    )
    input_ids = bert_input["input_ids"].to(device)
    masks = bert_input["attention_mask"].to(device)

    # 运行模型获取输出
    output = model(input_ids, masks)
    probs = F.softmax(output, dim=1)
    probabilities = probs[0].cpu().detach().numpy()

    # 获取所有类别标签
    all_labels = list(ID2LABEL.values())

    # 创建类别概率字典
    label_prob_dict = dict(zip(all_labels, probabilities))

    # 找出概率最大的前 top_n 个标签及其概率
    top_n_labels = sorted(
        label_prob_dict.items(), key=lambda item: item[1], reverse=True
    )[:top_n]

    # 返回预测结果和 top_n 标签
    pred = output.argmax(dim=1).cpu().item()
    return ID2LABEL[pred], top_n_labels


def write_predictions_to_dataframe(df, text_column, predictions, top_n):
    """将预测结果写入 DataFrame"""
    labels, top_ns = zip(*predictions)  # 解包预测结果和 top_n 信息

    # 提取 top_n 标签和概率
    top_n_labels = [[item[0] for item in t] for t in top_ns]
    top_n_probs = [[item[1] for item in t] for t in top_ns]

    # 创建新的列名
    col_names = [
        f"Top{i + 1}_{field}"
        for i in range(top_n)
        for field in ["Label", "Probability"]
    ]

    # 将 top_n 信息添加到 DataFrame 中
    for i, col_name in enumerate(col_names):
        df[col_name] = [
            top_n_labels[j][i // 2] if i % 2 == 0 else top_n_probs[j][i // 2]
            for j in range(len(top_ns))
        ]

    # 添加预测结果列
    df["Prediction"] = labels
    return df


def judgement(label_prob_dict, topic, get_number=2, threshold=0.4):
    """判断文本是否属于指定类别"""
    # 获取前 get_number 个标签及其概率
    top = sorted(label_prob_dict.items(), key=lambda item: item[1], reverse=True)[
        :get_number
    ]

    # 判断 topic 是否在 top 中且其概率是否超过阈值
    for label, prob in top:
        if label == topic and prob >= threshold:
            return "属于"
    return "不属于"


def judge_predict(text, model, topic):
    """判断文本是否属于指定类别"""
    bert_input = tokenizer(
        text, padding="max_length", max_length=512, truncation=True, return_tensors="pt"
    )
    input_ids = bert_input["input_ids"].to(device)
    masks = bert_input["attention_mask"].to(device)

    # 运行模型获取输出
    output = model(input_ids, masks)
    probs = F.softmax(output, dim=1)
    probabilities = probs[0].cpu().detach().numpy()

    # 获取所有类别标签
    all_labels = list(ID2LABEL.values())

    # 创建类别概率字典
    label_prob_dict = dict(zip(all_labels, probabilities))

    # 判断是否属于指定类别
    return judgement(label_prob_dict, topic)


def multiple_judgement(
    excel_name, sheet_name, label_column_name, text_column_name, model
):
    """多标签判断"""
    df = pd.read_excel(excel_name, sheet_name=sheet_name)
    columns_to_load = [col for col in df.columns if col.startswith(label_column_name)]
    labels = {col: df[col].tolist() for col in columns_to_load}
    texts = df[text_column_name].tolist()

    result = []
    for index in tqdm(range(len(texts)), desc="完成进度："):
        topics = [labels[col][index] for col in columns_to_load]
        bert_input = tokenizer(
            texts[index],
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = bert_input["input_ids"].to(device)
        masks = bert_input["attention_mask"].to(device)

        # 运行模型获取输出
        output = model(input_ids, masks)
        probs = F.softmax(output, dim=1)
        probabilities = probs[0].cpu().detach().numpy()

        # 获取所有类别标签
        all_labels = list(ID2LABEL.values())

        # 创建类别概率字典
        label_prob_dict = dict(zip(all_labels, probabilities))

        text_result_unit = []
        for topic in topics:
            result_unit = judgement(label_prob_dict, topic)
            text_result_unit.append(result_unit)
        result.append(text_result_unit)

    # 转置结果并保存到新 DataFrame
    transposed_result = [list(row) for row in zip(*result)]
    new_df = pd.DataFrame()
    new_df[text_column_name] = df[text_column_name]
    for index, label in enumerate(columns_to_load):
        new_df[label] = labels[label]
        new_df[f"result{index}"] = transposed_result[index]

    # 保存结果到文件
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"multiple_judgements_{timestamp}.xlsx"
    new_df.to_excel(output_filename, index=False)
    print(f"多标签判断结果已保存到 {output_filename}")


def input_with_validation(prompt, validation_func, error_message):
    """带验证的输入函数"""
    while True:
        user_input = input(prompt)
        if validation_func(user_input):
            return user_input
        print(error_message)


def validate_excel_file(file_path):
    """验证 Excel 文件是否存在"""
    try:
        pd.read_excel(file_path)
        return True
    except FileNotFoundError:
        return False


def validate_sheet_name(file_path, sheet_name):
    """验证工作表名是否存在"""
    try:
        pd.read_excel(file_path, sheet_name=sheet_name)
        return True
    except ValueError:
        return False


def validate_column_name(file_path, sheet_name, column_name):
    """验证列名是否存在"""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        return column_name in df.columns
    except Exception:
        return False


def main():
    """主程序"""
    # 加载模型
    model = load_model(f"{SAVE_PATH}\\{MODEL_FILENAME}", out_features=len(LABEL2ID))

    # 用户选择功能
    print("\n请选择功能:")
    print("1. 多选一分类模型")
    print("2. 是否判断模型单标签")
    print("3. 是否判断模型多标签")
    choice = input("请输入数字以选择: ")

    if choice == "1":
        # 多选一分类模型
        top_n = 3  # 用户可以在这里修改 top_n 的值
        print(f"将返回前 {top_n} 个选项")

        # 读取数据
        excel_name = input_with_validation(
            "请输入表格名：", validate_excel_file, "文件不存在，请重新输入！"
        )
        sheet_name = input_with_validation(
            "请输入工作表名（例如 Sheet1）：",
            lambda x: validate_sheet_name(excel_name, x),
            "工作表不存在，请重新输入！",
        )
        text_column = input_with_validation(
            "请输入文本列列名：",
            lambda x: validate_column_name(excel_name, sheet_name, x),
            "列名不存在，请重新输入！",
        )

        # 预测并保存结果
        df = pd.read_excel(excel_name, sheet_name=sheet_name)
        texts = df[text_column].fillna("").tolist()
        predictions = [
            predict(text, model, top_n=top_n) for text in tqdm(texts, desc="预测进度：")
        ]
        updated_df = write_predictions_to_dataframe(
            df, text_column, predictions, top_n=top_n
        )

        # 保存结果到文件
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = (
            f"多选一分类模型_{MODEL_FILENAME.split('.')[0]}_{timestamp}.xlsx"
        )
        updated_df.to_excel(output_filename, index=False)
        print(f"预测结果已保存到 {output_filename}")

    elif choice == "2":
        # 判断模型
        excel_name = input_with_validation(
            "请输入表格名：", validate_excel_file, "文件不存在，请重新输入！"
        )
        sheet_name = input_with_validation(
            "请输入工作表名（例如 Sheet1）：",
            lambda x: validate_sheet_name(excel_name, x),
            "工作表不存在，请重新输入！",
        )
        text_column = input_with_validation(
            "请输入文本列列名：",
            lambda x: validate_column_name(excel_name, sheet_name, x),
            "列名不存在，请重新输入！",
        )
        label_column = input_with_validation(
            "请输入待判断的列名：",
            lambda x: validate_column_name(excel_name, sheet_name, x),
            "列名不存在，请重新输入！",
        )

        # 判断并保存结果
        df = pd.read_excel(excel_name, sheet_name=sheet_name)
        texts = df[text_column].fillna("").tolist()
        topics = df[label_column].tolist()
        predictions = [
            judge_predict(text, model, topic)
            for text, topic in tqdm(zip(texts, topics), desc="判断进度：")
        ]
        df["Prediction"] = predictions

        # 保存结果到文件
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = (
            f"是否判断模型单标签_{MODEL_FILENAME.split('.')[0]}_{timestamp}.xlsx"
        )
        df.to_excel(output_filename, index=False)
        print(f"判断结果已保存到 {output_filename}")

    elif choice == "3":
        # 输出多标签判断模型
        excel_name = input_with_validation(
            "请输入表格名：", validate_excel_file, "文件名错误或不存在，请重新输入！"
        )
        sheet_name = input_with_validation(
            "请输入工作表名（例如 Sheet1）：",
            lambda x: validate_sheet_name(excel_name, x),
            "工作表不存在或表名错误，请重新输入！",
        )
        label_column = input_with_validation(
            "请输入待判断的文本列名：",
            lambda x: validate_column_name(excel_name, sheet_name, x),
            "列名错误或不存在，请重新输入！",
        )
        text_column = input_with_validation(
            "请输入标签列名：",
            lambda x: validate_column_name(excel_name, sheet_name, x),
            "列名错误或不存在，请重新输入！",
        )
        multiple_judgement(excel_name, sheet_name, label_column, text_column, model)

    else:
        print("无效的选择！")


if __name__ == "__main__":
    main()
