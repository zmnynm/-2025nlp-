import torch
from transformers import BertTokenizer
from models.joint_model import JointModel
from utils.utils import get_label_map
import json

# ⛓ 加载模型和标签
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 加载标签映射表
with open('label2id.json', 'r', encoding='utf-8') as f:
    label2id = json.load(f)
id2label = {int(v): k for k, v in label2id.items()}

model = JointModel('bert-base-chinese', num_labels=len(label2id))
model.load_state_dict(torch.load('joint_model.pt', map_location=device))
model.to(device)
model.eval()

def predict(sentence):
    tokens = tokenizer.tokenize(sentence)
    input_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])
    attention_mask = [1] * len(input_ids)

    input_ids += [0] * (128 - len(input_ids))
    attention_mask += [0] * (128 - len(attention_mask))

    input_ids = torch.tensor([input_ids]).to(device)
    attention_mask = torch.tensor([attention_mask]).to(device)

    with torch.no_grad():
        _, logits = model(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=-1).squeeze().tolist()

    pred_labels = [id2label.get(p, 'O') for p in preds[1:len(tokens)+1]]  # 去掉 [CLS] 和 [SEP]

    # 组装分词+词性
    result = []
    word = ''
    for char, label in zip(sentence, pred_labels):
        if '-' not in label:
            result.append(char + '/O')
            continue
        pos_tag = label.split('-')[1]
        boundary = label.split('-')[0]
        if boundary == 'B':
            word = char
        elif boundary == 'M':
            word += char
        elif boundary == 'E':
            word += char
            result.append(f'{word}/{pos_tag}')
            word = ''
        elif boundary == 'S':
            result.append(f'{char}/{pos_tag}')
    # 避免丢失最后一个词
    if word:
        result.append(f'{word}/{pos_tag}')

    return ' '.join(result)

if __name__ == '__main__':
    while True:
        sent = input("请输入一句中文（回车退出）: ").strip()
        if not sent:
            break
        output = predict(sent)
        print("分词 + 词性标注：")
        print(output)
