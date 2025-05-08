import json
import torch
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from utils.data_loader import CWSPOSDataset
from models.joint_model import JointModel
from tqdm import tqdm

def evaluate_simple():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载标签映射
    with open("label2id.json", "r", encoding="utf-8") as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}

    # 加载分词器和测试集
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    test_dataset = CWSPOSDataset("data/test.txt", label2id, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # 初始化模型
    model = JointModel("bert-base-chinese", num_labels=len(label2id))
    model.load_state_dict(torch.load("joint_model.pt"))
    model.to(device)
    model.eval()

    total_tokens = 0
    correct_seg = 0
    correct_pos = 0
    correct_both = 0

    for batch in tqdm(test_loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            _, logits = model(input_ids=input_ids, attention_mask=attention_mask)

        predictions = torch.argmax(logits, dim=-1)

        for pred, gold in zip(predictions, labels):
            for p, g in zip(pred.cpu().numpy(), gold.cpu().numpy()):
                if g == -100:
                    continue
                total_tokens += 1
                pred_label = id2label[p]
                gold_label = id2label[g]

                # 分词正确
                if pred_label[0] == gold_label[0]:
                    correct_seg += 1

                # 词性正确
                if "-" in pred_label and "-" in gold_label:
                    if pred_label.split("-")[1] == gold_label.split("-")[1]:
                        correct_pos += 1

                # 联合标签完全一致
                if pred_label == gold_label:
                    correct_both += 1

    print()
    print(f"分词准确率:     {correct_seg / total_tokens:.4f}")
    print(f"词性准确率:     {correct_pos / total_tokens:.4f}")
    print(f"联合标签准确率: {correct_both / total_tokens:.4f}")

if __name__ == "__main__":
    evaluate_simple()
