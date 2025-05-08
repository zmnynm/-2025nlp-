import torch
import json
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from models.joint_model import JointModel
from utils.data_loader import CWSPOSDataset
from utils.utils import get_label_map
from tqdm import tqdm

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚙️ 使用设备: {device}")

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    label2id, id2label = get_label_map('data/train.txt')

    # 保存 label2id 映射
    with open('label2id.json', 'w', encoding='utf-8') as f:
        json.dump(label2id, f, ensure_ascii=False)

    train_dataset = CWSPOSDataset('data/train.txt', label2id, tokenizer)
    dev_dataset = CWSPOSDataset('data/dev.txt', label2id, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=32)

    model = JointModel('bert-base-chinese', num_labels=len(label2id)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss, _ = model(batch['input_ids'], batch['attention_mask'], batch['labels'])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] 训练损失: {avg_loss:.4f}")

    torch.save(model.state_dict(), 'joint_model.pt')
    print("✅ 模型训练完成，已保存为 joint_model.pt")

if __name__ == "__main__":
    train()
