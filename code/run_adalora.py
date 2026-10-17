"""
run_adalora.py - Chạy AdaLoRA baseline tren Kaggle.
Dung cho so sanh: Adaptive Rank vs Global Rank.
"""
import os, sys, argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from peft import AdaLoraConfig, get_peft_model
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(str(self.texts[idx]), truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def evaluate(model, loader, device):
    model.eval()
    preds, golds, nlls = [], [], []
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            loss = loss_fn(logits, batch["labels"])
            nlls.extend(loss.cpu().numpy())
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            golds.extend(batch["labels"].cpu().numpy())
    return f1_score(golds, preds, average="macro"), np.mean(nlls)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model", default="FacebookAI/xlm-roberta-base")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[300, 301, 302])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--init_rank", type=int, default=8)
    parser.add_argument("--target_rank", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_ds = TextDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer)
    test_ds = TextDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    results = []
    for seed in args.seeds:
        print(f"\n=== Running AdaLoRA Seed {seed} ===")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3)
        
        config = AdaLoraConfig(
            init_r=args.init_rank,
            target_r=args.target_rank,
            tinit=200,
            tfinal=1000,
            deltaT=10,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["query", "key", "value", "dense"],
            task_type="SEQ_CLS"
        )
        model = get_peft_model(model, config)
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)

        global_step = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Cap nhat phan bo rank cua AdaLoRA
                model.base_model.update_and_allocate(global_step)
                global_step += 1
                epoch_loss += loss.item()
                
            print(f"Epoch {epoch} Loss: {epoch_loss/len(train_loader):.4f}")

        # Eval
        f1, nll = evaluate(model, test_loader, device)
        print(f"Seed {seed} - Test F1: {f1:.4f}, NLL: {nll:.4f}")
        
        results.append({
            "seed": seed,
            "method": "adalora",
            "init_rank": args.init_rank,
            "target_rank": args.target_rank,
            "epochs": args.epochs,
            "lr": args.lr,
            "macro_f1": f1,
            "nll": nll
        })
        
        del model, optimizer
        torch.cuda.empty_cache()

    pd.DataFrame(results).to_csv(args.out, index=False)
    print(f"Da luu ket qua tai: {args.out}")

if __name__ == "__main__":
    main()