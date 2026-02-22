

import json
import argparse
import re
from difflib import SequenceMatcher
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_cosine_schedule_with_warmup



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class RegexDataset(Dataset):


    def __init__(self, data: list[dict], tokenizer: T5Tokenizer, max_input_len=128, max_target_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        input_text = f"generate regex: {entry['description']}"
        target_text = entry["regex"]

        input_enc = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target_enc = self.tokenizer(
            target_text,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = target_enc.input_ids.squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_enc.input_ids.squeeze(),
            "attention_mask": input_enc.attention_mask.squeeze(),
            "labels": labels,
        }


import warnings


def validate_regex(pattern: str) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            re.compile(pattern)
        return True
    except re.error:
        return False



def evaluate(model, dataloader, tokenizer, device, max_gen_batches=50, num_beams=4):

    model.eval()
    total_loss = 0
    total = 0
    total_score = 0
    accuracy = 0
    compilable = 0

    gen_kwargs = dict(
        max_length=256,
        do_sample=False,
        num_beams=num_beams,
        num_return_sequences=num_beams,
        early_stopping=True,
    )

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type="cuda"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()


            if i >= max_gen_batches:
                continue

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )


            all_decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

            label_ids = labels.clone()
            label_ids[label_ids == -100] = tokenizer.pad_token_id
            refs = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

            for sample_idx, ref in enumerate(refs):
                total += 1
                r = ref.strip()


                candidates = [
                    all_decoded[sample_idx * num_beams + b].strip()
                    for b in range(num_beams)
                ]


                best = candidates[0]
                best_compiles = False
                for cand in candidates:
                    if cand == r:
                        best = cand
                        best_compiles = True
                        break
                    if not best_compiles and validate_regex(cand):
                        best = cand
                        best_compiles = True

                p = best

                if p == r:
                    score = 1.0
                    compilable += 1
                    accuracy += 1
                else:
                    sim = SequenceMatcher(None, p, r).ratio()
                    compiles = validate_regex(p)
                    if compiles:
                        compilable += 1
                    if compiles and sim >= 0.8:
                        score = sim * 0.3
                    elif sim >= 0.8:
                        score = sim * 0.1
                    else:  
                        score = 0.0

                total_score += score

    avg_loss = total_loss / len(dataloader)
    avg_score = total_score / total if total > 0 else 0
    compile_rate = compilable / total if total > 0 else 0
    avg_accuracy = accuracy / total if total > 0 else 0

    return avg_loss, avg_score, compile_rate, avg_accuracy


def main():
    parser = argparse.ArgumentParser(description="Fine-tune T5-small for regex generation")
    parser.add_argument("--train-data", type=str,
                        default=str(Path(__file__).parent.parent / "data" / "processed" / "train.json"))
    parser.add_argument("--val-data", type=str,
                        default=str(Path(__file__).parent.parent / "data" / "processed" / "val.json"))
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).parent.parent / "checkpoints" / "t5-regex"))
    parser.add_argument("--model-name", type=str, default="google-t5/t5-small")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--max-input-len", type=int, default=256)
    parser.add_argument("--max-target-len", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps (effective batch = batch-size * grad-accum)")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing factor")

    args = parser.parse_args()


    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")


    print(f"Loading {args.model_name}...")
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<BS>", "{", "}", "^", "<", "`", "~"]})
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)



    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {param_count:,} total, {trainable:,} trainable")


    with open(args.train_data, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(args.val_data, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    train_dataset = RegexDataset(train_data, tokenizer, args.max_input_len, args.max_target_len)
    val_dataset = RegexDataset(val_data, tokenizer, args.max_input_len, args.max_target_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
    )


    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)


    scaler = torch.amp.GradScaler(enabled=False)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


    if args.label_smoothing > 0:
        print(f"Using label smoothing: {args.label_smoothing}")


    effective_batch = args.batch_size * args.grad_accum
    print(f"\nTraining for {args.epochs} epochs ({total_steps} optimizer steps)...")
    print(f"Effective batch size: {args.batch_size} × {args.grad_accum} = {effective_batch}")
    print("-" * 70)

    best_val_loss = float("inf")
    patience_counter = 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)


            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / args.grad_accum

            scaler.scale(loss).backward()


            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * args.grad_accum

            if step % args.log_every == 0:
                avg = epoch_loss / step
                lr = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch}/{args.epochs} | Step {step}/{len(train_loader)} | "
                      f"Loss: {avg:.4f} | LR: {lr:.2e}")


        val_loss, val_score, val_compile, val_accuracy = evaluate(model, val_loader, tokenizer, device)
        avg_train_loss = epoch_loss / len(train_loader)

        print(f"\n  Epoch {epoch} Summary:")
        print(f"    Train Loss:     {avg_train_loss:.4f}")
        print(f"    Val Loss:       {val_loss:.4f}")
        print(f"    Val Score:      {val_score:.4f}")
        print(f"    Val Compile:    {val_compile:.1%}")
        print(f"    Val Accuracy:   {val_accuracy:.1%}")


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            save_model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"    Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"    No improvement ({patience_counter}/{args.patience})")

            if patience_counter >= args.patience:
                print(f"\n  Early stopping at epoch {epoch}")
                break

        print("-" * 70)


    print(f"\nModel saved to: {output_dir}")

    print("Done!")


if __name__ == "__main__":
    main()