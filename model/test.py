import json, random

data = json.load(open("data/processed/train.json", encoding="utf-8"))

print(f"Total: {len(data)}")
print(f"Avg desc len: {sum(len(d['description']) for d in data) / len(data):.0f}")
print(f"Avg regex len: {sum(len(d['regex']) for d in data) / len(data):.0f}")
print()

for item in random.sample(data, 15):
    print(f"DESC: {item['description'][:100]}")
    print(f"REG:  {item['regex'][:100]}")
    print()