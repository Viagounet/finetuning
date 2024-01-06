import sys
import yaml
import pathlib
import json
import glob
from random import choice

for j, file in enumerate(glob.glob("dataset_philo_rag/data*.json")):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        data = data["data"]
    for i, turn in enumerate(data):
        prompt = f"""<s>### Instruction: A partir des conditions et des documents fournis, répondre à la question.
### Input:
{turn['full_prompt']}

### Response:
{turn['answer']}</s>"""
        with open(f"dataset_philo_rag2/rag-{i}-{j}.json", "w", encoding="utf-8") as f:
            json.dump({"prompt":prompt}, f, ensure_ascii=False, indent=4)