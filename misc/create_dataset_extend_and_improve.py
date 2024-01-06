import sys
import yaml
import pathlib
import json

with open("StructGPT\examples\parameters\philosophy.yaml", "r") as file:
    parameters = yaml.safe_load(file)

sys.path.append("StructGPT")
from StructGPT.engine import Engine

def split_markdown(text):
    # Split the text by lines
    lines = text.split('\n')
    
    # Initialize variables
    parts = []
    current_part = []

    # Iterate through each line
    for line in lines:
        # Check if the line is a level 1 title
        if line.startswith('### '):
            # If there's content in the current part, add it to the parts list
            if current_part:
                parts.append('\n'.join(current_part))
                current_part = []
        # Add the line to the current part
        current_part.append(line)
    
    # Add the last part if it exists
    if current_part:
        parts.append('\n'.join(current_part))
    
    return parts

for i in range(230):
    if i == 76:
        break

    engine = Engine("gpt-3.5-turbo-1106", parameters=parameters)
    engine.library.create_folder("web")
    engine.library.folders["web"].add_document(f"url::https://encyclo-philo.fr/item/{i+1}")
    for j in range(3):
        with open(f"dataset_philo/{i}-{0}.json", "r", encoding="utf-8") as f:
            d = json.load(f)
            prompt = d["prompt"]
            answer = prompt.split("### Response:")[1].replace("</s>", "")
        parts = split_markdown(answer)
        for k, part in enumerate(parts):
            intro_opt = ""
            conclusion_opt = ""
            if "Introduction" not in answer:
                intro_opt = "Ne pas ajouter de partie d'introduction"
            if "Conclusion" not in answer:
                conclusion_opt = "Ne pas ajouter de partie de conclusion"
            ans_question = engine.query(f"Texte d'origine:\n{part}\n\n---\nRéécrire le texte d'origine en l'améliorant (enlever les références à l'article de l'encyclopédie et les contacts). Vous garderez le format markdown, et garderez les titres principaux.  Vous expliquerez cependant de manière plus détaillée et plus longue chaque partie du texte. L'objectif est d'avoir un ensemble de paragraphes relativement longs et cohérents, qui forment un argument fort. Vous écrirez entre 3 et 5 parties (au format markdown, chaque titre devra être précédé de ###). Les paragraphes explicatifs seront détaillées et sourcés. {intro_opt}\n{conclusion_opt}\n\nTexte détaillé au format Markdown: ", max_tokens=3000)
            prompt = f"<s>### Instruction: Vous êtes un professeur de philosophie, vous allez développer un texte qui nécessite plus de détails.\n### Input:\n{part}\n### Response:\n{ans_question.content}</s>"
            with open(f"dateset_philo_extend/{i}-{j}-{k}.json","w", encoding="utf-8") as f:
                json.dump({"prompt":prompt}, f, ensure_ascii=False, indent=4)