import sys
import yaml
import pathlib
import json

with open("StructGPT\examples\parameters\philosophy.yaml", "r") as file:
    parameters = yaml.safe_load(file)

sys.path.append("StructGPT")
from StructGPT.engine import Engine

for i in range(230):
    if i == 76:
        break
    engine = Engine("gpt-3.5-turbo", parameters=parameters)
    engine.library.create_folder("web")
    engine.library.folders["web"].add_document(f"url::https://encyclo-philo.fr/item/{i+1}")
    for j in range(3):
        ans_rewrite = engine.query_folder("Réécrire le texte en plusieurs paragraphes de manière claire et formatée, en incluant si besoin des listes à puce et des tableaux (markdown). Garder suffisamment de détails. Si possible, rajouter des exemples de comment utiliser les informations présentées pour une dissertation ou une explication de texte.", "web", max_tokens=2048, temperature=0.3)
        ans_question = engine.query(f"Texte : {ans_rewrite.content}\n\n---\nImaginer une question qu'un étudiant en philosophie pourrait poser, et dont le texte pourrait être une réponse.", max_tokens=256)
        prompt = f"<s>### Instruction: Vous êtes un professeur de philosophie, et vous répondez de manière détaillée aux questions d'un élève.\n### Input:\n{ans_question.content}\n### Response:\n{ans_rewrite.content}</s>"
        with open(f"dataset_philo/{i}-{j}.json","w", encoding="utf-8") as f:
            json.dump({"prompt":prompt}, f)