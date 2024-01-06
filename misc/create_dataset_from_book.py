import sys
import yaml
import pathlib
import json

with open("StructGPT\examples\parameters\philosophy.yaml", "r") as file:
    parameters = yaml.safe_load(file)

sys.path.append("StructGPT")
from StructGPT.engine import Engine


engine = Engine("gpt-3.5-turbo", parameters=parameters)
engine.library.create_folder("book")
engine.library.folders["book"].add_document(f"Histoire de la philosophie.pdf")
for chunk_number, chunk in enumerate(engine.library.folders["book"].documents[0].chunks[215:]):
    content = chunk.content
    engine.model = "gpt-3.5-turbo"
    print(content)
    print("/////////////////////")
    questions = engine.query(f"Texte: {content}\n\n---\nImaginer 5 questions différentes qu'un élève de philosophie pourrait poser, dont la réponse se trouve dans le texte fourni (exemples de question : 'expliquer le raisonnement de Platon lorsqu'il dit que...', 'décrire le processus de ..'). Ecrire une question par ligne.\n: Questions d'étudiants:\n1. Expliquer ", 
                             max_tokens=1024, penalties_ids=[48681, 421, 23013, 39128, 421, 23013, 627, 13485, 627, 417, 627, 1424, 2912])
    questions.content = "1. Expliquer " + questions.content
    print(questions.content)
    engine.model = "gpt-3.5-turbo-1106"
    for question_number, question in enumerate(questions.content.split("\n")):
        for i in range(5):
            question = question.replace(f"{i+1}. ", "")
        answer = engine.query(f"Texte: {content}\n\nRépondre à la question suivante en utilisant le texte et des connaissances générales pour compléter: {question}\nLa réponse doit être claire, au format markdown avec des titres (### Titre), en accentuant en gras ou souligné les parties importantes. Important: expliquer de manière détaillée en plusieurs paragraphes, en gardant les tournures de phrase du texte d'origine.", max_tokens=2048)
        prompt = f"<s>### Instruction: Vous êtes un professeur de philosophie, et vous répondez de manière détaillée aux questions d'un élève.\n### Input:\n{question}\n### Response:\n{answer.content}</s>"
        with open(f"dataset_philo_book/{chunk_number+215}-{question_number}.json","w", encoding="utf-8") as f:
            json.dump({"prompt":prompt}, f, ensure_ascii=False, indent=4)