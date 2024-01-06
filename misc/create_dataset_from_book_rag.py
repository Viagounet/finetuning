import sys
import yaml
import pathlib
import json
from random import choice

with open("StructGPT\examples\parameters\philosophy.yaml", "r") as file:
    parameters = yaml.safe_load(file)

sys.path.append("StructGPT")
from StructGPT.engine import Engine


engine = Engine("gpt-4-1106-preview", parameters=parameters)
engine.library.create_folder("book")
engine.library.folders["book"].add_document(f"conférence arelacler.pdf")
themes = ["l'histoire de la philosophie", 
          "les présocratiques", 
          "Socrate", 
          "Platon", 
          "Platon et l'Académie", 
          "Aristote", 
          "Aristote et le lycée", 
          "la philosophie (période helléniste et romaine)", 
          "l'épicurisme"]
questions = {"questions":[]}
data = {"data": []}
for theme in themes:
    random_chunk = choice(engine.library.folders["book"].documents[0].chunks)
    random_chunk_text = random_chunk.content
    questions = engine.query(f"Texte: {random_chunk_text}\n---\nImaginer 3 questions dont la réponse se trouve dans le texte fourni. Une question par ligne. Uniquement écrire la question, sans écrire 'selon le texte'.\n3 questions:\n- ", 
                            max_tokens=2048)
    for question in questions.content.split("\n"):
        for i in range(3):
            question = question.replace(f"{i+1}. ", "")
        print(question)
        prompt = f"""Conditions:
1. Dire la vérité
2. Indiquer si la réponse n'est présente dans les textes fournis.
3. Citer les segments de texte soutenant l'explication entre balises <citation> et </citation>.

Exemple:
Socrate était (...)
<citation> Il est vrai que Socrate (...) </citation>
De plus, il convient de noter que l'époque (...)
<citation> L'époque (...) </citation>


4. Ecrire la réponse au format Markdown. (titres, liste à puce, accentuation etc.)
5. En plus du texte, ajouter une partie supplémentaire à la fin de la réponse, sur des idées complémentaires qui permettraient de répondre à la question en plus de détails.
---
Répondre à la question suivante en respectant les conditions: {question}
"""
        answer, full_prompt = engine.query_folder(prompt, "book", 2048, 0, top_N=5, return_prompt=True)
        data["data"].append({"question":question, "mini_prompt":prompt, "full_prompt":full_prompt, "answer": answer.content})
        
        with open(f"dataset_philo_rag/data4.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)