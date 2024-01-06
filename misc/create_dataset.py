import sys
import yaml
import pathlib

with open("StructGPT\examples\parameters\philosophy.yaml", "r") as file:
    parameters = yaml.safe_load(file)

sys.path.append("StructGPT")
from StructGPT.agents import Agent, AgentFunction, AdditionAF, FinalAnswerAF, ReadChunkAF, JournalistAF, DocumentMetaDataAF, ReadDocumentAF
from StructGPT.engine import Engine









instructions_structure_and_content = """I want you to summarize what this document is and what it talks about.
More precisely, I want you to tell me the overall structure of the document (what are the main parts? what language? what are the motivations? who's the author?)
Then I want you to tell me more about the document content and to give example of similar documents that could be useful as references.
Finally, I want you to propose improvements on the document. Please give a rather in-depth answer."""
instructions_spelling_mistakes = """I want you to identify all the possible spelling mistakes"""
instructions_parts = """What are the main parts of the document?"""
instructions_translation = """Translate the conclusion into Spanish"""
instructions_journalist_tabloid = """You're a tabloid journalist, write a report about the paper, for the general public"""
instructions_journalist_competent = """You're a highly competent journalist, write a report about the paper, for fellow scientists"""

instructions = [instructions_journalist_competent]
files = [f for f in pathlib.Path().glob("papiers/50.pdf")]
for document in files:
    document = str(document).replace("\\", "/")
    print(document)
    engine = Engine("gpt-4-1106-preview", parameters=parameters)
    engine.library.create_folder("papers")
    engine.library.folders["papers"].add_document(document)
    for instruction in instructions:
        final_answer = FinalAnswerAF("final_answer", "your final answer to the user")
        document_reader = ReadDocumentAF(
            "read_document", "will return the content of the document", engine
        )
        chunk_reader = ReadChunkAF(
            "read_chunk",
            "will return the content of a document chunk (index starts at 0)",
            engine,
        )

        journalist = JournalistAF(
            "journalist", "will write a news report with great skill about any subject", engine
        )
        metadata = DocumentMetaDataAF(
            "metadata",
            "returns metadata about the document (type, number of pages, chunks, letters etc.)",
            engine,
        )

        my_agent = Agent(
            engine,
            [
                final_answer,
                metadata,
                document_reader,
                chunk_reader,
                journalist
            ],
        )
        agent_answer = my_agent.run(instruction, save_for_dataset=True)
