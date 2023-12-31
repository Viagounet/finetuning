import json
import glob
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from trl import SFTTrainer
from peft import PeftModelForCausalLM

prompt = "<s>### Instruction: Your role is to choose the corresponding function to answer the user query. You will be given a history of your previous actions and several other information in the input.\n### Input:\nYour goal is to You're a highly competent journalist, write a report about the paper, for fellow scientists.\nTo achieve this goal you will make good use of the following functions:\n- final_answer(your_final_answer: str) -> str ; final_answer your final answer to the user\n- metadata(document_path: str) -> str ; metadata returns metadata about the document (type, number of pages, chunks, letters etc.)\n- read_document(document_path: str) -> str ; read_document will return the content of the document\n- read_chunk(document_path: str, chunk_number: int) -> str ; read_chunk will return the content of a document chunk (index starts at 0)\n- journalist(subject: str, style: str, length: str, language: str) -> str ; journalist will write a news report with great skill about any subject\n\nNote: You will not make use of composite functions.\nThe following are the files you can work with. Always write their full path.\n- papers/papiers/50.pdf\n\n\n\nThese were the previous actions & results :\n\n- Action 1: metadata\nArguments: ['/papers/papiers/50.pdf']\nOutput: path: papiers/50.pdf\nwords: 10150\nletters: 85768\nchunks: 45\ndocument type: pdf\n- Action 2: read_chunk\nArguments: ['/papers/papiers/50.pdf', 0]\nOutput: B chooktor patos\nA planetary boundary for green water\nLan Wang-Erlandsson@, Ame Tobian@, Ruud J. van der Ent@, Ingo Fetzer\u00ae,\nSofie te Wierik\u00ae, Miina Porkka\u00ae, Arie Staal\u00ae, Fernando Jaramillo\nHeindriken Dahimann@, Chandrakant Singh\u00ae, Peter Greve@\u00ae, Dieter Gerten\u00ae,\nPatrick W. Keys, Tom Gleeson, Sarah E. Cornell, Will Steffen@, Xuemei Bai)\nand Johan Rockstrom@\nAbstract | Green water \u2014 terrestrial precipitation, evaporation and soil moisture\nis fundamental to Earth system dynamics and is now extensively perturbed by\nhuman pressures at continental to planetary scales. However, green water lacks\nexplicit consideration in the existing planetary boundaries framework that\ndemarcates a global safe operating space for humanity. In this Perspective, we\npropose a green water planetary boundary and estimate its current status. The\ngreen water planetary boundary can be represented by the percentage of ice-free\nland area on which root-zone soil moisture deviates from Holocene variability for\nany month of the year. Provisional estimates of departures from Holocene-like\nconditions, alongside evidence of widespread deterioration in Earth system\nfunctioning, indicate that the green water planetary boundary is already\ntransgressed, Moving forward, research needs to address and account for the role\nof root-zone soil moisture for Earth system resilience in view of ecohydrological,\nhydroclimatic and sociohydrological interactions.\n\u2018The planetary boundaries framework Global-scale and basin-scale definitions\ndemarcates a global safe operating space\nfor humanity based on Earth system\ndynamics\u2019 (1G, |:). The framework defines\nboundaries to human pressures on nine\nbiophysical systems and processes that\nregulate the state and resilience of the\nEarth system, using the comparatively stable\ninterglacial Holocene (initiated 11,700 years\nago) asthe baseline. Conceptually, each\nboundary is associated with a control\nvariable that allows tracking of risks for\nEarth system impacts. Notably, persistent\nand substantial boundary transgression of\neither ofthe two core boundaries \u2014 those\nfor Biosphere integrity\u2019 and \u2018Climate\nchange \u2014 can push the Earth system\ntowards an irreversible state shift.\n\u2018Transgression of other boundaries, including.\nthat of freshwater use, imply deterioration\nin Earth system functioning that can\nincrease the risk of regional \n- Action 3: read_chunk\nArguments: ['/papers/papiers/50.pdf', 1]\nOutput: regimeshifts\nand predispose transgressions of core\nboundaries. Based on the precautionary\nprinciple, the boundary is placed\nconservatively at the lower level of\nscientific uncertainty.\nof the \u2018Freshwater use\u2019 planetary boundary\n(PB) are solely defined by blue water\n(civers, lakes, reservoirs and renewable\ngroundwater stores) asa provisional proxy\nfor overall water flux changes ina river\nbasin. At the global scale, the boundary\nis currently set toan annual maximum\nof 4,000 km: consumptive blue water use.\nAt the basin scale, boundary positions are\nset based on minimum levels of monthly\nenvironmental water flow required to\nmaintain adequate aquatic ecosystem states.\nAccording to an estimated current global\nwater withdrawal rate of 2,600km\u2019 year\",\nthe \u2018Freshwater use\u2019 PB is deemed to be\nwithin the planetary-scale boundary, despite\nwidespread basin-scale transgressions\".\nYet, human pressures on green water\n(terrestrial precipitation, evaporation and\nsoil moisture) Earth system functions\nwere intended to be implicitly represented\nby the \u2018Freshwater use\u2019 PB\u2019, which\nfocuses solely on blue water. The lack of\nan explicit representation of green water\nin the planetary boundaries framework\ncan, therefore, conceal and misrepresent\nPERSPECTIVES\nextensive human modifications of green\nwater functions\u201d. For example, according\nto the current definition, deforestation that\ndeteriorates green water functioning in\nfavour of increased blue water availability\n\u2018would not contribute towards boundary\ntransgression\u2019. Given the fundamental\nimportance of green water for Earth system\nresilience, there isan urgent need to better\nunderstand the level of terrestrial wetness\nthat maintains a Holocene-like state of\nthe Earth system. Indeed, green water is\ncritical for supporting and regulating most\nterrestrial biosphere processes, including\nenergy, carbon, water and biogeochemical\ncycles\u2019, with human perturbation generating\nnon-linear changes, collapse and irreversible\nregime shifts in terrestrial ecosystems and\nhydroclimatic regimes\nIn this Perspective, we propose a green\nwater PB for quantifying green-water-related\nchanges that reflect the capacity of the Earth\nsystem to cope with human perturbations\n(FIG. 1b), We identify a set of processes\nthat comprehensively captures the\nhydroecological and hydroclimatic functions\nof green water \n- Action 4: read_chunk\nArguments: ['/papers/papiers/50.pdf', 2]\nOutput: inthe Earth system, and,\nbased on scientific evidence, propose a\ndefinition of a green water PB control\nvariable. Subsequently, the green water PB's\nboundary position and current status are\nset, and the use and interpretation of the PB\ndiscussed to guide sustainability governance.\nFinally, we discuss research priorities\nto better understand the biophysical\nand societal Earth-system-scale risks of\nsubstantial and persistent green water\n\u2018modifications. In doing so, we argue that\nthe \u2018Freshwater use\u2019 PB should be renamed.\nto the \u2018Freshwater change\u2019 PB composed of\ngreen and blue water components (1G.\nGreen water as control variable\nIn order to establish and define a green\nwater PB, an appropriate control variable\nneeds to be selected. Candidate variables\n\u2018must represent important green-water\ncontrol (FIG. |), rather than green-water\nresponses, to ecological and climatic change.\nFor this reason, green water indicators\nof anthropogenic appropriation (for\nexample, green water footprint) cannot\nbe considered. Instead, major non-blue\nfreshwater flows and stocks \u2014 precipitation,\nevaporation and soil moisture \u2014 form viable\n\u2018NATURE REVIEWS [EARTH & ENVIRONMENT\nPERSPECTIVES\n\u2018a Conceptual illustration of a green water planetary boundary\nOcean\nacidification\nBiogeochemical\nflows \u201c\nAerosol\nloading\nFreshwater\u201d\nchange\nOzone\ndepletion\nLand system\nchange\nNovel\nClimate\nchange\nBiosphere\nintegrity\nBoundary\nposition\n\u2018Zone within boundary\nLow degree of\nhuman modifications\nGreen water\nEarth system state driven by\nSupports\nHolocene-tke\nconditions\nIncreasing riskof\nobservable Earth\nsystem impacts\nHigh degree of\nhuman modifications\nb Green water relationships that are consi\n[|\n>\nLv la\n>\n* Greenhouse gas emissions\n* Aerosol emissions\n* Deforestation and other\nland eover changes\n+ Land and soil degradation\nWater withdrawal for\nirrigation and other uses\nTil Precipitation\nIl Evaporation\nTl Soil moisture\nFig. |The conceptual framework of a green water planetary boundary.\n|The planetary boundaries framework witht nine boundaries, including\nthe proposed renaming of Freshwater use\u2019 as'Freshwater change\u2019, subd\nVided into a blue and a green water sub-boundary. The lower panel ilus-\ntrates the relationship between the degree of human modification ofthe\n<reen \n\n---\nYou will now answer with an action (using a function) by precisely following this template :\n\nExplaination: Replace this text with your reasoning behind your action choice.\nAction: function(argument1, ...)\n\n### Response:"


peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM"
)
# model = get_peft_model("./mistral_function_calling_v0", peft_config)
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
peft_model_id = "./mistral_function_calling_v0"

model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto')
model.load_adapter(peft_model_id)

# merged_model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def generate_response(prompt, model):
  encoded_input = tokenizer(prompt,  return_tensors="pt", add_special_tokens=True)
  model_inputs = encoded_input.to('cuda')

  generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)

  decoded_output = tokenizer.batch_decode(generated_ids)

  return decoded_output[0].replace(prompt, "")

print(generate_response(prompt, model))