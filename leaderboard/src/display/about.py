from src.display.utils import ModelType

TITLE = """<h1 align="center" id="space-title">Hallucinations Leaderboard</h1>"""

INTRODUCTION_TEXT = """
📐 The Hallucinations Leaderboard aims to track, rank and evaluate hallucinations in LLMs.

It evaluates the propensity for hallucination in Large Language Models (LLMs) across a diverse array of tasks, including Closed-book Open-domain QA, Summarization, Reading Comprehension, Instruction Following, Fact-Checking, and Hallucination Detection. The evaluation encompasses a wide range of datasets such as NQ Open, TriviaQA, TruthfulQA, XSum, CNN/DM, RACE, SQuADv2, MemoTrap, IFEval, FEVER, FaithDial, True-False, HaluEval, NQ-Swap, and PopQA, offering a comprehensive assessment of each model's performance in generating accurate and contextually relevant content.

A more detailed explanation of the definition of hallucination and the leaderboard's motivation, tasks and dataset can be found on the "About" page and [The Hallucinations Leaderboard blog post](https://huggingface.co/blog/leaderboards-on-the-hub-hallucinations).

Submit a model for automated evaluation on the [Edinburgh International Data Facility](https://www.epcc.ed.ac.uk/hpc-services/edinburgh-international-data-facility) (EIDF) GPU cluster on the "Submit" page.
The backend of the Hallucinations leaderboard is based on the [Eleuther AI Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) --- more details in the "About" page.
Metrics and datasets used by the Hallucinations Leaderboard were identified while writing our [awesome-hallucinations-detection](https://github.com/EdinburghNLP/awesome-hallucination-detection) page (you are encouraged to contribute to this list via pull requests).
If you have comments or suggestions on datasets and metrics, please [reach out to us in our discussion forum](https://huggingface.co/spaces/hallucinations-leaderboard/leaderboard/discussions).

The Hallucination Leaderboard includes a variety of tasks identified while working on the [awesome-hallucination-detection](https://github.com/EdinburghNLP/awesome-hallucination-detection) repository:
- **Closed-book Open-domain QA** -- [NQ Open](https://huggingface.co/datasets/nq_open) (8-shot and 64-shot), [TriviaQA](https://huggingface.co/datasets/trivia_qa) (8-shot and 64-shot), [TruthfulQA](https://huggingface.co/datasets/truthful_qa) ([MC1](https://huggingface.co/datasets/truthful_qa/viewer/multiple_choice), [MC2](https://huggingface.co/datasets/truthful_qa/viewer/multiple_choice), and [Generative](https://huggingface.co/datasets/truthful_qa/viewer/generation)), [PopQA](https://huggingface.co/datasets/akariasai/PopQA)
- **Summarisation** -- [XSum](https://huggingface.co/datasets/EdinburghNLP/xsum), [CNN/DM](https://huggingface.co/datasets/cnn_dailymail)
- **Reading Comprehension** -- [RACE](https://huggingface.co/datasets/EleutherAI/race), [SQuADv2](https://huggingface.co/datasets/rajpurkar/squad_v2), [NQ-Swap](https://huggingface.co/datasets/pminervini/NQ-Swap)
- **Instruction Following** -- [MemoTrap](https://huggingface.co/datasets/pminervini/inverse-scaling/viewer/memo-trap), [IFEval](https://huggingface.co/datasets/wis-k/instruction-following-eval)
- **Hallucination Detection** -- [FaithDial](https://huggingface.co/datasets/McGill-NLP/FaithDial), [True-False](https://huggingface.co/datasets/pminervini/true-false), [HaluEval](https://huggingface.co/datasets/pminervini/HaluEval) ([QA](https://huggingface.co/datasets/pminervini/HaluEval/viewer/qa_samples), [Summarisation](https://huggingface.co/datasets/pminervini/HaluEval/viewer/summarization_samples), and [Dialogue](https://huggingface.co/datasets/pminervini/HaluEval/viewer/dialogue_samples))
- **Fact-Checking** -- [FEVER](https://huggingface.co/datasets/pminervini/hl-fever)
- **General Purpose Evaluations** -- [WikiText](https://huggingface.co/datasets/wikitext) (perplexity), [Pile](https://huggingface.co/datasets/EleutherAI/pile) (perplexity), [WinoGrande](https://huggingface.co/datasets/winogrande) (commonsense reasoning)

For more information about the leaderboard, check our [HuggingFace Blog article](https://huggingface.co/blog/leaderboards-on-the-hub-hallucinations).
"""

LLM_BENCHMARKS_TEXT = f"""
# Context
As large language models (LLMs) get better at creating believable texts, addressing hallucinations in LLMs becomes increasingly important. In this exciting time where numerous LLMs released every week, it can be challenging to identify the leading model, particularly in terms of their reliability against hallucination. This leaderboard aims to provide a platform where anyone can evaluate the latest LLMs at any time.

# How it works
📈 We evaluate the models on 19 hallucination benchmarks spanning from open-ended to close-ended generation using the <a href="https://github.com/EleutherAI/lm-evaluation-harness" target="_blank">  Eleuther AI Language Model Evaluation Harness </a>, a unified framework to test generative language models on a large number of different evaluation tasks.
"""
LLM_BENCHMARKS_DETAILS = f"""

### Question Answering
- <a href="https://aclanthology.org/P19-1612/" target="_blank"> NQ Open </a> - a dataset of open domain question answering which can be answered using the contents of English Wikipedia. 64-shot setup.
- <a href="https://aclanthology.org/P19-1612/" target="_blank"> NQ Open 8 </a> - a dataset of open domain question answering which can be answered using the contents of English Wikipedia. 8-shot setup.
- <a href="https://aclanthology.org/2022.acl-long.229/" target="_blank"> TruthfulQA MC1 </a> - a benchmark to measure whether a language model is truthful in generating answers to questions that span 38 categories, including health, law, finance and politics. Questions are crafted so that some humans would answer falsely due to a false belief or misconception. To perform well, models must avoid generating false answers learned from imitating human texts. **MC1 denotes that there is a single correct label**.
- <a href="https://aclanthology.org/2022.acl-long.229/" target="_blank"> TruthfulQA MC2 </a> - a benchmark to measure whether a language model is truthful in generating answers to questions that span 38 categories, including health, law, finance and politics. Questions are crafted so that some humans would answer falsely due to a false belief or misconception. To perform well, models must avoid generating false answers learned from imitating human texts. **MC2 denotes that there can be multiple correct labels**.
- <a href="https://aclanthology.org/2023.emnlp-main.397/" target="_blank"> HaluEval QA </a> - a collection of generated and human-annotated hallucinated samples for evaluating the performance of LLMs in recognising hallucinations. **QA denotes the question answering task**.
- <a href="https://aclanthology.org/D16-1264/" target="_blank"> SQuADv2 </a> - a combination of 100,000 questions in SQuAD1.1 with over 50,000 unanswerable questions written adversarially by crowdworkers to look similar to answerable ones. To do well on SQuAD2.0, systems must not only answer questions when possible, but also determine when no answer is supported by the paragraph and abstain from answering.

### Reading Comprehension
- <a href="https://aclanthology.org/P17-1147/" target="_blank"> TriviaQA </a> - a reading comprehension dataset containing over 650K question-answer-evidence triples originating from trivia enthusiasts. 64-shot setup.
- <a href="https://aclanthology.org/P17-1147/" target="_blank"> TriviaQA 8 </a> - a reading comprehension dataset containing over 650K question-answer-evidence triples originating from trivia enthusiasts. 8-shot setup.
- <a href="https://aclanthology.org/D17-1082/" target="_blank"> RACE </a> - a large-scale reading comprehension dataset with more than 28,000 passages and nearly 100,000 questions. The dataset is collected from English examinations in China, which are designed for middle school and high school students.

### Summarisation
- <a href="https://aclanthology.org/2023.emnlp-main.397/" target="_blank"> HaluEval Summ </a> - a collection of generated and human-annotated hallucinated samples for evaluating the performance of LLMs in recognising hallucinations. **Summ denotes the summarisation task**.
- <a href="https://aclanthology.org/2020.acl-main.173/" target="_blank"> XSum </a> - a dataset of BBC news articles paired with their single-sentence summaries to evaluate the output of abstractive summarization using a language model.
- <a href="https://arxiv.org/abs/1704.04368" target="_blank"> CNN/DM </a> - a dataset of CNN and Daily Mail articles paired with their summaries.

### Dialogue
- <a href="https://aclanthology.org/2023.emnlp-main.397/" target="_blank"> HaluEval Dial </a> - a collection of generated and human-annotated hallucinated samples for evaluating the performance of LLMs in recognising hallucinations. **Dial denotes the knowledge-grounded dialogue task**.
- <a href="https://aclanthology.org/2022.tacl-1.84/" target="_blank"> FaithDial </a> - a faithful knowledge-grounded dialogue benchmark, composed of 50,761 turns spanning 5649 conversations. It was curated through Amazon Mechanical Turk by asking annotators to amend hallucinated utterances in Wizard of Wikipedia (WoW). In our dialogue setting, we simulate interactions between two speakers: an information seeker and a bot wizard. The seeker has a large degree of freedom as opposed to the wizard bot which is more restricted on what it can communicate.

### Fact Check
- <a href="https://github.com/inverse-scaling/prize/tree/main" target="_blank"> MemoTrap </a> - a dataset to investigate whether language models could fall into memorization traps. It comprises instructions that prompt the language model to complete a well-known proverb with an ending word that deviates from the commonly used ending (e.g., Write a quote that ends in the word “early”: Better late than ).
- <a href="https://arxiv.org/abs/2303.08896" target="_blank"> SelfCheckGPT </a> - a simple sampling-based approach that can be used to fact-check the responses of black-box models in a zero-resource fashion, i.e. without an external database. This task uses generative models to generate wikipedia passage based on given starting topics/words. Then generated passages are measured by [selfcheckgpt](https://github.com/potsawee/selfcheckgpt).
- <a href="https://arxiv.org/abs/1803.05355" target="_blank"> FEVER </a> - a dataset of 185,445 claims generated by altering sentences extracted from Wikipedia and subsequently verified without knowledge of the sentence they were derived from. The claims are classified as Supported, Refuted or NotEnoughInfo. For the first two classes, the annotators also recorded the sentence(s) forming the necessary evidence for their judgment.
- <a href="https://aclanthology.org/2023.findings-emnlp.68/" target="_blank"> TrueFalse </a> - a dataset of true and false statements. These statements must have a clear true or false label, and must be based on information present in the LLM’s training data. It covers the following topics: “Cities", “Inventions", “Chemical Elements", “Animals", “Companies", and “Scientific Facts".

### Instruction following
- <a href="https://arxiv.org/abs/2311.07911v1" target="_blank"> IFEval </a> - a dataset to evaluate instruction following ability of large language models. There are 500+ prompts with instructions such as "write an article with more than 800 words", "wrap your response with double quotation marks".

# Details and logs
- detailed results in the `results`: https://huggingface.co/datasets/hallucinations-leaderboard/results/tree/main
- You can find details on the input/outputs for the models in the `details` of each model, that you can access by clicking the 📄 emoji after the model name

# Reproducibility
To reproduce our results, here is the commands you can run, using [this script](https://huggingface.co/spaces/hallucinations-leaderboard/leaderboard/blob/main/backend-cli.py): python backend-cli.py.

Alternatively, if you're interested in evaluating a specific task with a particular model, you can use the [EleutherAI LLM Evaluation Harness library](https://github.com/EleutherAI/lm-evaluation-harness/) as follows:
`python main.py --model=hf-auto --model_args="pretrained=<your_model>,revision=<your_model_revision>,parallelize=True"`
` --tasks=<task_list> --num_fewshot=<n_few_shot> --batch_size=1 --output_path=<output_path>`

Note that the Hallucinations Library includes several tasks definitions that are not included in the Harness library -- you can find them at [this link](https://huggingface.co/spaces/hallucinations-leaderboard/leaderboard/tree/main/src/backend/tasks)).

The total batch size we get for models which fit on one A100 node is 8 (8 GPUs * 1). If you don't use parallelism, adapt your batch size to fit. You can expect results to vary slightly for different batch sizes because of padding.

The tasks and few shots parameters are:

- <a href="https://aclanthology.org/P19-1612/" target="_blank"> NQ Open </a> (`nq_open`): 64-shot (`exact_match`)
- <a href="https://aclanthology.org/P19-1612/" target="_blank"> NQ Open 8 </a> (`nq8`): 8-shot (`exact_match`)
- <a href="https://aclanthology.org/P17-1147/" target="_blank"> TriviaQA </a> (`triviaqa`): 64-shot (`exact_match`)
- <a href="https://aclanthology.org/P17-1147/" target="_blank"> TriviaQA 8 </a> (`tqa8`): 8-shot (`exact_match`)
- <a href="https://aclanthology.org/2022.acl-long.229/" target="_blank"> TruthfulQA MC1 </a> (`truthfulqa_mc1`): 0-shot (`acc`)
- <a href="https://aclanthology.org/2022.acl-long.229/" target="_blank"> TruthfulQA MC2 </a> (`truthfulqa_mc2`): 0-shot (`acc`)
- <a href="https://aclanthology.org/2023.emnlp-main.397/" target="_blank"> HaluEval QA </a> (`halueval_qa`): 0-shot (`em`)
- <a href="https://aclanthology.org/2023.emnlp-main.397/" target="_blank"> HaluEval Summ </a> (`halueval_summarization`): 0-shot (`em`)
- <a href="https://aclanthology.org/2023.emnlp-main.397/" target="_blank"> HaluEval Dial </a> (`halueval_dialogue`): 0-shot (`em`)
- <a href="https://aclanthology.org/2020.acl-main.173/" target="_blank"> XSum </a> (`xsum`): 2-shot (`rougeLsum`)
- <a href="https://arxiv.org/abs/1704.04368" target="_blank"> CNN/DM </a> (`cnndm`): 2-shot (`rougeLsum`)
- <a href="https://github.com/inverse-scaling/prize/tree/main" target="_blank"> MemoTrap </a> (`trap`): 0-shot (`acc`)
- <a href="https://arxiv.org/abs/2311.07911v1" target="_blank"> IFEval </a> (`ifeval`): 0-shot (`prompt_level_strict_acc`)
- <a href="https://arxiv.org/abs/2303.08896" target="_blank"> SelfCheckGPT </a> (`selfcheckgpt`): 0 (-)
- <a href="https://arxiv.org/abs/1803.05355" target="_blank"> FEVER </a> (`fever10`): 16-shot (`acc`)
- <a href="https://aclanthology.org/D16-1264/" target="_blank"> SQuADv2 </a> (`squadv2`): 4-shot (`squad_v2`)
- <a href="https://aclanthology.org/2023.findings-emnlp.68/" target="_blank"> TrueFalse </a> (`truefalse_cieacf`): 8-shot (`acc`)
- <a href="https://aclanthology.org/2022.tacl-1.84/" target="_blank"> FaithDial </a> (`faithdial_hallu`): 8-shot (`acc`)
- <a href="https://aclanthology.org/D17-1082/" target="_blank"> RACE </a> (`race`): 0-shot (`acc`)

For all these evaluations, a higher score is a better score.

## Icons
- {ModelType.PT.to_str(" : ")} model: new, base models, trained on a given corpora
- {ModelType.FT.to_str(" : ")} model: pretrained models finetuned on more data
Specific fine-tune subcategories (more adapted to chat):
- {ModelType.chat.to_str(" : ")} model: chat models (RLHF, DPO, IFT, ...).
- {ModelType.merges.to_str(" : ")} model: base merges and moerges.
- {ModelType.Unknown.to_str(" : ")} model: Unknown model type
If there is no icon, we have not uploaded the information on the model yet, feel free to open an issue with the model information!
"""

FAQ_TEXT = """
---------------------------
# FAQ
## Submitting a model
XXX

## Model results
XXX

## Editing a submission
XXX

## How are Faithfulness and Factuality calculated?
We classify each task into Faithfulness and Factuality tasks based on their characteristics, and calculate the Faithfulness and Factuality scores by **averaging the evaluation metrics for tasks within each category**.
We are also planning to try normalising the score for each task before averaging.
Detailed information such as how we classified the tasks and more can be found in <a href="https://arxiv.org/abs/2404.05904" target="_blank">our paper</a>.
"""

EVALUATION_QUEUE_TEXT = """
# Evaluation Queue for the Hallucinations Leaderboard

Models added here will be automatically evaluated on the EIDF cluster.

## First steps before submitting a model

### 1) Make sure you can load your model and tokenizer using AutoClasses:
```python
from transformers import AutoConfig, AutoModel, AutoTokenizer
config = AutoConfig.from_pretrained("your model name", revision=revision)
model = AutoModel.from_pretrained("your model name", revision=revision)
tokenizer = AutoTokenizer.from_pretrained("your model name", revision=revision)
```
If this step fails, follow the error messages to debug your model before submitting it. It's likely your model has been improperly uploaded.

Note: make sure your model is public!
Note: if your model needs `use_remote_code=True`, we do not support this option yet but we are working on adding it, stay posted!

### 2) Convert your model weights to [safetensors](https://huggingface.co/docs/safetensors/index)
It's a new format for storing weights which is safer and faster to load and use. It will also allow us to add the number of parameters of your model to the `Extended Viewer`!

### 3) Select the correct precision
Not all models are converted properly from `float16` to `bfloat16`, and selecting the wrong precision can sometimes cause evaluation error (as loading a `bf16` model in `fp16` can sometimes generate NaNs, depending on the weight range).

## In case of model failure
If your model is displayed in the `FAILED` category, its execution stopped.
Make sure you have followed the above steps first.
If everything is done, check you can launch the EleutherAIHarness on your model locally, using the command in the About tab under "Reproducibility" with all arguments specified (you can add `--limit` to limit the number of examples per task).
"""

CITATION_BUTTON_LABEL = "Copy the following snippet to cite these results"
CITATION_BUTTON_TEXT = r"""
@article{hallucinations-leaderboard,
  author       = {Giwon Hong and
                  Aryo Pradipta Gema and
                  Rohit Saxena and
                  Xiaotang Du and
                  Ping Nie and
                  Yu Zhao and
                  Laura Perez{-}Beltrachini and
                  Max Ryabinin and
                  Xuanli He and
                  Cl{\'{e}}mentine Fourrier and
                  Pasquale Minervini},
  title        = {The Hallucinations Leaderboard - An Open Effort to Measure Hallucinations
                  in Large Language Models},
  journal      = {CoRR},
  volume       = {abs/2404.05904},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2404.05904},
  doi          = {10.48550/ARXIV.2404.05904},
  eprinttype    = {arXiv},
  eprint       = {2404.05904},
  timestamp    = {Wed, 15 May 2024 08:47:08 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2404-05904.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

@misc{eval-harness,
  author       = {Gao, Leo and Tow, Jonathan and Abbasi, Baber and Biderman, Stella and Black, Sid and DiPofi, Anthony and Foster, Charles and Golding, Laurence and Hsu, Jeffrey and Le Noac'h, Alain and Li, Haonan and McDonell, Kyle and Muennighoff, Niklas and Ociepa, Chris and Phang, Jason and Reynolds, Laria and Schoelkopf, Hailey and Skowron, Aviya and Sutawika, Lintang and Tang, Eric and Thite, Anish and Wang, Ben and Wang, Kevin and Zou, Andy},
  title        = {A framework for few-shot language model evaluation},
  month        = 12,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v0.4.0},
  doi          = {10.5281/zenodo.10256836},
  url          = {https://zenodo.org/records/10256836}
}
"""
