# Relation extraction and textual entailment
In this course project, We propose to investigate
the intersection of Natural Language Inference
and Relation Classification. Relation
Classification is the task of determining the
relationship between two entities. It is considered
one of the Natural Language Understanding
building blocks and has critical applications,
e.g., it is a sub-task of Relation
Extraction and KB completion. We think
about the task as a standard supervised learning
task, which might have an interesting intersection
with natural language inference. We
could exploit this aspect to train better performing
models – To investigate such, we
conduct transformer-based experiments on an
established benchmark in Relation Classification:
the SemEVal2010 Task 8 dataset.

## login to leonhard

```shell
ssh username@login.leonhard.ethz.ch
```

## Actionables

- [x] create github repo
- [ ] set up environment (no anaconda anymore, updated steps below)
- [ ] everybody gets access to leonhard
- [ ] download RoBERTa models (gpu nodes don't have internet access, so before running it one, has to download the models manually)
- [ ] everybody can run the initial code


## environment

- [ ]perhaps delete anaconda (something along those lines)

```shell
rm -r anaconda3
```

load the right modules

```shell
module load gcc/6.3.0 python_gpu/3.7.4
```

install relevant libraries

- [ ] install transformers
```shell
pip install --user transformers
```

- [ ] install sklearn
```shell
pip install --user sklearn
```

- [ ] copy repo to leonhard
```shell
git clone https://github.com/dominiksinsaarland/relation_extraction_and_textual_entailment.git
```


## download model
need to do this once: Save the model into a local directory. If we run things on gpu nodes, they won't have internet access and fail as we have seen in the zoom call with Gabriele's examples

```shell
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer
# this is roberta-large
output_dir = "roberta-large-local-copy"
model_name = "roberta-large"

config = RobertaConfig.from_pretrained(model_name)
config.save_pretrained(output_dir)
tokenizer = RobertaTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(output_dir)
model = RobertaModel.from_pretrained(model_name)
model.save_pretrained(output_dir)

# this is roberta-large-mnli
model_name = "roberta-large-mnli"
output_dir = "roberta-large-mnli-local-copy"
config = RobertaConfig.from_pretrained(model_name)
config.save_pretrained(output_dir)
tokenizer = RobertaTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(output_dir)
model = RobertaModel.from_pretrained(model_name)
model.save_pretrained(output_dir)
```
## run R-BERT with roberta-large-mnli on leonhard

go to the right directory
```shell
cd relation_extraction_and_textual_entailment
```

now, we have to point model_name_or_path to the right directory. if you saved them in the previous folder, replace with e.g. model_name_or_path ../roberta-large-mnli-local-copy

```shell
bsub -n 1 -R "rusage[mem=12800,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python main.py --do_train --do_eval --train_batch_size 4 --gradient_accumulation_steps 4 --model_name_or_path roberta-large-mnli-local-copy
```

After training for 10 epochs:
acc = 0.8587
f1 = 0.8921

same here with model_name_or_path

## run R-BERT with roberta-large on leonhard

```shell
bsub -n 1 -R "rusage[mem=12800,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python main.py --do_train --do_eval --train_batch_size 4 --gradient_accumulation_steps 4 --model_name_or_path roberta-large-local-copy
```

After training for 10 epochs:
acc = 0.8653
f1 = 0.8966g

# Gabriele results

roberta-large-mnli-local-copy
acc = 0.8623
f1 = 0.8921
loss = 1.1602

roberta-large-mnli-local-copy
05/13/2021 13:32:09 - INFO - trainer -   ***** Eval results *****
05/13/2021 13:32:09 - INFO - trainer -     acc = 0.8623
05/13/2021 13:32:09 - INFO - trainer -     f1 = 0.8921
05/13/2021 13:32:09 - INFO - trainer -     loss = 1.1602

roberta-large-mnli-local-copy
05/13/2021 13:32:38 - INFO - trainer -   ***** Eval results *****
05/13/2021 13:32:38 - INFO - trainer -     acc = 0.8623
05/13/2021 13:32:38 - INFO - trainer -     f1 = 0.8921
05/13/2021 13:32:38 - INFO - trainer -     loss = 1.1602

roberta-large-mnli-local-copy
05/13/2021 13:33:14 - INFO - trainer -   ***** Eval results *****
05/13/2021 13:33:14 - INFO - trainer -     acc = 0.8623
05/13/2021 13:33:14 - INFO - trainer -     f1 = 0.8921
05/13/2021 13:33:14 - INFO - trainer -     loss = 1.1602

roberta-large-local-copy
acc = 0.8579
f1 = 0.8869
loss = 1.1475

roberta-large-local-copy
05/13/2021 13:33:00 - INFO - trainer -   ***** Eval results *****
05/13/2021 13:33:00 - INFO - trainer -     acc = 0.8579
05/13/2021 13:33:00 - INFO - trainer -     f1 = 0.8869
05/13/2021 13:33:00 - INFO - trainer -     loss = 1.1475

roberta-large-local-copy
05/13/2021 13:47:33 - INFO - trainer -   ***** Eval results *****
05/13/2021 13:47:33 - INFO - trainer -     acc = 0.8623
05/13/2021 13:47:33 - INFO - trainer -     f1 = 0.8940
05/13/2021 13:47:33 - INFO - trainer -     loss = 1.1431

roberta-large-local-copy
05/13/2021 13:52:39 - INFO - trainer -   ***** Eval results *****
05/13/2021 13:52:39 - INFO - trainer -     acc = 0.8623
05/13/2021 13:52:39 - INFO - trainer -     f1 = 0.8940
05/13/2021 13:52:39 - INFO - trainer -     loss = 1.1431

# Didem-Results
## roberta-large-mnli-local-copy

05/12/2021 19:36:13 - INFO - trainer -   ***** Eval results ***** <br/>
05/12/2021 19:36:13 - INFO - trainer -     acc = 0.8623 <br/>
05/12/2021 19:36:13 - INFO - trainer -     f1 = 0.8921 <br/>
05/12/2021 19:36:13 - INFO - trainer -     loss = 1.1602 <br/>


## roberta-large-local-copy

05/12/2021 19:53:55 - INFO - trainer -   ***** Eval results ***** <br/>
05/12/2021 19:53:55 - INFO - trainer -     acc = 0.8623 <br/>
05/12/2021 19:53:55 - INFO - trainer -     f1 = 0.8940 <br/>
05/12/2021 19:53:55 - INFO - trainer -     loss = 1.1431 <br/>
# NLP-project
