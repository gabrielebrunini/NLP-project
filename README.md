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

For a complete project description, read the [Latex document]("https://github.com/gabrielebrunini/NLP-project/blob/main/Relation_Classification.pdf")


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
