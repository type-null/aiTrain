# AI training
Authors: Sahil, Thomas, Ren

Training a Deep Learning classifier for a sentiment classifiction task.
Pre-processing library see [here](https://github.com/sahil350/ai_or_assignment_pre_processing).

## 1. Pre Processing ETL
- Modified [pre-processing library](https://github.com/sahil350/ai_or_assignment_pre_processing) to be able to load the dictionary word list from a zip file directly without unzipping if first.
- Added `<pad>`, `<unknown>` to the top of our word list `"index_arry.txt"` in [pre-processing library](https://github.com/sahil350/ai_or_assignment_pre_processing).

## 2. Run the Pre Processing on the dataset
- Shuffled and splitted the dataset into 'train' is 85% of the data, 'dev' is 10% and 'eval' is 5% of the data.

- Ran crawlers on each of them to created 3 tables to catalog.
![tableCreated](https://i.imgur.com/J0Sa4hf.png)

- Created Glue ETL job to map features. Code see [`glue_my_job_2.py`](glue_my_job_2.py).
![glueScript](https://i.imgur.com/0KUN1Wk.png)

- Ran the Glue ETL job on each one of them to create the 3 feature sets. Output json files see [`eval_data/eval.json`](eval_data/eval.json), [`validation_data/dev.json`](validation_data/dev.json) and [`training_data/train.json`](training_data/train.json).
![jobSucceeded](https://i.imgur.com/XybOSva.png)

## 3. Tensorflow model
Forked from https://github.com/pharnoux/columbia-aiops-training

- Changed embeddings same as our word list mentioned before. Embedding file is oversize for github, see [S3 bucket](https://ai2020.s3.amazonaws.com/hwk4/embeddings/embedding25d.txt).

- Built our model see folder [`model_training`](model_training)

- Ran the model locally. Successful result see below highlight line.
![modelLocal](https://i.imgur.com/yDaIct4.png)

## 4. SageMaker training
- Created a Notebook on SageMaker. Modified code to be able to load the data and the dictionary from S3.
![embeddingFromS3](https://i.imgur.com/B3CZPZ8.png)

- Ran the code form `Step #3` over there successfully.
![notebookResult](https://i.imgur.com/0Ox4ufD.png)

- Output result directly to S3.
![resultInS3](https://i.imgur.com/xvpMqMf.png)
