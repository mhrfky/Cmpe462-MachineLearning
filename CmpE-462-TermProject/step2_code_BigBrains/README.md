# CMPE 462 - Spring 2021

## Machine Learning Project Sentiment Analysis on IMDB User Reviews

## Mahir Efe Kaya - Yahya Bedirhan Pak (Team BigBrains)

### How to execute tests:

- Create a conda environment:

```
conda create -n 462projectstep2 python=3.6
conda activate 462projectstep2
```

- Install packages:

```
pip install -r requirements.txt
```

- Execute run script:

```
python 462project_step2_BigBrains.py <pkl-file> <test-dataset-folder>
```

Example:

```
python 462project_step2_BigBrains.py ./step2_model_BigBrains.pkl ./TEST
```

### How to train

- When you enter the test command, pkl file doesn't exists, our training algorithm runs and generates a pkl file. Then you can use it for test datasets
- Add train dataset under the ./TRAIN folder and validation dataset under the ./VAL folder
- Execute the script (even though if there is no pkl file yet, you need to pass it as an argument):

```
python 462project_step2_BigBrains.py ./step2_model_BigBrains.pkl ./VAL
```

- Then, you can test it with another test dataset, replacing with ./VAL folder:

```
python 462project_step2_BigBrains.py ./step2_model_BigBrains.pkl ./TEST
```
