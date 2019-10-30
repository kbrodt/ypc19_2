# Yandex Programming Championship 2019, October

## Machine Learining track

### Stage 2

#### Topic model

Given a title and description of the document develop a classification model on 100 topics [(description of the task in russian)](https://contest.yandex.ru/contest/14696/problems/).

#### Appoach

For each document concatenate title and description and extract TfIdf features on words and char n-grams.
Then train one-vs-the-rest (OvR) multiclass linear model with modified Huber loss and stochastic gradient descent method.
 
This will reach a f1 score 60.83 (see [leaderboard](https://contest.yandex.ru/contest/14696/standings/?p=1)).

#### Usage

```bash
pip install -r requirements.txt
python train.py
```
