### Google QUEST Q&A Labeling

比赛数据 [here](https://www.kaggle.com/c/google-quest-challenge/data)

训练示例:

`run.py --epochs=5 --max_sequence_length=500 --max_title_length=26 --max_question_length=260 --max_answer_length=210 --batch_accumulation=1 --batch_size=8 --warmup=300 --lr=1e-5 --bert_model=bert-base-uncased`

pseudo 训练示例:

`run.py --epochs=4 --max_sequence_length=500 --max_title_length=26 --max_question_length=260 --max_answer_length=210 --batch_accumulation=4 --batch_size=2 --warmup=250 --lr=2e-5 --bert_model=./bart.large --pseudo_file ../input/leak-free-pseudo-100k/pseudo-100k-4x-blend-no-leak-fold-{}.csv.gz --split_pseudo --leak_free_pseudo` 


