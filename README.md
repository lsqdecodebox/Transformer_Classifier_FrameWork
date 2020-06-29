## 基于bert分类 的 代码框架

主要依赖：
	pytorch
	hugging face transformers
	

训练示例:

`run.py --epochs=5 --max_sequence_length=500 --max_title_length=26 --max_question_length=260 --max_answer_length=210 --batch_accumulation=1 --batch_size=8 --warmup=300 --lr=1e-5 --bert_model=bert-base-uncased`

pseudo 训练示例:

`run.py --epochs=4 --max_sequence_length=500 --max_title_length=26 --max_question_length=260 --max_answer_length=210 --batch_accumulation=4 --batch_size=2 --warmup=250 --lr=2e-5 --bert_model=./bart.large --pseudo_file ../input/leak-free-pseudo-100k/pseudo-100k-4x-blend-no-leak-fold-{}.csv.gz --split_pseudo --leak_free_pseudo` 





概述：
自适应层权重
multi-samplt-drop  https://arxiv.org/abs/1905.09788 
pseudo label
自动padding至batch内最长文本的长度
不同的model parameter，不同的optimizer参数 
优化内存管理