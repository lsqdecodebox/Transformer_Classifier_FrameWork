import logging
from transformers.modeling_bert import BertPreTrainedModel
from transformers import (
    BertTokenizer,
    BertModel,
    BertForSequenceClassification,
    BertConfig,
    AdamW,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
import torch
import torch.nn.functional as F
from torch import nn

logging.getLogger("transformers").setLevel(logging.ERROR)


class Squeeze(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)

# 自制tensor size 打印
def print_tensor_shape(target,depth = 0):
    if isinstance(target,tuple):
        print('--' * depth, '>', 'tuple len:', len(target))
        for t in target:
            print_tensor_shape(t,depth=depth+1)
    elif isinstance(target,torch.Tensor):
        print('--'*depth, '>', target.shape)
    else:
        print('unknown type',type(target))


class CustomBert(BertPreTrainedModel):  # 重写
    def __init__(self, config):
        config.output_hidden_states = True
        super(CustomBert, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(p=0.2)
        self.high_dropout = nn.Dropout(p=0.5)

        n_weights = config.num_hidden_layers + 1
        weights_init = torch.zeros(n_weights).float()
        weights_init[:-1] = -3   # 咋想的
        self.layer_weights = torch.nn.Parameter(weights_init)
        # print('weights_init',weights_init)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_layers = outputs[2]
        last_hidden = outputs[0]
        # print(hidden_layers)
        cls_outputs = torch.stack(   # torch.Size([7, 768, 13])
            [self.dropout(layer[:, 0, :]) for layer in hidden_layers],
            dim=2
        )

        cls_output = (      # torch.Size([7, 768])
            torch.softmax(self.layer_weights, dim=0) * cls_outputs
        ).sum(-1)

        # multisample dropout (wut): https://arxiv.org/abs/1905.09788   论文很到位
        logits = torch.mean(torch.stack([   # torch.Size([7, 30])
            self.classifier(self.high_dropout(cls_output))
            for _ in range(5)
        ], dim=0), dim=0)

        outputs = logits

        return outputs


def get_model_optimizer(args):
    model = CustomBert.from_pretrained(args.bert_model, num_labels=args.num_classes) # 30个类别输出
    if args.is_cuda:
        model.cuda()
        model = nn.DataParallel(model)  # 放置多核GPU
    params = list(model.named_parameters())

    def is_backbone(n):
        return "bert" in n

    optimizer_grouped_parameters = [
        {'params': [p for n, p in params if is_backbone(n)],
         'lr': args.lr},
        {'params': [p for n, p in params if not is_backbone(n)],
         'lr': args.lr * 500}
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.lr,
        weight_decay=0
    )

    return model, optimizer
