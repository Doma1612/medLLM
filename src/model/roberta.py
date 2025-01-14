from torch import nn
from transformers import RobertaForSequenceClassification

class CustomRobertaForSequenceClassification(nn.Module):
    def __init__(self, pretrained_model_name='FacebookAI/roberta-base', num_labels=10):
        super(CustomRobertaForSequenceClassification, self).__init__()
        self.roberta = RobertaForSequenceClassification.from_pretrained(pretrained_model_name,num_labels=num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return outputs.logits