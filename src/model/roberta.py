from torch import nn
import torch
from transformers import RobertaForSequenceClassification

class CustomRobertaForSequenceClassification(nn.Module):
    def __init__(self, pretrained_model_name='FacebookAI/roberta-base', num_labels=10):
        super(CustomRobertaForSequenceClassification, self).__init__()
        self.roberta = RobertaForSequenceClassification.from_pretrained(pretrained_model_name,num_labels=num_labels)
        self.num_labels = num_labels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return outputs.logits

    def load_model(self):
        path = '../../models/llm_roberta_last_state.pth'
        self.load_state_dict(torch.load(path, map_location=torch.device(self.device)))
        self.eval()

    def predict(self, tokenized_text):
        self.load_model()
        with torch.no_grad():
            logits = self(**tokenized_text)
        probs = torch.sigmoid(logits)
        predictions = (probs > 0.8).nonzero(as_tuple=True)[1].tolist()
        return predictions
