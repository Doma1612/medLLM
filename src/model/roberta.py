import ast
import os.path
from torch import nn
import torch
from transformers import RobertaForSequenceClassification, AutoTokenizer, RobertaConfig


class CustomRobertaForSequenceClassification(nn.Module):
    def __init__(self, pretrained_model_name='FacebookAI/roberta-base', num_labels=10):
        super(CustomRobertaForSequenceClassification, self).__init__()
        self.roberta = RobertaForSequenceClassification(RobertaConfig(num_labels = num_labels))
        self.num_labels = num_labels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.roberta.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return outputs.logits

    def load_model(self, root_dir):
        path = os.path.join(root_dir, "models", "llmrobertasymp.pth")
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location=self.device))
            self.eval()

    @staticmethod
    def top_k_above_threshold(tensor, k=3, threshold=0.6):
        above_threshold_indices = (tensor > threshold).nonzero(as_tuple=True)[1]
        values_above_threshold = tensor[0, above_threshold_indices]
        sorted_values, sorted_indices = torch.sort(values_above_threshold, descending=True)
        num_to_return = min(k, sorted_values.numel())
        result_indices = above_threshold_indices[sorted_indices[:num_to_return]]
        return result_indices.tolist()

    def predict(self, input_text):
        self.eval()
        input_text = ' '.join(input_text)
        print(input_text)
        inputs = self.tokenizer(input_text, padding='max_length', max_length=500, truncation=True, return_tensors='pt')
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = self(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            probs = torch.sigmoid(logits)
            top_k_indices = self.top_k_above_threshold(probs)
            return top_k_indices, probs

#model = CustomRobertaForSequenceClassification()
#model.load_model(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
#model.predict(["Cold, Sneeze, Cough"])