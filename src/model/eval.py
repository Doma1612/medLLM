import torch
from transformers import AutoTokenizer
from src.data.data_preparation import prepare_data
from src.model.roberta import CustomRobertaForSequenceClassification
from src.model.train_eval import train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 128

print("Start Data transformation")
train_dataloader, eval_dataloader = prepare_data(device, batch_size)

model = CustomRobertaForSequenceClassification(num_labels=20000).to(device)

print("Start Training")
train(train_dataloader, eval_dataloader, model, 10, lr=0.001, device=device)

def eval(input_text):
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
    inputs = tokenizer(input_text, padding='max_length', max_length=50, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs)
    probs = torch.sigmoid(logits)
    predictions = (probs > 0.8).nonzero(as_tuple=True)[1].tolist()
    print('Predicted Labels:', predictions)

eval("Cold, Sneeze, Cough")