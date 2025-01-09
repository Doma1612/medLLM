import torch
from torch.utils.data import DataLoader, RandomSampler
from torch import nn, optim
from transformers import RobertaForSequenceClassification, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")

device = 'cuda'

dummydata = [
    ["Cold, Sneeze, Cough", [1, 2, 3]],
    ["Stomach Hurt, Headache", [4, 5, 6]],
    ["Low blood pressure, high heartbeatrate", [2, 5, 8]],
    ["Fainting, low oxygen", [1, 8, 9]],
    ["Hurting leg, fainting, high thirst", [4, 7, 8]],
    ["Nausea, Vomiting", [0, 6, 9]],
    ["High fever, muscle pain", [2, 4, 9]],
    ["Chest pain, shortness of breath", [1, 5, 7]],
    ["Dizziness, lightheadedness", [3, 6, 8]],
    ["Joint pain, fatigue", [5, 6, 7]],
    ["Rash, itching", [4, 8, 9]],
    ["Earache, fever", [1, 2, 5]],
    ["Back pain, discomfort", [3, 4, 6]],
    ["Dry throat, difficulty swallowing", [2, 5, 9]],
    ["Weight loss, increased thirst", [1, 7, 8]],
    ["Mood swings, insomnia", [3, 5, 9]],
    ["Sore throat, fatigue", [2, 6, 9]],
    ["Swelling, bruising", [1, 3, 8]],
    ["Blurred vision, headache", [4, 5, 9]],
    ["Cold extremities, shivering", [2, 6, 8]]
]

dummyset=[]
for data in dummydata:
    inp = tokenizer(data[0],padding='max_length', max_length=50)
    outp = data[1]
    a = [*[torch.tensor(v).to(device) for k,v in inp.items()], torch.tensor(outp).to(device)]
    dummyset.append(a)

traindata, evaldata = torch.utils.data.random_split(dummyset,[16,4])

train_sampler = RandomSampler(traindata)
train_dataloader = DataLoader(traindata, sampler=train_sampler, batch_size=1)
eval_sampler = RandomSampler(evaldata)
eval_dataloader = DataLoader(evaldata, sampler=eval_sampler, batch_size=1)


class CustomRobertaForSequenceClassification(nn.Module):
    def __init__(self, pretrained_model_name='FacebookAI/roberta-base', num_labels=10):
        super(CustomRobertaForSequenceClassification, self).__init__()
        self.roberta = RobertaForSequenceClassification.from_pretrained(pretrained_model_name,num_labels=num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        return outputs.logits
    
model = CustomRobertaForSequenceClassification().to(device)

def train_epoch(dataloader, encoder, optimizer, criterion):
    for data in dataloader:
        inp, att, out = data

        optimizer.zero_grad()

        logits = encoder(inp, attention_mask = att)

        targets = torch.zeros(logits.size(), dtype=torch.float32).to(device)
        targets[0, out] = 1.0
        loss = criterion(logits, targets)
        loss.backward()

        optimizer.step()
        
    return loss.item()

def test_epoch(dataloader, encoder,  criterion):
    for data in dataloader:
        inp, att, out = data

        logits = encoder(inp, attention_mask = att)

        targets = torch.zeros(logits.size(), dtype=torch.float32).to(device)
        targets[0, out] = 1.0
        loss = criterion(logits, targets)
        
    return loss.item()

def train(train_dataloader, eval_dataloader, encoder, epochs, lr = 0.0001):
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    encoder.train()
    for epoch in range(epochs):
        loss = train_epoch(train_dataloader, encoder, encoder_optimizer,criterion)
        test_loss = test_epoch(eval_dataloader,encoder,criterion)

        print('Epoch:',epoch + 1,',Loss:',loss,',Testloss', test_loss)

train(train_dataloader,eval_dataloader,model,10)


def eval(input):
    model.eval()

    input = tokenizer(input,padding='max_length', max_length=50)

    inp = torch.tensor(input['input_ids']).to(device).unsqueeze(0)
    att = torch.tensor(input['attention_mask']).to(device).unsqueeze(0)

    out = model(inp, attention_mask = att)

    out = torch.sigmoid(out)
    #print('Probs: ', out)
    out = out > 0.8
    out = torch.nonzero(out, as_tuple=True)

    papers = out[1].tolist()
    print(papers)

eval("Stomach Hurt, Headache")