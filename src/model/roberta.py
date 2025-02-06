import math
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch import nn, optim
from transformers import RobertaForSequenceClassification, AutoTokenizer
import time

batch_size = 16
num_labels=128836

savedataset = False # True if you want to save it. Disclaimer they are each 5GB large

tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

homedir = ''#Define your Home Directory

print("Start Data transformation")

dataset = torch.load(homedir+'/reduceddata.pt')
print(len(dataset))

def createset():
    tokenized_dataset=[]
    for data in dataset:
        inp = tokenizer(data[0],padding='max_length', max_length=512,truncation=True)#=3589)
        outp = data[1]
        try: 
            a = [*[torch.tensor(v).to(device) for k,v in inp.items()], torch.tensor(outp).to(device)]
        except:
            continue
        else:
            outlogits = torch.zeros(num_labels, dtype=torch.float32).to(device)
            outlogits[outp] = 1.0
            a = [*[torch.tensor(v).to(device) for k,v in inp.items()], outlogits]
        tokenized_dataset.append(a)

    if savedataset:
        torch.save(tokenized_dataset, "tokenizeddata.pt")
    return tokenized_dataset

tokenized_dataset = createset()
#dummyset = torch.load(homedir+"/tokenizeddata.pt")

trainsize = int(len(tokenized_dataset)*0.8)
testsize = len(tokenized_dataset) - trainsize
traindata, evaldata = torch.utils.data.random_split(tokenized_dataset,[trainsize,testsize])

def createdataloader(traindata,evaldata):
    train_sampler = RandomSampler(traindata)
    train_dataloader = DataLoader(traindata, sampler=train_sampler, batch_size=batch_size)
    eval_sampler = RandomSampler(evaldata)
    eval_dataloader = DataLoader(evaldata, sampler=eval_sampler, batch_size=batch_size)
    
    if savedataset:
        torch.save(train_dataloader, "train_dataloader.pt")
        torch.save(eval_dataloader, "eval_dataloader.pt")
    return train_dataloader,eval_dataloader

train_dataloader,eval_dataloader = createdataloader(traindata,evaldata)
# train_dataloader = torch.load(homedir+"/train_dataloader.pt")
# eval_dataloader = torch.load(homedir+"/eval_dataloader.pt")




class CustomRobertaForSequenceClassification(nn.Module):
    def __init__(self, pretrained_model_name='FacebookAI/roberta-base', num_labels=10):
        super(CustomRobertaForSequenceClassification, self).__init__()
        self.roberta = RobertaForSequenceClassification.from_pretrained(pretrained_model_name,num_labels=num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        return outputs.logits

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
    
model = CustomRobertaForSequenceClassification(num_labels=128836).to(device)

def train_epoch(dataloader, encoder, optimizer, criterion):
    i=0

    total_loss = 0
    
    for data in dataloader:
        inp, att, out = data
    

        optimizer.zero_grad()

        logits = encoder(inp, attention_mask = att)
        loss = criterion(logits, out)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        i += 1
        if i % 10 == 0:
            print(i*batch_size, "von", len(traindata))

        
    return total_loss / len(dataloader)

def test_epoch(dataloader, encoder,  criterion):
    total_loss = 0

    for data in dataloader:
        inp, att, out = data

        logits = encoder(inp, attention_mask = att)

        loss = criterion(logits, out)

        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def train(train_dataloader, eval_dataloader, encoder, epochs, lr = 0.0001, print_every=1):
    print_loss_total = 0
    test_loss_total = 0
    min_loss = 100
    
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)

    #_ ,_, fb = next(iter(train_dataloader))
    #pos_weight = torch.tensor([(encoder.num_labels - len(fb[0])) / len(fb[0])]).to(device)
    pos_weight = torch.tensor([(encoder.num_labels - 18) / 18]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    encoder.train()

    start = time.time()

    for epoch in range(1, epochs+1):
        loss = train_epoch(train_dataloader, encoder, encoder_optimizer,criterion)
        test_loss = test_epoch(eval_dataloader,encoder,criterion)

        print_loss_total += loss
        test_loss_total += test_loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            test_loss_avg = test_loss_total / print_every
            test_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / epochs),
                                        epoch, epoch / epochs * 100, print_loss_avg))

        if test_loss_avg < min_loss:
            min_loss = test_loss_avg
            torch.save(model.state_dict(), homedir+'/llmroberta.pth')
            print("Test Loss: " + str(test_loss_avg) + "\n")

    torch.save(model.state_dict(), homedir+'/llmrobertalaststate.pth')

        

print("Start Training")
train(train_dataloader,eval_dataloader,model,10, lr = 0.001)

model.load_state_dict(torch.load(homedir+"/llmrobertalaststate.pth"))



def top_k_above_threshold(tensor, k=2, threshold=0.4):
    above_threshold_indices = (tensor > threshold).nonzero(as_tuple=True)[1]
    values_above_threshold = tensor[0, above_threshold_indices]
    sorted_values, sorted_indices = torch.sort(values_above_threshold, descending=True)
    num_to_return = min(k, sorted_values.numel())
    result_indices = above_threshold_indices[sorted_indices[:num_to_return]]
    
    return result_indices.tolist()

def eval(input):
    model.eval()

    input = tokenizer(input,padding='max_length', max_length=50)

    inp = torch.tensor(input['input_ids']).to(device).unsqueeze(0)
    att = torch.tensor(input['attention_mask']).to(device).unsqueeze(0)

    out = model(inp, attention_mask = att)

    out = torch.sigmoid(out)
    print(out)
    print(top_k_above_threshold(out))

eval("Cold, Sneeze, Cough")
