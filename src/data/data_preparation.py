import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import AutoTokenizer
from src.data.dummy_data import DummyData

def prepare_data(device, batch_size=128):
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
    dummy_data = DummyData().get_dummy_data()
    inputs, labels = zip(*dummy_data)

    tokenized_inputs = tokenizer(list(inputs), padding='max_length', max_length=50, return_tensors='pt')
    input_ids = tokenized_inputs['input_ids'].to(device)
    attention_mask = tokenized_inputs['attention_mask'].to(device)
    labels = torch.tensor(labels).to(device)

    dataset = TensorDataset(input_ids, attention_mask, labels)
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])

    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    eval_dataloader = DataLoader(eval_dataset, sampler=RandomSampler(eval_dataset), batch_size=batch_size)

    return train_dataloader, eval_dataloader
