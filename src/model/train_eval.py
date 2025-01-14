import torch
from torch import optim, nn
import time
from src.utils import timeSince

def train_epoch(dataloader, encoder, optimizer, criterion, device):
    total_loss = 0
    encoder.train()
    for input_ids, attention_mask, labels in dataloader:
        optimizer.zero_grad()
        logits = encoder(input_ids=input_ids, attention_mask=attention_mask)
        targets = torch.zeros(logits.size(), dtype=torch.float32).to(device)
        targets[0, labels] = 1.0
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def test_epoch(dataloader, encoder, criterion, device):
    total_loss = 0
    encoder.eval()
    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            logits = encoder(input_ids=input_ids, attention_mask=attention_mask)
            targets = torch.zeros(logits.size(), dtype=torch.float32).to(device)
            targets[0, labels] = 1.0
            loss = criterion(logits, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def train(train_dataloader, eval_dataloader, encoder, epochs, lr=0.0001, print_every=1, device='cpu'):
    print_loss_total = 0
    test_loss_total = 0
    min_loss = float('inf')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    _, _, fb = next(iter(train_dataloader))
    pos_weight = torch.tensor([(encoder.num_labels - len(fb[0])) / len(fb[0])]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    start = time.time()

    for epoch in range(1, epochs + 1):
        print("Epoch: " + str(epoch))
        loss = train_epoch(train_dataloader, encoder, encoder_optimizer, criterion, device)
        test_loss = test_epoch(eval_dataloader, encoder, criterion, device)
        print_loss_total += loss
        test_loss_total += test_loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            test_loss_avg = test_loss_total / print_every
            test_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / epochs), epoch, epoch / epochs * 100, print_loss_avg))

        if test_loss_avg < min_loss:
            min_loss = test_loss_avg
            torch.save(encoder.state_dict(), '../../models/llm_roberta_best_current_state.pth')
            print("Test Loss: " + str(test_loss_avg) + "\n")

    torch.save(encoder.state_dict(), '../../models/llm_roberta_last_state.pth')