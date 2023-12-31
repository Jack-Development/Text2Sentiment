import torch
import time
import os
from TextClassificationModel import TextClassificationModel
from torchtext import data
from torchtext.datasets import AG_NEWS
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset

root = "./Model"
DATASET_NAME = "IMDB"

def save(input, filename):
	print(f"Saving {filename}...")
	new_root = os.path.join(root, DATASET_NAME)
	if not os.path.exists(new_root):
		os.makedirs(new_root)
	path = os.path.join(new_root, (filename + ".pth"))
	torch.save(input, path)
	print(f"Saved {filename} to {path}")

tokenizer = get_tokenizer("spacy", "en_core_web_sm")

print("Loading dataset...")
train_iter = IMDB(split=("train"))
print("Dataset loaded.")

print("Creating vocabulary...")
train_iter = IMDB(split="train")

def yield_tokens(data_iter):
	for _, text in data_iter:
		yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
print("Created vocabulary.")

save(vocab, "vocab")

##print("Testing the representation of 'Here is an example'...")
##print(vocab(['here', 'is', 'an', 'example']))

print("Preparing pipelines...")
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1
##print("Testing pipeline...")
##print(f"Expected: {vocab(['here', 'is', 'an', 'example'])}, Return: {text_pipeline('here is an example')}")
##assert vocab(['here', 'is', 'an', 'example']) == text_pipeline("here is an example")
##print(f"Expected: 9, Reutrn: {label_pipeline('10')}")
##assert label_pipeline("10") == 9
print("Success! Pipeline created.")

device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
)

def collate_batch(batch):
	label_list, text_list, offsets = [], [], [0]
	for _label, _text in batch:
		label_list.append(label_pipeline(_label))
		processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
		text_list.append(processed_text)
		offsets.append(processed_text.size(0))
	label_list = torch.tensor(label_list, dtype=torch.int64)
	offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
	text_list = torch.cat(text_list)
	return label_list.to(device), text_list.to(device), offsets.to(device)

train_iter = IMDB(split="train")
dataloader = DataLoader(train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch)

train_iter = IMDB(split="train")
num_class = len(set([label for (label, _) in train_iter]))
vocab_size = len(vocab)
emsize = 64
model = TextClassificationModel(vocab_size, emsize, num_class).to(device)

def train(dataloader):
	model.train()
	total_acc, total_count = 0, 0
	log_interval = 100
	start_time = time.time()

	for idx, (label, text, offsets) in enumerate(dataloader):
		optimizer.zero_grad()
		predicted_label = model(text, offsets)
		loss = criterion(predicted_label, label)
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
		optimizer.step()
		total_acc += (predicted_label.argmax(1) == label).sum().item()
		total_count += label.size(0)
		if idx % log_interval == 0 and idx > 0:
			elapsed = time.time() - start_time
			print(
			f"| Epoch {epoch:>3d} | {idx:>5d}/{len(dataloader):>5d} | Accuracy: {(total_acc/total_count):>8.3f}"
			)
			total_acc, total_count = 0, 0
			start_time = time.time()

def evaluate(dataloader):
	model.eval()
	total_acc, total_count = 0, 0

	with torch.no_grad():
		for idx, (label, text, offsets) in enumerate(dataloader):
			predicted_label = model(text, offsets)
			loss = criterion(predicted_label, label)
			total_acc += (predicted_label.argmax(1) == label).sum().item()
			total_count += label.size(0)
	return total_acc / total_count

## TRAIN

EPOCHS = 10
LR = 5
BATCH_SIZE = 64

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None
train_iter, test_iter = IMDB()
train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)
num_train = int(len(train_dataset) * 0.7)
split_train_, split_valid_ = random_split(
	train_dataset, [num_train, len(train_dataset) - num_train]
)

train_dataloader = DataLoader(
	split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)
valid_dataloader = DataLoader(
	split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)
test_dataloader = DataLoader(
	test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch
)

for epoch in range(1, EPOCHS+1):
	epoch_start_time = time.time()
	train(train_dataloader)
	accu_val = evaluate(valid_dataloader)
	if total_accu is not None and total_accu > accu_val:
		scheduler.step()
	else:
		total_accu = accu_val
	print("-" * 59)
	print(
	f"| End of Epoch {epoch:>3d} | Time: {(time.time() - epoch_start_time):>5.2f}s | Valid Accuracy: {accu_val:>8.3f}"
	)
	print("-" * 59)

## TEST

print("Checking result of test dataset...")
accu_test = evaluate(test_dataloader)
print(f"Test Accuracy: {accu_test:>8.3f}")

## SAVE

save(model.state_dict(), "model_state_dict")
save(model, "model")
