import torch
import os
from TextClassificationModel import TextClassificationModel
from torchtext.data.utils import get_tokenizer

root = "./Model"
DATASET_NAME = "IMDB"

def load(filename, filetype=".pth"):
	path = os.path.join(root, DATASET_NAME, (filename + filetype))
	print(f"Loading {path}...")
	result = torch.load(path)
	print(f"Loaded {filename}.")
	return result

device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
)

## PREDICT

model = load("model")
model.eval()

tokenizer = get_tokenizer("spacy", "en_core_web_sm")
vocab = load("vocab")

text_pipeline = lambda x: vocab(tokenizer(x))

review_label = {1: "Negative", 2: "Positive"}
def predict(text, text_pipeline):
	with torch.no_grad():
		text = torch.tensor(text_pipeline(text)).to(device)
		output = model(text, torch.tensor([0]).to(device))
		return output.argmax(1).item() + 1

model.to(device)

def run(ex_string):
	print(ex_string)
	result = predict(ex_string, text_pipeline)
	print(f"This is a {review_label[result]} review.")
	return result

run("Wow! I loved this movie so much! It was the best!")
run("I hated this movie! It was the absolute worst!")
