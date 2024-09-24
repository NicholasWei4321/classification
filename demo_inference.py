from transformers import AutoTokenizer, AutoConfig, RobertaForSequenceClassification
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
from datasets import load_from_disk
from tqdm import tqdm
from typing import Dict, List, Tuple

# text_model = RobertaForSequenceClassification.from_pretrained('/media/ken/sentiment/model/sentiment_classifier')
# tokenizer = AutoTokenizer.from_pretrained("/media/ken/sentiment/model/sentiment_classifier")
# config =  AutoConfig.from_pretrained("/media/ken/sentiment/model/sentiment_classifier")

# audio_model = HubertForSequenceClassification.from_pretrained("superb/hubert-base-superb-er")
# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-base-superb-er")

import torch
from torch import nn
from transformers import RobertaForSequenceClassification, HubertForSequenceClassification

import numpy as np

def f1_score(true_labels, predicted_labels):
    # Convert lists to NumPy arrays
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Calculate true positives, false positives, and false negatives for each class
    classes = np.unique(true_labels)
    tp = np.zeros(len(classes))
    fp = np.zeros(len(classes))
    fn = np.zeros(len(classes))

    for i, c in enumerate(classes):
        tp[i] = np.sum((true_labels == c) & (predicted_labels == c))
        fp[i] = np.sum((true_labels != c) & (predicted_labels == c))
        fn[i] = np.sum((true_labels == c) & (predicted_labels != c))

    # Calculate precision, recall, and F1 score for each class
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Calculate weighted average of F1 scores based on class frequencies
    class_counts = np.bincount(true_labels)
    class_weights = class_counts / len(true_labels)
    weighted_f1_score = np.sum(f1_score * class_weights)

    return weighted_f1_score, precision, recall, f1_score

import pickle

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_attention_heads)

    def forward(self, input):
        # print(input)
        # Permute input dimensions to (seq_len, batch_size, hidden_size)
        input = input.permute(1, 0, 2)
        
        # Apply self-attention
        output, _ = self.attention(input, input, input)
        
        # Permute output dimensions back to (batch_size, seq_len, hidden_size)
        output = output.permute(1, 0, 2)
        
        return output

class ConcatModel(torch.nn.Module):
    def __init__(self):
        super(ConcatModel, self).__init__()
        self.text_model = RobertaForSequenceClassification.from_pretrained('/nfs/nicholas/data/sentiment_classifier')
        # self.text_model = RobertaForSequenceClassification.from_pretrained("roberta-base")
        self.audio_model = HubertForSequenceClassification.from_pretrained('superb/hubert-base-superb-er')
        self.audio_model.load_state_dict(torch.load("/nfs/nicholas/outputs/improved_audio2/epoch4/hubert_switchboard.pth"))
        self.concat_fc = nn.Linear(self.text_model.config.hidden_size, 3)
        self.text_layer_norm = nn.LayerNorm(self.text_model.config.hidden_size)
        self.audio_layer_norm = nn.LayerNorm(self.audio_model.config.hidden_size)

        # self.text_projection = nn.Linear(self.text_model.config.hidden_size, self.text_model.config.hidden_size)
        # self.audio_projection = nn.Linear(self.audio_model.config.hidden_size, self.audio_model.config.hidden_size)

        # self.dropout = nn.Dropout(0.2)
        # self.concat_fc = nn.Linear(self.audio_model.config.hidden_size, 64)
        # self.relu = nn.ReLU()
        # self.concat_fc2 = nn.Linear(512, 3)
        # self.concat_fc2 = nn.Linear(64, 3)
        self.softmax = nn.Softmax(dim=1)
        self.attention = SelfAttention(self.audio_model.config.hidden_size, num_attention_heads=8)
        
    def forward(self, text_input, audio_input):
        with torch.no_grad():
            text_output = self.text_model(**text_input, output_hidden_states=True)
            text_hidden = text_output.hidden_states[-1][:, :, :]
            # text_hidden = self.dropout(text_hidden)
            text_hidden = self.text_layer_norm(text_hidden)
            # text_hidden = self.text_projection(text_hidden)
        
            audio_output = self.audio_model(**audio_input, output_hidden_states=True)
            audio_hidden = audio_output.hidden_states[-1][:, :, :]
            audio_hidden = self.text_layer_norm(audio_hidden)
            # audio_hidden = self.audio_projection(audio_hidden)

            # text_hidden = nn.BatchNorm1d(text_hidden.size(1))(text_hidden)
            # audio_hidden = nn.BatchNorm1d(audio_hidden.size(1))(audio_hidden) -- layer norm

        
        # print(text_hidden.shape)
        # print(audio_hidden.shape)
        concatenated_hidden = torch.cat((text_hidden, audio_hidden), dim=1)
        attn_concatenated_hidden = self.attention(concatenated_hidden)
        concatenated_hidden = attn_concatenated_hidden + concatenated_hidden

        logits = self.concat_fc(concatenated_hidden[:, 0, :])  # Logits without softmax
        # logits = self.concat_fc(audio_hidden)  # Logits without softmax
        # logits = self.relu(logits)
        # logits = self.concat_fc2(logits)
        probabilities = self.softmax(logits)  # Apply softmax to obtain probabilities

        return probabilities

from torch.utils.data import Dataset
class TrainDataset(Dataset):  # type: ignore
    def __init__(
            self, 
            queries: List[str], 
            labels: List[int], 
            tokenizer
        ) -> None:
        # store your raw data
        self.queries: List[str] = queries
        self.labels: List[int] = labels
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int):
        # from raw to features/numbers/tensors
        tokenized_query = self.tokenizer(self.queries[idx], truncation=True)
        return {**tokenized_query, "labels": torch.tensor(self.labels[idx])}
    
class TrainAudioDataset(Dataset):  # type: ignore
    def __init__(
            self, 
            audio_dataset,
            labels, 
            extractor
        ) -> None:
        # store your raw data
        self.audio_dataset: List[str] = audio_dataset
        self.labels: List[int] = labels
        self.feature_extractor = extractor # Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-base-superb-er")

    def __len__(self) -> int:
        return self.audio_dataset.num_rows

    def __getitem__(self, idx: int):
        # from raw to features/numbers/tensors
        # print(self.audio_dataset)
        tokenized_query = self.feature_extractor(self.audio_dataset[idx]["speech"], sampling_rate=16000, padding=True, return_tensors="pt")
        return {**tokenized_query, "labels": torch.tensor(self.labels[idx])}
    
class TrainTextDataset(Dataset):  # type: ignore
    def __init__(
            self, 
            text_dataset,
            labels: List[int], 
            tokenizer
        ) -> None:
        # store your raw data
        self.text_dataset: List[str] = text_dataset
        self.labels: List[int] = labels
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return self.text_dataset.num_rows

    def __getitem__(self, idx: int):
        # from raw to features/numbers/tensors
        tokenized_query = self.tokenizer(self.text_dataset[idx]["text"], truncation=True)
        return {**tokenized_query, "labels": torch.tensor(self.labels[idx])}
    
sentiment_to_id = {"Negative": 0, "Neutral": 1, "Positive": 2}
# Mapping Labels to IDs
def map_label2id(example):
    example['sentiment'] = sentiment_to_id[example['sentiment']]
    return example

def _create_train_audio_dataset(
        dataset,
        feature_extractor
    ) -> Tuple[TrainDataset, int]:
    # dataset is either train, val, or test
    
    audio_dataset = load_from_disk(f"/nfs/nicholas/demo_content/{dataset}/zoomiq_formatted_audio")
    audio_dataset = audio_dataset.map(map_label2id)

    train_dataset = TrainAudioDataset(
        audio_dataset,
        audio_dataset["sentiment"],
        feature_extractor
    )
    return train_dataset, audio_dataset.num_rows

def _create_train_text_dataset(
        dataset,
        tokenizer,
    ) -> Tuple[TrainDataset, int]:
    # dataset is either train, val, or test
    
    text_dataset = load_from_disk(f"/nfs/nicholas/demo_content/{dataset}/zoomiq_formatted_text")
    text_dataset = text_dataset.map(map_label2id)

    train_dataset = TrainTextDataset(
        text_dataset,
        text_dataset["sentiment"],
        tokenizer
    )
    return train_dataset, text_dataset.num_rows

class CustomAudioDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        # print(features[0]["input_values"].shape)
        max_length = max([feature["input_values"].shape[1] for feature in features])
        # print(max_length)
        input_values = [torch.tensor(self._pad_sequence(feature["input_values"].flatten().tolist(), max_length), dtype=torch.float) for feature in features]
        input_values = torch.stack(input_values)
        # print("Done stacking input values")
        # Get the attention_mask for the padded input_values
        attention_mask = torch.ones(input_values.shape, dtype=torch.long)
        attention_mask[input_values == self.processor.padding_value] = 0
        # print("Done padding attention masks")

        batch = {"input_values": input_values, "attention_mask": attention_mask}

        # If labels are available, pad them as well
        # if "labels" in features[0]:
        #     max_label_length = max([len(feature["labels"]) for feature in features])
        #     labels = [torch.tensor(self._pad_sequence(feature["labels"], max_label_length), dtype=torch.long) for feature in features]
        #     labels = torch.stack(labels)
        #     batch["labels"] = labels

        return batch

    def _pad_sequence(self, sequence, max_length):
        # print("Padding Sequences")
        pad_length = max_length - len(sequence)
        padded_sequence = sequence + [self.processor.padding_value] * pad_length
        return padded_sequence

def clear_batch_from_cuda(batch_inputs):
    for key, value in batch_inputs.items():
        if torch.is_tensor(value) and value.is_cuda:
            batch_inputs[key] = value.cpu()
    torch.cuda.empty_cache()

import torch
from torch import nn, optim
from transformers import RobertaForSequenceClassification, HubertForSequenceClassification, DataCollatorWithPadding
from zoomiq_nlu.intent_detection.utils import set_random_seed
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.optim.lr_scheduler import CosineAnnealingLR

# batch_size = 16
# num_epochs = 100
# # Step 1: Prepare your training data

# # Step 2: Initialize the model
# model = ConcatModel()
# # model.load_state_dict(torch.load("/nfs/nicholas/outputs/betterhubert_epoch1/betterhubert_lr_5e-5_batch16_selfattn_epoch1.pth"))

# # Step 3: Define the loss function
# criterion = nn.CrossEntropyLoss(
#     label_smoothing=0.1
# )

# # Step 4: Configure the optimizer
# optimizer = optim.Adam(model.parameters(), lr=0.00005)

# # Step 5: Perform the training loop

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # device = "cpu"
# model.to(device)

# set_random_seed.set_random_seed(42)

# text_tokenizer = AutoTokenizer.from_pretrained("/nfs/nicholas/data/sentiment_classifier")
# # text_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
# audio_tokenizer = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-base-superb-er")

# # create training datasets
# train_text_dataset, num_labels = _create_train_text_dataset("train", text_tokenizer)
# train_audio_dataset, num_labels = _create_train_audio_dataset("train", audio_tokenizer)
# dataset_size = num_labels
# indices = list(range(dataset_size))
# train_sampler1 = SubsetRandomSampler(indices, generator=torch.Generator().manual_seed(42))
# train_sampler2 = SubsetRandomSampler(indices, generator=torch.Generator().manual_seed(42))
# train_data_collator1 = DataCollatorWithPadding(tokenizer=text_tokenizer)
# train_data_collator2 = CustomAudioDataCollator(processor=audio_tokenizer)
# train_dataloader_text = DataLoader(train_text_dataset, batch_size=batch_size, sampler=train_sampler1, collate_fn=train_data_collator1)
# train_dataloader_audio = DataLoader(train_audio_dataset, batch_size=batch_size, sampler=train_sampler2, collate_fn=train_data_collator2)


# # create validation datasets
# val_text_dataset, num_labels = _create_train_text_dataset("val", text_tokenizer)
# val_audio_dataset, num_labels = _create_train_audio_dataset("val", audio_tokenizer)
# dataset_size = num_labels
# val_indices = list(range(dataset_size))
# val_sampler1 = SubsetRandomSampler(val_indices, generator=torch.Generator().manual_seed(42))
# val_sampler2 = SubsetRandomSampler(val_indices, generator=torch.Generator().manual_seed(42))
# val_data_collator1 = DataCollatorWithPadding(tokenizer=text_tokenizer)
# val_data_collator2 = CustomAudioDataCollator(processor=audio_tokenizer)
# val_dataloader_text = DataLoader(val_text_dataset, batch_size=batch_size//4, sampler=val_sampler1, collate_fn=val_data_collator1)
# val_dataloader_audio = DataLoader(val_audio_dataset, batch_size=batch_size//4, sampler=val_sampler2, collate_fn=val_data_collator2)

# scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_dataloader_text), eta_min=0) # check huggingface / glue

# losses = []
# epochlosses = []

# val_epochlosses = []

# evaluations = []

# best_val_loss = float('inf')
# patience = 0
# max_patience = 3

# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0

#     progress_bar = tqdm(zip(train_dataloader_text, train_dataloader_audio), desc=f'Epoch {epoch+1}', leave=False, total=len(train_dataloader_text))
#     for text_batch, audio_batch in progress_bar:
#         # Move data to the appropriate device
#         text_batch["input_ids"] = text_batch["input_ids"].to(device)
#         audio_batch["input_values"] = audio_batch["input_values"].to(device)
#         text_batch["attention_mask"] = text_batch["attention_mask"].to(device)
#         audio_batch["attention_mask"] = audio_batch["attention_mask"].to(device)
#         text_batch["labels"] = text_batch["labels"].to(device)
#         # print(text_batch["input_ids"].shape)

#         optimizer.zero_grad()

#         outputs = model(text_batch, audio_batch)
#         loss = criterion(outputs, text_batch["labels"])
#         loss.backward()

#         optimizer.step()
#         scheduler.step()

#         running_loss += loss.item()
#         losses.append(loss.item())
#         progress_bar.set_postfix({'Loss': f"{loss.item():.4f}"})

#         clear_batch_from_cuda(text_batch)
#         clear_batch_from_cuda(audio_batch)
#         # torch.cuda.empty_cache()

#     # Print average loss for the epoch
#     epoch_loss = running_loss / len(train_dataloader_text)
#     print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")
#     epochlosses.append(epoch_loss)

#     # validation
#     model.eval()
#     val_losses = []
#     val_running_loss = 0.0
#     preds = []
#     eval_labels = []

#     progress_bar = tqdm(zip(val_dataloader_text, val_dataloader_audio), desc=f'Epoch {epoch+1}', leave=False, total=len(val_dataloader_text))
#     for text_batch, audio_batch in progress_bar:
#         with torch.no_grad():
#             # Move data to the appropriate device
#             text_batch["input_ids"] = text_batch["input_ids"].to(device)
#             audio_batch["input_values"] = audio_batch["input_values"].to(device)
#             text_batch["attention_mask"] = text_batch["attention_mask"].to(device)
#             audio_batch["attention_mask"] = audio_batch["attention_mask"].to(device)
#             text_batch["labels"] = text_batch["labels"].to(device)

#             outputs = model(text_batch, audio_batch)
#             loss = criterion(outputs, text_batch["labels"])

#             val_running_loss += loss.item()
#             val_losses.append(loss.item())
#             progress_bar.set_postfix({'Val Loss': f"{loss.item():.4f}"})

#             _, y_hat = outputs.max(1)
#             preds += [idx.item() for idx in y_hat]
#             eval_labels += text_batch["labels"].tolist()

#         clear_batch_from_cuda(text_batch)
#         clear_batch_from_cuda(audio_batch)
#         # torch.cuda.empty_cache()

#     eval_accuracy = np.mean(np.array(preds) == np.array(eval_labels))
#     f1 = f1_score(eval_labels, preds)
#     evaluations.append((eval_accuracy, f1))

#     # Print average loss for the epoch
#     val_epoch_loss = val_running_loss / len(val_dataloader_audio)
#     print(f"Epoch {epoch+1} Val Loss: {val_epoch_loss:.4f}")
#     print(f"Accuracy: {eval_accuracy} | F1: {f1}")
#     val_epochlosses.append(val_epoch_loss)

#     np.save("/nfs/nicholas/outputs/newest_betterhubert/train_losses_hubert_switchboard", np.array(losses)) 
#     np.save("/nfs/nicholas/outputs/newest_betterhubert/train_epochlosses_hubert_switchboard", np.array(epochlosses))
#     np.save("/nfs/nicholas/outputs/newest_betterhubert/val_losses_hubert_switchboard", np.array(val_losses)) 
#     np.save("/nfs/nicholas/outputs/newest_betterhubert/val_epochlosses_hubert_switchboard", np.array(val_epochlosses)) 
#     np.save("/nfs/nicholas/outputs/newest_betterhubert/f1_hubert_switchboard", np.array(evaluations)) 

#     if val_epoch_loss < best_val_loss:
#         best_val_loss = val_epoch_loss
#         patience = 0
#         torch.save(model.state_dict(), "/nfs/nicholas/outputs/newest_betterhubert/hubert_switchboard.pth")

#     else:
#         patience += 1
#         if patience >= max_patience:
#             print("Early stopping triggered. No improvement in validation loss.")
#             break
    
#     print(patience)

# Step 6: Evaluate the model on a validation or test set

# torch.save(model.state_dict(), "/nfs/nicholas/outputs/betterhubert_lr_5e-5_batch16_selfattn_epoch2.pth")
# np.save("/nfs/nicholas/outputs/losses_betterhubert_lr_5e-5_batch16_selfattn_epoch2", np.array(losses)) 
# np.save("/nfs/nicholas/outputs/epochlosses_betterhubert_lr_5e-5_batch16_selfattn_epoch2", np.array(epochlosses)) 


batch_size = 16

text_tokenizer = AutoTokenizer.from_pretrained("/nfs/nicholas/data/sentiment_classifier")
audio_tokenizer = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-base-superb-er")

test_text_dataset, num_labels = _create_train_text_dataset("demo_meeting_8_1_2023", text_tokenizer)
test_audio_dataset, num_labels = _create_train_audio_dataset("demo_meeting_8_1_2023", audio_tokenizer)

dataset_size = num_labels
indices = list(range(dataset_size))
test_sampler1 = SubsetRandomSampler(indices, generator=torch.Generator().manual_seed(42))
test_sampler2 = SubsetRandomSampler(indices, generator=torch.Generator().manual_seed(42))

# sampler1 = SubsetRandomSampler(indices)

test_data_collator1 = DataCollatorWithPadding(tokenizer=text_tokenizer)
test_data_collator2 = CustomAudioDataCollator(processor=audio_tokenizer)

# test_dataloader_text = DataLoader(test_text_dataset, batch_size=batch_size, sampler=test_sampler1, collate_fn=test_data_collator1)
# test_dataloader_audio = DataLoader(test_audio_dataset, batch_size=batch_size, sampler=test_sampler2, collate_fn=test_data_collator2)

test_dataloader_text = DataLoader(test_text_dataset, batch_size=batch_size,collate_fn=test_data_collator1,shuffle=False)
test_dataloader_audio = DataLoader(test_audio_dataset, batch_size=batch_size,collate_fn=test_data_collator2, shuffle=False)

import numpy as np
def evaluate(
    model, 
    text_model,
    audio_model,
    eval_text_dataloader,
    eval_audio_dataloader
):
    model.eval()

    preds: List[int] = []
    preds_text = []
    preds_audio = []

    eval_labels: List[int] = []
    all_logits = []
    all_logits_audio = []
    all_logits_text = []

    for text_batch, audio_batch in tqdm(zip(eval_text_dataloader, eval_audio_dataloader), total=len(eval_text_dataloader)):
        with torch.no_grad():
            text_batch["input_ids"] = text_batch["input_ids"].to(device)
            audio_batch["input_values"] = audio_batch["input_values"].to(device)
            text_batch["attention_mask"] = text_batch["attention_mask"].to(device)
            audio_batch["attention_mask"] = audio_batch["attention_mask"].to(device)
            text_batch["labels"] = text_batch["labels"].to(device)
            proba = model(text_batch, audio_batch)
            proba_text = text_model(**text_batch).logits.softmax(dim=1)
            proba_audio = audio_model(**audio_batch).logits.softmax(dim=1)
            _, y_hat = proba.max(1)
            # y_hat_text = torch.argmax(proba_text, dim=-1)
            # y_hat_audio = torch.argmax(proba_audio, dim=-1)
            _, y_hat_text = proba_text.max(1)
            _, y_hat_audio = proba_audio.max(1)

            logits = [tuple(nums.cpu().tolist()) for nums in proba]
            logits_text = [tuple(nums.cpu().tolist()) for nums in proba_text]
            logits_audio = [tuple(nums.cpu().tolist()) for nums in proba_audio]

            all_logits += logits
            all_logits_text += logits_text
            all_logits_audio += logits_audio

            preds += [idx.item() for idx in y_hat]
            preds_text += [idx.item() for idx in y_hat_text]
            preds_audio += [idx.item() for idx in y_hat_audio]

            eval_labels += text_batch["labels"].tolist()

        clear_batch_from_cuda(text_batch)
        clear_batch_from_cuda(audio_batch)
        
    eval_accuracy = np.mean(np.array(preds) == np.array(eval_labels))
    f1 = f1_score(eval_labels, preds)

    return (all_logits, all_logits_text, all_logits_audio), (preds, preds_text, preds_audio), eval_labels, eval_accuracy, f1


text_model = RobertaForSequenceClassification.from_pretrained('/nfs/nicholas/data/sentiment_classifier')
audio_model = HubertForSequenceClassification.from_pretrained('superb/hubert-base-superb-er')
audio_model.load_state_dict(torch.load("/nfs/nicholas/outputs/improved_audio2/epoch4/hubert_switchboard.pth"))

model = ConcatModel()
model.load_state_dict(torch.load("/nfs/nicholas/outputs/newest_betterhubert/hubert_switchboard.pth"))
model.to(device)
text_model.to(device)
audio_model.to(device)

evaluation = evaluate(model, text_model, audio_model, test_dataloader_text, test_dataloader_audio)

file_path = '/nfs/nicholas/demo_content/outputs/demo_meeting_8_1_2023_eval_all.pkl'

# Write the tuple to disk using pickle
with open(file_path, 'wb') as file:
    pickle.dump(evaluation, file)

print(evaluation[4])
# print(evaluation[0], epoch_loss)