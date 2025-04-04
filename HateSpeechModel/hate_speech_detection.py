import pandas as pd
import numpy as np
import re
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # If needed for Colab compatibility


# Load dataset
df = pd.read_csv("combined_hate_speech_dataset.csv")

# Sample only 10% of the data (stratified)
df = df[df['label'].isin(['hate', 'offensive', 'normal'])]  # Filter only known labels
df_sampled, _ = train_test_split(df, stratify=df['label'], test_size=0.90, random_state=42)

# Encode labels
label_map = {'hate': 0, 'offensive': 1, 'normal': 2}
df_sampled['labels'] = df_sampled['label'].map(label_map)

# Optional: Save the label classes
np.save("label_classes.npy", list(label_map.keys()))

# Train/val/test split
train_df, temp_df = train_test_split(df_sampled, stratify=df_sampled['labels'], test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, stratify=temp_df['labels'], test_size=0.5, random_state=42)

# Convert to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df[['text', 'labels']])
val_dataset = Dataset.from_pandas(val_df[['text', 'labels']])
test_dataset = Dataset.from_pandas(test_df[['text', 'labels']])

# Load tokenizer and model
model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Tokenize data
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Define metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average='macro')
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    no_cuda=True,
)

# Trainer setup
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate on test set
predictions = trainer.predict(test_dataset)
y_true = predictions.label_ids
y_pred = np.argmax(predictions.predictions, axis=1)
print(classification_report(y_true, y_pred))
