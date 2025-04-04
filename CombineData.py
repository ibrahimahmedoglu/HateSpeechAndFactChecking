import pandas as pd
import json
import numpy as np

# Davidson dataset
df_davidson = pd.read_csv("Data/davidson.csv")
label_map_davidson = {0: "hate", 1: "offensive", 2: "normal"}
df_davidson['label'] = df_davidson['class'].map(label_map_davidson)
df_davidson['text'] = df_davidson['tweet']
df_davidson['source'] = 'davidson'
df_davidson = df_davidson[['text', 'label', 'source']]

# HateXplain dataset
with open("Data/HateXplain.json", "r") as f:
    data = json.load(f)

rows = []
for post_id, content in data.items():
    tokens = content['post_tokens']
    text = " ".join(tokens)
    labels = [ann['label'] for ann in content['annotators']]
    majority_label = max(set(labels), key=labels.count)
    rows.append({
        'text': text,
        'label': majority_label.lower(),  # Normalize to lowercase for consistency
        'source': 'hatexplain'
    })
df_hatexplain = pd.DataFrame(rows)

# Gab Hate Corpus dataset
df_gab = pd.read_csv("Data/GabHateCorpus.tsv", sep="\t")
gab_texts = df_gab.groupby("ID")["Text"].first().reset_index()
gab_labels = df_gab.groupby("ID")["Hate"].max().reset_index()  # Use max as majority
gab = pd.merge(gab_texts, gab_labels, on="ID")
gab["label"] = gab["Hate"].apply(lambda x: "hate" if x == 1 else "normal")
gab["text"] = gab["Text"]
gab["source"] = "gab"
df_gab_clean = gab[["text", "label", "source"]]

# Combine all
df_all = pd.concat([df_davidson, df_hatexplain, df_gab_clean], ignore_index=True)

# Save to file
df_all.to_csv("combined_hate_speech_dataset.csv", index=False)
print("Combined dataset saved.")
