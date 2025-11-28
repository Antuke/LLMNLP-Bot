from transformers import DistilBertTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import DistilBertForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_score": f1_score(labels, predictions, average="weighted"),
        "recall": recall_score(labels, predictions, average="weighted")
    }

# Load and prepare data
df_train = pd.read_csv('./dataset/bigger_dataset.csv', delimiter=";")
df_test = pd.read_csv('./dataset/test_set.csv', delimiter=";")
#train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# Convert DataFrames to Datasets directly
train_dataset = Dataset.from_pandas(df_train)
val_dataset = Dataset.from_pandas(df_test)


# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Define tokenization function
def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)

# Tokenize datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

print(train_dataset)

# Load model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",    # Fixed the evaluation_strategy parameter
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate
eval_results = trainer.evaluate()
print(f"Validation Accuracy: {eval_results['eval_accuracy']}")
print(f"Validation F1-Score: {eval_results['eval_f1_score']}")
print(f"Validation Recall: {eval_results['eval_recall']}")

# Save model and tokenizer
model.save_pretrained('./fine-tuned-distilbert')
tokenizer.save_pretrained('./fine-tuned-distilbert')