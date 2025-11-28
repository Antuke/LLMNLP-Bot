import pandas as pd
import random
import nltk
from nlpaug.augmenter.word import SynonymAug
from nlpaug.augmenter.char import RandomCharAug
import string

# Load the dataset
df = pd.read_csv('./dataset.csv', sep=";", header=None, names=["sentence", "label"])

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Initialize augmenters
synonym_aug = SynonymAug(aug_src='wordnet', aug_max=1)  # Synonym replacement
noise_aug = RandomCharAug(action="swap", aug_char_max=1, aug_char_p=0.01)  # Noise injection (character-level)

# Apply augmentations to the dataset and create new rows
augmented_data = []
for _, row in df.iterrows():
    sentence = row["sentence"]
    label = row["label"]

    # 1. Augmented sentence (with noise and/or synonym replacement)
    if random.random() < 0.1:
        augmented_sentence = noise_aug.augment(sentence)[0]  # Apply noise injection
        augmented_sentence = synonym_aug.augment(augmented_sentence)[0]  # Then synonym replacement
    else:
        augmented_sentence = synonym_aug.augment(sentence)[0]
    augmented_data.append([augmented_sentence, label])
    
    # 2. Lemmatized version of the sentence
    tokens = nltk.word_tokenize(sentence)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    lemmatized_sentence = " ".join(lemmatized_tokens)
    augmented_data.append([lemmatized_sentence, label])
    
    # 3. Sentence with common (stop) words removed
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    stopword_removed_sentence = " ".join(filtered_tokens)
    augmented_data.append([stopword_removed_sentence, label])
    
    # 4. Sentence with punctuation removed
    sentence_no_punct = sentence.translate(str.maketrans('', '', string.punctuation))
    augmented_data.append([sentence_no_punct, label])

# Create a new DataFrame with augmented data
augmented_df = pd.DataFrame(augmented_data, columns=["sentence", "label"])

# Concatenate the original dataset with the augmented data
final_df = pd.concat([df, augmented_df], ignore_index=True)

# Save the final dataset
final_df.to_csv('./bigger_dataset.csv', sep=";", index=False, header=False)

print("Augmentation complete! Final dataset saved to './bigger_dataset.csv'.")