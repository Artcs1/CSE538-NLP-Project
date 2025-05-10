import json
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import os
import numpy as np

# Ensure the stats directory exists
os.makedirs('stats', exist_ok=True)

# Load annotations_translated.json
with open('annotations_translated.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Filtering records with 'translated' field and ignoring non-string texts
data = [item for item in data if 'translated' in item and isinstance(item.get('text'), str)]

# Total records
total_records = len(data)

# Language distribution
languages = [item['language'] for item in data]
lang_distribution = Counter(languages)

# Label-wise statistics per language
label_stats = {'Language': [], 'Label 1': [], 'Label 0': [], 'Total': []}
for lang in set(languages):
    lang_data = [item for item in data if item['language'] == lang]
    label_1 = sum(1 for item in lang_data if item['label'] == 1)
    label_0 = sum(1 for item in lang_data if item['label'] == 0)
    label_stats['Language'].append(lang)
    label_stats['Label 1'].append(label_1)
    label_stats['Label 0'].append(label_0)
    label_stats['Total'].append(label_1 + label_0)

# Adding total, min, avg, max rows
label_stats_df = pd.DataFrame(label_stats)
label_stats_df = label_stats_df.sort_values(by='Total', ascending=False)
label_stats_df.loc['Total'] = label_stats_df.sum(numeric_only=True).astype(int)
label_stats_df.loc['Total', 'Language'] = 'Total'
label_stats_df.loc['Min'] = label_stats_df.min(numeric_only=True).astype(int)
label_stats_df.loc['Min', 'Language'] = 'Min'
label_stats_df.loc['Avg'] = label_stats_df.mean(numeric_only=True).round(1)
label_stats_df.loc['Avg', 'Language'] = 'Avg'
label_stats_df.loc['Max'] = label_stats_df.max(numeric_only=True).astype(int)
label_stats_df.loc['Max', 'Language'] = 'Max'
label_stats_df.to_csv('stats/label_stats.csv', index=False)

# Character-wise statistics with total columns
char_stats = []
for lang in set(languages):
    texts = [item['text'] for item in data if item['language'] == lang]
    translated_texts = [item['translated'] for item in data if item['language'] == lang]

    char_stats.append([
        lang,
        round(np.mean([len(text) for text in texts]), 1),
        min(len(text) for text in texts),
        max(len(text) for text in texts),
        sum(len(text) for text in texts),
        round(np.mean([len(text) for text in translated_texts]), 1),
        min(len(text) for text in translated_texts),
        max(len(text) for text in translated_texts),
        sum(len(text) for text in translated_texts)
    ])

# Character statistics dataframe with total columns
char_stats_df = pd.DataFrame(char_stats, columns=[
    'Language', 'Original Text Avg', 'Original Text Min', 'Original Text Max', 'Original Text Total',
    'Translated Text Avg', 'Translated Text Min', 'Translated Text Max', 'Translated Text Total'
])
char_stats_df = char_stats_df.sort_values(by='Original Text Total', ascending=False)
char_stats_df.loc['Total'] = [
    'Total',
    round(char_stats_df['Original Text Avg'].mean(), 1),
    int(char_stats_df['Original Text Min'].min()),
    int(char_stats_df['Original Text Max'].max()),
    int(char_stats_df['Original Text Total'].sum()),
    round(char_stats_df['Translated Text Avg'].mean(), 1),
    int(char_stats_df['Translated Text Min'].min()),
    int(char_stats_df['Translated Text Max'].max()),
    int(char_stats_df['Translated Text Total'].sum())
]
char_stats_df.to_csv('stats/char_stats.csv', index=False)

# Handling outliers in Character-wise Box Plot
char_lengths = {lang: [len(item['text']) for item in data if item['language'] == lang] for lang in set(languages)}
plt.figure(figsize=(14, 7))
plt.boxplot(char_lengths.values(), tick_labels=char_lengths.keys(), patch_artist=True, showfliers=False)
plt.title('Character-wise Statistics by Language (Box Plot - Outliers Excluded)')
plt.xlabel('Languages')
plt.ylabel('Character Count')
plt.tight_layout()
plt.savefig('stats/characterwise_boxplot.png')

import matplotlib.cm as cm

# Language Distribution Pie Chart with Larger Labels, Adjusted % and Distinct Colors
plt.figure(figsize=(10, 10))
colors = plt.colormaps.get_cmap('tab20')(np.linspace(0, 1, len(lang_distribution)))
wedges, texts, autotexts = plt.pie(
    lang_distribution.values(),
    labels=lang_distribution.keys(),
    autopct='%1.1f%%',
    textprops={'fontsize': 16, 'weight': 'bold'},
    colors=colors,
    startangle=90
)

# Adjusting % text positions slightly towards the edge
for autotext in autotexts:
    autotext.set_position((autotext.get_position()[0] * 1.1, autotext.get_position()[1] * 1.1))

plt.title('Language Distribution Percentage', fontsize=18)
plt.tight_layout()
plt.savefig('stats/language_distribution.png')


# Label Distribution Stacked Bar Plot (Fixing Total in Stacked Bar)
label_stats_df.drop(['Total', 'Min', 'Avg', 'Max']).set_index('Language')[['Label 1', 'Label 0']].plot(
    kind='bar', stacked=True, figsize=(12, 6)
)

plt.title('Label Distribution by Language (Stacked)')
plt.xlabel('Languages')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('stats/label_distribution_stacked.png')

# Word-wise statistics with min, max, avg, total
word_stats = []
for lang in set(languages):
    texts = [item['text'] for item in data if item['language'] == lang]
    translated_texts = [item['translated'] for item in data if item['language'] == lang]
    
    word_stats.append([
        lang,
        len(texts),
        min(len(text.split()) for text in texts),
        max(len(text.split()) for text in texts),
        round(np.mean([len(text.split()) for text in texts]), 1),
        len(translated_texts),
        min(len(text.split()) for text in translated_texts),
        max(len(text.split()) for text in translated_texts),
        round(np.mean([len(text.split()) for text in translated_texts]), 1)
    ])

# Word statistics dataframe
word_stats_df = pd.DataFrame(word_stats, columns=[
    'Language', 'Original Records', 'Original Words Min', 'Original Words Max', 'Original Words Avg',
    'Translated Records', 'Translated Words Min', 'Translated Words Max', 'Translated Words Avg'
])

# Calculating Total Row (on All Records)
total_texts = [item['text'] for item in data]
total_translated_texts = [item['translated'] for item in data]

word_stats_df.loc['Total'] = [
    'Total',
    len(total_texts),
    min(len(text.split()) for text in total_texts),
    max(len(text.split()) for text in total_texts),
    round(np.mean([len(text.split()) for text in total_texts]), 1),
    len(total_translated_texts),
    min(len(text.split()) for text in total_translated_texts),
    max(len(text.split()) for text in total_translated_texts),
    round(np.mean([len(text.split()) for text in total_translated_texts]), 1)
]

word_stats_df = word_stats_df.sort_values(by='Original Records', ascending=False)
word_stats_df.to_csv('stats/word_stats.csv', index=False)


# Language-wise Record Count Statistics
lang_record_counts = pd.DataFrame.from_dict(lang_distribution, orient='index', columns=['Record Count'])
lang_record_counts = lang_record_counts.sort_values(by='Record Count', ascending=False)
lang_record_counts.loc['Total'] = lang_record_counts.sum()

# Saving Record Count CSV
lang_record_counts.to_csv('stats/language_record_counts.csv')

# Plotting Language-wise Record Counts
plt.figure(figsize=(12, 6))
lang_record_counts.drop('Total').plot(kind='bar', legend=False, color='skyblue')
plt.title('Language-wise Record Counts')
plt.xlabel('Languages')
plt.ylabel('Record Count')
plt.tight_layout()
plt.savefig('stats/language_record_counts.png')


# Generating Word-wise Box Plot (Outliers Excluded)
word_lengths = {lang: [len(item['text'].split()) for item in data if item['language'] == lang] for lang in set(languages)}
plt.figure(figsize=(14, 7))
plt.boxplot(word_lengths.values(), tick_labels=word_lengths.keys(), patch_artist=True, showfliers=False)
plt.title('Word-wise Statistics by Language (Box Plot - Outliers Excluded)')
plt.xlabel('Languages')
plt.ylabel('Word Count')
plt.tight_layout()
plt.savefig('stats/wordwise_boxplot.png')

print("✅ All statistics generated and saved under 'stats/' directory.")
