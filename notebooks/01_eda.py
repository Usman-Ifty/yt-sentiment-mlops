import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/raw/YoutubeCommentsDataSet.csv")

# Basic info
print(df.shape)
print(df.head())
print(df.columns.tolist())
print(df.isnull().sum())

# Sentiment distribution
print(df['Sentiment'].value_counts())
print(df['Sentiment'].value_counts(normalize=True).round(3) * 100)

# Comment length distribution
df['length'] = df['Comment'].astype(str).apply(len)
print(df['length'].describe())

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

df['Sentiment'].value_counts().plot(kind='bar', ax=axes[0], color=['#4CAF50','#2196F3','#F44336'])
axes[0].set_title('Sentiment Distribution')
axes[0].set_xlabel('Sentiment')
axes[0].tick_params(rotation=0)

df['length'].hist(bins=50, ax=axes[1], color='#7F77DD')
axes[1].set_title('Comment Length Distribution')
axes[1].set_xlabel('Characters')

plt.tight_layout()
plt.savefig("notebooks/eda_plots.png")
print("Saved plots to notebooks/eda_plots.png")
plt.show()

# Check for class imbalance
# Expected: ~62% positive, ~25% neutral, ~13% negative
# We'll handle this with weighted loss during training
