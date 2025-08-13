# Now test the traditional ML models with NLTK data available
preprocessor = DataPreprocessor()
feature_engineer = FeatureEngineer()
ml_models = TraditionalMLModels()

# Preprocess sample data
df_processed = preprocessor.preprocess_dataset(df_sample)
print(f"Processed dataset shape: {df_processed.shape}")
print("\nCleaned text examples:")
for i, row in df_processed.iterrows():
    print(f"{i}: {row['cleaned_text'][:100]}...")

# Extract features
feature_matrix = feature_engineer.create_feature_matrix(df_processed)
print(f"\nFeature matrix shape: {feature_matrix.shape}")
print("\nFeature matrix:")
print(feature_matrix)

# Since we have a small sample, let's create a larger synthetic dataset for demonstration
np.random.seed(42)

# Create larger synthetic dataset
fake_news_templates = [
    "BREAKING: {} discovered! Click here to learn the shocking truth!!!",
    "You won't believe what {} revealed! The {} doesn't want you to know this!",
    "URGENT: {} exposed! They are hiding this from the public!",
    "SHOCKING: Scientists hate this one simple trick about {}!",
    "{} conspiracy revealed! The truth will amaze you!"
]

real_news_templates = [
    "According to a peer-reviewed study published in {}, researchers have found {}.",
    "The {} organization announced new findings regarding {}.",
    "A comprehensive analysis conducted by {} indicates that {}.",
    "Researchers from {} university have published evidence of {}.",
    "The {} journal reports significant progress in {} research."
]

# Generate synthetic examples
topics = ["climate change", "medical research", "technology", "economics", "education"]
organizations = ["Nature", "Science", "WHO", "UNESCO", "Harvard", "MIT", "Stanford"]

synthetic_texts = []
synthetic_labels = []

# Generate fake news examples
for _ in range(50):
    template = np.random.choice(fake_news_templates)
    topic = np.random.choice(topics)
    org = np.random.choice(organizations)
    text = template.format(topic, org)
    synthetic_texts.append(text)
    synthetic_labels.append(0)

# Generate real news examples
for _ in range(50):
    template = np.random.choice(real_news_templates)
    topic = np.random.choice(topics)
    org = np.random.choice(organizations)
    text = template.format(org, topic)
    synthetic_texts.append(text)
    synthetic_labels.append(1)

# Create synthetic dataset
synthetic_df = pd.DataFrame({
    'text': synthetic_texts,
    'label': synthetic_labels
})

print(f"\nSynthetic dataset created with {len(synthetic_df)} examples")
print(f"Class distribution: {synthetic_df['label'].value_counts().to_dict()}")
print("\nSample entries:")
print(synthetic_df.head())