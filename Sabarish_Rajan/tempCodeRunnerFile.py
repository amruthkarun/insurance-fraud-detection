base_estimators = [
    ('rf', RandomForestClassifier(n_estimators=100,
    max_depth = 5,
    min_samples_split=5,
    random_state=42)),
    ('lr', LogisticRegression(max_iter=1000, random_state=42))
]