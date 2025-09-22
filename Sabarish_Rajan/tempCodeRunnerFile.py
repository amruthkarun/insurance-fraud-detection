y_prob = best_model.predict_proba(X_test)

thresholds = np.arange(0.1, 0.9, 0.05)

f1_scores = [f1_score(y_test, (y_prob[:,1]>t).astype(int)) for t in thresholds]

best_threshold = thresholds[np.argmax(f1_scores)]

y_pred = (y_prob[:,1] >= best_threshold).astype(int)
