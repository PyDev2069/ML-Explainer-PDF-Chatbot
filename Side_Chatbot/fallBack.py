fallback_answers = {
    "what is cross validation" : """ Cross-validation is a resampling technique used to assess how well a model
generalizes. K-fold CV is
commonly used.
CODE : 
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
print(scores)
""",
"data leakage":""" Data leakage occurs when information from outside the training dataset is used to
create the model,
resulting in overly optimistic performance estimates.""",

"Supervised Learning":"Supervised learning is a type of machine learning where the model is trained on labeled data, meaning each input has a corresponding correct output.",
"Unsupervised Learning":"Unsupervised learning is a type of machine learning where the model is trained on unlabeled data and tries to identify patterns or groupings without predefined outputs."
}

def get_fallback_answer(query: str):
    query_lower = query.lower()
    for key in fallback_answers:
        if key in query_lower:
            return fallback_answers[key]
    return None