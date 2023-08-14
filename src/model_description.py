model_descriptions = {
    "Multi-layer Perceptron": "The **Multi-layer Perceptron** is like a brain-inspired system that learns patterns from examples. It's great at understanding complex relationships in data, making it good for handwriting prediction. Think of it as a series of connected switches that adapt to recognize different aspects of handwriting.",
    "K-Nearest Classifier": "The **K-Nearest Classifier** is like asking your neighbors for advice. It looks at handwriting samples that are close to the one you're trying to predict and asks them what they think. By considering several similar samples, it makes a prediction based on what the majority of those neighbors suggest.",
    "Random Forest": "Imagine a group of experts discussing handwriting styles together. Each expert is like a small tree that focuses on certain aspects of handwriting. The **Random Forest** combines the opinions of these experts to make a prediction. It's like getting multiple perspectives to improve accuracy.",
    "Logistic Regression": "Think of **Logistic Regression** as a simple yet smart friend who can decide between two options. It looks at various handwriting characteristics and figures out the likelihood of the writing belonging to a certain category. It's like saying, 'Based on these traits, it's more likely to be this type of handwriting.'",
    "Quadratic Discriminant Analysis": "**Quadratic Discriminant Analysis** is like studying the curves and shapes of handwriting. It focuses on how handwriting features come together to form patterns. It's like using the shapes of the letters to distinguish between different writing styles.",
    "Decision Tree": "A **Decision Tree** is like a flowchart for predicting handwriting. It asks questions about specific handwriting features, like 'Is the loop of 'g' open or closed?' Based on the answers, it follows paths until it reaches a prediction. It's like a game of 20 Questions for identifying handwriting styles.",
}

model_strengths = {
    "Multi-layer Perceptron": "Great for handling complex patterns and relationships in data.",
    "K-Nearest Classifier": "Effective for simple and quick predictions based on similar examples.",
    "Random Forest": "Strong at combining diverse opinions to achieve high prediction accuracy.",
    "Logistic Regression": "Useful for binary classification tasks and interpreting feature importance.",
    "Quadratic Discriminant Analysis": "Capable of modeling intricate relationships using data shapes.",
    "Decision Tree": "Offers clear insights into decision-making processes with easily interpretable rules."
}

model_weaknesses = {
    "Multi-layer Perceptron": "May require careful tuning of hyperparameters and more data for optimal performance.",
    "K-Nearest Classifier": "Can be sensitive to the choice of 'k' and may not perform well in high-dimensional spaces.",
    "Random Forest": "Can become complex and computationally expensive for very large datasets.",
    "Logistic Regression": "May struggle with capturing complex relationships and interactions in data.",
    "Quadratic Discriminant Analysis": "Sensitive to overfitting, especially with small datasets.",
    "Decision Tree": "Prone to overfitting, resulting in poor generalization to new data."
}
