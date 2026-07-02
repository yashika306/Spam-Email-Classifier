from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(model_name, y_true, y_pred):
    """
    Print evaluation metrics for a model.
    """

    print("\n" + "=" * 60)
    print(f"{model_name}")
    print("=" * 60)

    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Ham", "Spam"]
        )
    )


def compare_models(nb_acc, log_acc):
    """
    Compare both models.
    """

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(f"Naive Bayes Accuracy        : {nb_acc:.4f}")
    print(f"Logistic Regression Accuracy: {log_acc:.4f}")

    if nb_acc > log_acc:
        print("\n🏆 Winner : Naive Bayes")

    elif log_acc > nb_acc:
        print("\n🏆 Winner : Logistic Regression")

    else:
        print("\n🤝 Both models performed equally.")