import pytest
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from email_classifier.training.classifier_model.base_classifier import BaseClassifier
from email_classifier.training.classifier_model.logistic_classifier import LogisticTextClassifier
from email_classifier.training.classifier_model.sgd_classifier import SGDTextClassifier
from email_classifier.training.classifier_model.factory import ClassifierFactory
from email_classifier.training.trainer.trainer import Trainer


def test_base_classifier_instantiation():
    """
    Test that BaseClassifier cannot be instantiated directly.
    """
    with pytest.raises(TypeError, match="Can't instantiate abstract class BaseClassifier"):
        BaseClassifier()


def test_logistic_text_classifier():
    """
    Test LogisticTextClassifier functionality.
    """
    classifier = LogisticTextClassifier()
    model = classifier.get_classifier()
    hyperparams = classifier.get_hyperparameters()

    # Assert the classifier is a LogisticRegression instance
    assert isinstance(model, LogisticRegression), "Expected LogisticRegression instance."

    # Assert hyperparameters are valid
    assert "C" in hyperparams, "Expected 'C' in hyperparameters."
    assert isinstance(hyperparams["C"], list), "Expected 'C' to be a list."


def test_sgd_text_classifier():
    """
    Test SGDTextClassifier functionality.
    """
    classifier = SGDTextClassifier()
    model = classifier.get_classifier()
    hyperparams = classifier.get_hyperparameters()

    # Assert the classifier is an SGDClassifier instance
    assert isinstance(model, SGDClassifier), "Expected SGDClassifier instance."

    # Assert hyperparameters are valid
    assert "loss" in hyperparams, "Expected 'loss' in hyperparameters."
    assert isinstance(hyperparams["loss"], list), "Expected 'loss' to be a list."


def test_classifier_factory_get_classifier():
    """
    Test that ClassifierFactory returns the correct classifier instance.
    """
    logistic_classifier = ClassifierFactory.get_classifier("logistic")
    sgd_classifier = ClassifierFactory.get_classifier("sgd")

    assert isinstance(logistic_classifier, LogisticTextClassifier), "Expected LogisticTextClassifier instance."
    assert isinstance(sgd_classifier, SGDTextClassifier), "Expected SGDTextClassifier instance."


def test_classifier_factory_unsupported_type():
    """
    Test that ClassifierFactory raises an error for unsupported classifier types.
    """
    with pytest.raises(ValueError, match="Unsupported classifier type: unsupported"):
        ClassifierFactory.get_classifier("unsupported")


def test_trainer_train_and_evaluate():
    """
    Test the Trainer class's train and evaluate methods using mock data.
    """
    # Mock data
    X_train = [
        "Dear User, your subscription is about to expire. Renew now to continue enjoying our services. Visit example.com/renew",
        "📣 Congratulations! You’ve won a $1,000 gift card! Click here to claim your prize: spam-link.com",
        "Call for papers: Submit your work to our journal. Deadline: June 30, 2024. Contact us at papers@journal.com.",
        "Reminder: Your account password will expire soon. Update it to maintain access. Visit account-security.com.",
        "Introducing our new product: A cutting-edge AI tool to enhance productivity. Learn more at ai-tool.com.",
        "Don't miss out on our special discount! Offer valid until December 31. Visit discounts.com.",
        "Important: Update your billing information to avoid service interruptions. Go to billing.com.",
        "Join us for a webinar on AI advancements. Register now at ai-webinar.com."
    ]
    y_train = [1, 0, 1, 1, 1, 0, 1, 0]

    X_test = [
        "Update your profile to access new features. Visit profile-update.com",
        "You’ve been selected to receive a special offer! Act now at special-offer.com.",
        "Important notice: Your account has been locked. Reset it at account-recovery.com."
    ]
    y_test = [1, 0, 1]

    # Create instances
    vectorizer = TfidfVectorizer()
    classifier = LogisticTextClassifier()
    trainer = Trainer(classifier=classifier, vectorizer=vectorizer)

    # Train the model
    trained_model = trainer.train(X_train, y_train)

    # Assert the trained model is a pipeline
    assert isinstance(trained_model, Pipeline), "Expected trained model to be a Pipeline instance."

    # Evaluate the model
    metrics = trainer.evaluate(trained_model, X_test, y_test)

    # Assert evaluation metrics
    assert "accuracy" in metrics, "Expected 'accuracy' in evaluation metrics."
    assert 0.0 <= metrics["accuracy"] <= 1.0, "Accuracy should be between 0 and 1."


