from typing import Protocol, runtime_checkable


@runtime_checkable
class ClassifierModel(Protocol):
    def fit(self, x_train, y_train) -> "ClassifierModel":
        ...

    def predict(self, x):
        ...

    def predict_proba(self, x):
        ...
