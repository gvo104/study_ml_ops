from pathlib import Path

import matplotlib
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_confusion_matrix(
    y_true,
    y_pred,
    classes: list[str],
    output_path: Path | None = None,
) -> Path:
    """Save a confusion matrix figure and return its path."""
    from src.config import FIGURES_DIR

    output_path = output_path or FIGURES_DIR / "confusion_matrix.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=classes,
    )
    display.plot(cmap="Blues", xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()

    return output_path


def save_classification_report(
    report_text: str,
    output_path: Path,
) -> Path:
    """Write sklearn classification report to a text file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")

    return output_path


def plot_class_distribution(
    df,
    target_column: str,
    output_path: Path | None = None,
) -> Path:
    """Save class distribution bar chart."""
    from src.config import FIGURES_DIR

    output_path = output_path or FIGURES_DIR / "class_distribution.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counts = df[target_column].value_counts()
    plt.figure(figsize=(10, 5))
    counts.plot(kind="bar")
    plt.title("Class distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()

    return output_path
