import pandas as pd

def get_importance(model, feature_columns):

    return (
        pd.DataFrame({
            "Feature": feature_columns,
            "Importance": model.feature_importances_
        })
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(10)
    )