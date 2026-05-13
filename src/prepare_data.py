import pandas as pd
from sklearn.model_selection import train_test_split

def csv_to_df(csv_file_path: str) -> pd.DataFrame:
    """
    Read csv file as pandas df
    """
    df = pd.read_csv(csv_file_path)
    return df


def drop_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    return df.drop(columns=columns)


def split_train_eval_test(
    df: pd.DataFrame,
    target_col: str,
    test_size: float=0.15,
    eval_size: float=0.15,
    random_state: int=123
) -> dict:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train_eval, X_test, y_train_eval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    eval_ratio = eval_size / (1-test_size)
    X_train, X_eval, y_train, y_eval = train_test_split(
        X_train_eval, y_train_eval, test_size=eval_ratio, random_state=random_state
    )
    
    # print(f"Shapes - X_train: {X_train.shape}, X_eval: {X_eval.shape}, X_test: {X_test.shape}")
    
    return {
        "X_train": X_train,
        "X_eval": X_eval,
        "X_test": X_test,
        "y_train": y_train,
        "y_eval": y_eval,
        "y_test": y_test
    }