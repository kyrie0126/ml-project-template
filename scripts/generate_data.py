import os
import sys
import numpy as np
import pandas as pd

def generate_example_csv(output_path: str="data/example.csv", n_rows: int=10000, random_state: int=123) -> str:
    rng = np.random.default_rng(random_state)
    
    df = pd.DataFrame({
        "ID": np.arange(1, n_rows+1),
        "CATEGORY_A": rng.choice(["alpha","beta"], size=n_rows),
        "CATEGORY_B": rng.choice(["low","medium","high"], size=n_rows),
        "NUMERIC_1": rng.normal(loc=50, scale=15, size=n_rows).round(2),
        "NUMERIC_2": rng.uniform(0, 100, size=n_rows).round(2),
        "NUMERIC_3": rng.integers(1, 500, size=n_rows),
    })
    
    # label generated from logistic function so that it's not purely random and a test model can actually learn something
    score = (
        (df["NUMERIC_1"] - 50) / 15 * 0.4
        + (df["NUMERIC_2"] - 50) / 50 * 0.3
        + (df["CATEGORY_A"] == "alpha").astype(float) * 0.5
        + (df["CATEGORY_B"] == "high").astype(float) * 0.3
        + rng.normal(0, 0.3, size=n_rows)
    )
    prob = 1 / (1 + np.exp(-score))
    df["LABEL"] = (rng.random(n_rows) < prob).astype(int)
    
    df.to_csv(output_path, index=False)
    return output_path

if __name__ == "__main__":
    output_path = sys.argv[1] if len(sys.argv) > 1 else "data/example.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    generate_example_csv(output_path)
    print(f"Generated: {output_path}")