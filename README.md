# Movie Recommendation Demo

This repository contains a small movie recommendation tool built with Python.
It demonstrates how to train an advanced collaborative filtering model using
[SVD++](https://surprise.readthedocs.io/en/stable/matrix_factorization.html#svdpp)
from the `scikit-surprise` library.

## Requirements

- Python 3
- `scikit-surprise`
- `pandas`

Install dependencies with:

```bash
pip install pandas scikit-surprise
```

## Usage

The repository includes sample data in `ratings.csv` and `movies.csv`.
Run the recommendation script by specifying a user ID:

```bash
python recommend.py --user 1
```

The script will output the top recommended movies that the user has not yet rated.

You can specify the number of results with `-n` and custom paths to the data files
with `--ratings` and `--movies`.
