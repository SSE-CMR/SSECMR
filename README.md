# SSE_CMR

## Data explaination

The are two types of data.

1. Binary hash codes of real-world datasets extracted by [GitHub - XizeWu/CIRH](https://github.com/XizeWu/CIRH)
2. Binary hash codes generated randomly by `genRandomVectors.py`.

## Installation

1. Install python. Recomment python version: 3.11.x
2. Install  dependencies:`pip install -r requirements.txt`
3. Compiling cython:`python setup.py build_ext --inplace`

## Usage

`python main.py [-h] [--db DB] [--r R] [--h H] [--s S] [--v V] [--t T] [--mode MODE]`

options:

```
  -h, --help   show this help message and exit
  --db DB      Database name
  --r R        Search radius
  --h H        Hashlen
  --s S        Number of subcodes
  --v V        Whether to verify the correctness of results
  --t T        Number of query
  --mode MODE  Search for a range of raddi of a specific radius
```

## Demo

1. Extract `Data.zip` to `Data` folder
2. Run `python main.py`