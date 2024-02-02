import os
from pathlib import Path

data_dir = Path(os.getenv('PATH_DIR')) / 'data'
house_prices_test = data_dir / 'test.csv'
house_prices_train = data_dir / 'train.csv'
target = data_dir / 'sample_submission.csv'