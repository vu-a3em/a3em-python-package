import os
from a3em.datasets import arden
from  dotenv import load_dotenv
from pathlib import Path

# load .env variables
load_dotenv()
prefetch_path = Path(os.getenv('PREFETCH_PATH'))

# load data from arden
df = arden.load_data(prefetch_path)
print(df)
