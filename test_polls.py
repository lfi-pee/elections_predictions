from pathlib import Path
from src.load_polls import load_poll_tokens

print("Loading polls...")
df = load_poll_tokens(Path("data"))
print(f"Loaded {len(df)} poll tokens.")
print("Columns:", list(df.columns))
print("Sample:")
print(df.head())
