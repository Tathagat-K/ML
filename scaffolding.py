# setup_scaffold.py
import os
from pathlib import Path
import shutil

BASE = Path("churn_project")

# Clean old scaffold if exists
if BASE.exists():
    shutil.rmtree(BASE)

# Define dirs
dirs = [
    BASE / "data" / "raw",
    BASE / "data" / "interim",
    BASE / "data" / "processed",
    BASE / "notebooks",
    BASE / "src" / "features",
    BASE / "src" / "models",
    BASE / "src" / "eval",
    BASE / "src" / "app",
    BASE / "src" / "utils",
    BASE / "tests",
    BASE / "models",
    BASE / "reports",
]

for d in dirs:
    d.mkdir(parents=True, exist_ok=True)

# Move your uploaded dataset into raw/
uploaded_file = Path("customer_churn.csv")
if uploaded_file.exists():
    shutil.copy2(uploaded_file, BASE / "data" / "raw" / uploaded_file.name)

# Create placeholders
(BASE / "README.md").write_text("# Customer Churn Project\n\nScaffold ready.\n")
(BASE / "requirements.txt").write_text("pandas\nscikit-learn\nmatplotlib\nseaborn\nfastapi\nuvicorn\n")

print("✅ Project scaffold created at", BASE.resolve())
