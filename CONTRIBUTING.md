# Contributing to EPred

Thank you for your interest in contributing to **EPred (Energy Prediction System)**! As a Free and Open Source Software (FOSS) project, we welcome contributions from the community.

---

## Code of Conduct
We are committed to providing a welcoming, inclusive, and harassment-free environment for all contributors. Please be respectful and constructive in all discussions.

---

## How to Contribute

### 1. Reporting Bugs & Suggesting Features
- Search existing issues to ensure your bug or feature request has not already been submitted.
- When opening an issue, provide:
  - Clear title and description.
  - Reproducible steps, Python version, and operating system.
  - Expected vs actual behavior.

### 2. Development Setup
```bash
# 1. Fork and clone the repository
git clone https://github.com/<your-username>/EPred.git
cd EPred

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies and development packages
pip install -r requirements.txt
```

### 3. Making Changes
- Create a new branch: `git checkout -b feature/my-feature` or `fix/issue-description`.
- Adhere to PEP 8 styling conventions.
- Maintain documentation integrity: docstrings, type hints, and comments.
- **Data Leakage Rule**: Never use future information in past feature sets. All scalers and transformers must strictly fit on the training split only.

### 4. Running Tests
Before submitting a pull request, ensure all unit and integration tests pass:
```bash
pytest -v
```

### 5. Submitting a Pull Request (PR)
- Push your changes: `git push origin feature/my-feature`.
- Open a Pull Request against the `main` branch.
- Detail the changes made, reason for the change, and provide test verification results.
