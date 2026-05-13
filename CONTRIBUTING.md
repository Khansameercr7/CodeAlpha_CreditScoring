# Contributing to Credit Scoring Project

Thank you for your interest in contributing! This guide explains how to contribute to the Credit Scoring & Creditworthiness Prediction System.

## Code of Conduct

Be respectful, inclusive, and professional. We welcome contributions from everyone regardless of experience level.

## Getting Started

### 1. Fork the Repository

```bash
# Click "Fork" on GitHub
```

### 2. Clone Your Fork

```bash
git clone https://github.com/YOUR-USERNAME/CodeAlpha_CreditScoring.git
cd CodeAlpha_CreditScoring
```

### 3. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 4. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

---

## Development Workflow

### Making Changes

1. **Write Code**
   - Follow PEP 8 style guidelines
   - Add docstrings to all functions
   - Include type hints where possible

2. **Test Your Code**
   ```bash
   # Run all tests
   pytest tests/ -v
   
   # Run specific test file
   pytest tests/test_pipeline.py -v
   
   # Check code coverage
   pytest --cov=src tests/
   ```

3. **Format Your Code**
   ```bash
   # Use black for formatting
   black src/ tests/
   
   # Check for style issues
   flake8 src/ tests/
   
   # Type checking
   mypy src/
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "Add feature: Brief description of changes"
   ```

### Commit Message Guidelines

Use clear, descriptive commit messages:

- ✅ Good: `feat: Add SHAP explainability for model predictions`
- ✅ Good: `fix: Correct SMOTE ratio calculation in preprocessor`
- ❌ Bad: `update code`
- ❌ Bad: `fixed stuff`

Format:
```
[type]: [short description]

[optional detailed explanation]
```

Types:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation update
- `refactor:` Code refactoring
- `test:` Test additions/fixes
- `perf:` Performance improvement
- `chore:` Maintenance tasks

---

## Code Style Guide

### Python Style

Follow PEP 8 with these specifics:

```python
# Good: Descriptive variable names
debt_to_income_ratio = loan_amount / annual_income

# Good: Type hints
def predict(applicant: dict) -> dict:
    """Make prediction for applicant."""
    pass

# Good: Docstrings
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create financial risk indicators.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input applicant data
    
    Returns:
    --------
    pd.DataFrame
        Data with engineered features
    """
    pass

# Good: Meaningful comments
# Apply SMOTE to balance training classes (22% → 50%)
X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
```

### Function Docstrings (Google Style)

```python
def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate model performance on test set.
    
    Computes 6 evaluation metrics and returns detailed results
    with confusion matrix analysis.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained classifier model
    X_test : np.ndarray
        Test features (n_samples, n_features)
    y_test : np.ndarray
        Test labels (n_samples,)
    
    Returns:
    --------
    dict
        Metrics dictionary containing:
        - accuracy: float (0-1)
        - precision: float (0-1)
        - recall: float (0-1)
        - f1_score: float (0-1)
        - roc_auc: float (0-1)
        - pr_auc: float (0-1)
    
    Example:
    --------
    metrics = evaluate_model(xgb_model, X_test, y_test)
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
    """
    pass
```

---

## Types of Contributions

### Bug Reports

If you find a bug:

1. Check existing issues (don't duplicate)
2. Create new issue with:
   - Clear title
   - Steps to reproduce
   - Expected vs actual behavior
   - Python/package versions
   - Error message/traceback

Example:
```
Title: Model fails when applicant age is negative

Steps:
1. Run app.py
2. Input person_age = -5
3. Click "Predict"

Expected: Error message or validation
Actual: Crashes with ValueError

Environment: Python 3.9, scikit-learn 1.3
```

### Feature Requests

Suggest new features by creating an issue with:
- Clear description of feature
- Use case/motivation
- Implementation ideas (optional)
- Examples of similar features

### Code Improvements

Suggest refactoring or optimization:
- Performance improvements
- Code readability
- Test coverage
- Documentation clarity

---

## Pull Request Process

### Before Submitting PR

1. **Test Everything**
   ```bash
   pytest tests/ -v
   black src/ tests/
   flake8 src/ tests/
   python main.py  # Test full pipeline
   streamlit run app.py  # Test dashboard
   ```

2. **Update Documentation**
   - Update README if changing features
   - Add docstrings to new functions
   - Update CHANGELOG.md

3. **Rebase on main**
   ```bash
   git fetch origin
   git rebase origin/main
   ```

### Creating Pull Request

1. Push to your fork
   ```bash
   git push origin feature/your-feature-name
   ```

2. Go to GitHub and click "New Pull Request"

3. Fill PR template:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Related Issue
Closes #[issue-number]

## Changes Made
- Point 1
- Point 2
- Point 3

## Testing
- [ ] All tests pass
- [ ] Added new tests
- [ ] Manual testing complete

## Checklist
- [ ] Code follows PEP 8
- [ ] Docstrings added
- [ ] Tests included
- [ ] No new warnings
```

### PR Review Process

- Maintainers will review your PR
- Address feedback and questions
- Make requested changes with new commits
- PR will be merged once approved

---

## Testing Requirements

### Test Coverage

Aim for >80% coverage:

```bash
pytest --cov=src --cov-report=html tests/
```

### Writing Tests

Example test structure:

```python
import pytest
import numpy as np
import pandas as pd
from src.models.trainer import ModelTrainer
from src.data.preprocessor import preprocess

class TestModelTrainer:
    """Test suite for model training."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)
        return X, y
    
    def test_train_returns_dict(self, sample_data):
        """Test that train() returns model dictionary."""
        X, y = sample_data
        trainer = ModelTrainer()
        models = trainer.train(X, y)
        
        assert isinstance(models, dict)
        assert "XGBoost" in models
    
    def test_model_makes_predictions(self, sample_data):
        """Test that trained model makes valid predictions."""
        X, y = sample_data
        trainer = ModelTrainer()
        models = trainer.train(X, y)
        
        predictions = models["XGBoost"].predict(X[:5])
        assert len(predictions) == 5
        assert all(p in [0, 1] for p in predictions)
```

---

## Documentation Standards

### README.md
- Clear project overview
- Quick start (copy-paste instructions)
- Dataset description
- Results summary
- Installation guide
- Usage examples
- Contributing section

### Module Docstrings
```python
"""
Data preprocessing pipeline.

This module handles all data preprocessing steps including:
- Outlier removal
- Missing value imputation
- Feature engineering
- Encoding categorical variables
- SMOTE balancing
- Feature scaling

Main function: preprocess_and_split()
"""
```

### Function Docstrings
- Purpose (1 sentence)
- Detailed description
- Parameters with types
- Returns with types
- Examples
- Raises (exceptions)

---

## Performance Considerations

When adding new features:

1. **Benchmark Performance**
   ```python
   import time
   
   start = time.time()
   result = new_feature(data)
   elapsed = time.time() - start
   print(f"Execution time: {elapsed:.3f}s")
   ```

2. **Check Memory Usage**
   ```python
   import tracemalloc
   
   tracemalloc.start()
   result = new_feature(data)
   current, peak = tracemalloc.get_traced_memory()
   print(f"Current: {current / 1024 / 1024:.1f}MB")
   ```

3. **Optimize Hot Paths**
   - Profile code to find bottlenecks
   - Vectorize with NumPy/Pandas
   - Consider Cython for tight loops

---

## Adding Dependencies

If you need a new package:

1. **Test locally first**
   ```bash
   pip install new-package
   # Verify it works
   ```

2. **Update requirements.txt**
   ```
   new-package>=1.0.0
   ```

3. **Explain in PR**
   - Why is it needed?
   - Are there alternatives?
   - What's the version constraint?

4. **Keep it minimal**
   - Avoid bloated dependencies
   - Prefer lightweight alternatives
   - Consider maintenance burden

---

## Reporting Security Issues

**Do not** create public issues for security vulnerabilities.

Instead, email: [maintainer-email]

Include:
- Description of vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if available)

---

## Getting Help

- **Questions:** Create a Discussion issue
- **Bugs:** Create a Bug Report issue
- **Features:** Create a Feature Request issue
- **Chat:** Connect on LinkedIn

---

## Contributor Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Recognized in release notes
- Credited in project documentation

---

## License

By contributing, you agree your code will be licensed under the MIT License (see LICENSE file).

---

## Questions?

- Read existing PRs for examples
- Check project documentation
- Ask in issues before starting big changes
- Reach out to maintainers

---

**Thank you for contributing!** 🎉

Your improvements help make this project better for everyone.

