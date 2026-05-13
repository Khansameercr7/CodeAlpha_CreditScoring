# Quick Start Guide - 5 Minutes to Credit Scoring Dashboard

Get the interactive dashboard running in under 5 minutes!

---

## Prerequisites

- Python 3.8+ installed
- Git installed
- ~200MB disk space

---

## Step 1: Clone Repository (1 minute)

```bash
git clone https://github.com/YOUR-USERNAME/CodeAlpha_CreditScoring.git
cd CodeAlpha_CreditScoring
```

---

## Step 2: Install Dependencies (2 minutes)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Step 3: Run Dashboard (2 minutes)

```bash
streamlit run app.py
```

The dashboard will open at: `http://localhost:8501`

---

## Using the Dashboard

### Home Page
- Overview of the credit scoring system
- Key statistics
- Pipeline explanation

### Predict Risk Page
**Try a prediction immediately:**

1. Fill in applicant details:
   - **Age:** 35
   - **Income:** $85,000
   - **Loan Amount:** $15,000
   - **Loan Grade:** A (or A-G)
   - **Home Ownership:** RENT

2. Click **"Get Risk Assessment"**

3. View:
   - Default probability (%)
   - Risk category (Low/Medium/High)
   - Recommendation (Approve/Reject)
   - Key risk factors

### Try Different Scenarios

**Low Risk Applicant:**
- Age: 45, Income: $120k, Loan: $10k, Grade: A, emp_length: 20

**High Risk Applicant:**
- Age: 23, Income: $25k, Loan: $20k, Grade: G, emp_length: 1

**Adjust Risk Threshold:**
- Use slider (0.0 - 1.0) to change sensitivity
- Lower = More approvals, higher risk
- Higher = Fewer approvals, safer portfolio

---

## Model Comparison Page

See how 4 different models perform:
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

**Key Metrics:**
- Accuracy: Overall prediction correctness
- ROC-AUC: How well model ranks risk
- F1-Score: Balance between precision & recall

XGBoost wins with ROC-AUC = 0.937 ⭐

---

## Feature Analysis Page

Explore which factors drive credit decisions:
- **Debt-to-Income Ratio** (most important)
- **Loan Amount**
- **Employment Stability**
- **Interest Rate**
- **Loan Grade**

Understand why applicants get approved or rejected.

---

## SHAP Explainability Page

**Waterfall Plot:**
- Shows exactly how features contributed to a decision
- Base value → Feature impacts → Final prediction

**Summary Plot:**
- Shows global feature importance
- Color: Red = increase risk, Blue = decrease risk

---

## Dataset Info Page

Explore the dataset:
- Raw data statistics
- Class distribution (78% good, 22% default)
- Missing values analysis
- Feature correlations

---

## Common Questions

### Q: Can I use my own data?

**Short term:** Edit the predict page to accept custom CSV  
**Long term:** Contact project maintainers for integration

### Q: How accurate is the model?

- ROC-AUC: 0.937 (catches 93.7% of risky applicants)
- Recall: 0.81 (catches 81% of actual defaults)
- This is **production-ready accuracy** for lending

### Q: What if I get an error?

**Check:**
1. Python version: `python --version` (needs 3.8+)
2. Dependencies installed: `pip list`
3. Data file exists: `data/credit_data.csv`

**Reinstall if needed:**
```bash
pip install --upgrade -r requirements.txt
```

### Q: Can I deploy this online?

**Yes! See DEPLOYMENT.md for options:**
- Streamlit Cloud (easiest)
- Docker + AWS
- API + FastAPI
- Google Cloud

### Q: How do I run the ML pipeline?

```bash
python main.py
```

This will:
1. Load data (32,581 records)
2. Preprocess & engineer features
3. Train 4 models
4. Evaluate performance
5. Generate visualizations
6. Save models to outputs/models/

Takes ~2-3 minutes.

### Q: Where are the predictions coming from?

- XGBoost model (best performer)
- 32 input features after encoding
- Trained on 32,581 real loan applications
- 93.7% ROC-AUC on test set

---

## File Structure

```
credit_scoring/
├── app.py                    ← Streamlit dashboard
├── main.py                   ← ML pipeline
├── requirements.txt          ← Python dependencies
├── data/
│   └── credit_data.csv      ← Raw dataset
├── outputs/
│   ├── models/              ← Trained models
│   ├── figures/             ← Visualizations
│   └── shap/                ← SHAP plots
└── src/                     ← Source code
    ├── data/                ← Loading & preprocessing
    ├── models/              ← Training & prediction
    ├── evaluation/          ← Metrics & testing
    ├── visualization/       ← Charts & SHAP
    └── utils/               ← Helpers & logging
```

---

## Keyboard Shortcuts (Streamlit)

- **r** - Rerun app
- **c** - Clear cache
- **q** - Quit

---

## Tips for Best Experience

1. **First Visit:** Go through all 6 dashboard pages
2. **Try Predictions:** Use the example applicants provided
3. **Explore SHAP:** Understand feature contributions
4. **Check Features:** See what matters in credit decisions
5. **Read Results:** Understand model comparison

---

## Next Steps

### For Learning:
- Read DEPLOYMENT.md to learn deployment options
- Read CONTRIBUTING.md to contribute improvements
- Check src/ code to understand ML pipeline

### For Using:
- Deploy to Streamlit Cloud (free)
- Use for credit assessment demos
- Build on top of this foundation

### For Internship Submission:
1. ✅ Dashboard works
2. ✅ Models trained and saved
3. ✅ Code on GitHub
4. ✅ Share on LinkedIn
5. ✅ Submit to CodeAlpha

---

## Troubleshooting

### Port 8501 Already in Use

```bash
streamlit run app.py --server.port 8502
```

### Out of Memory

```bash
# Reduce sample size in config.yaml
# Or use a machine with more RAM
```

### Model Prediction Errors

```bash
# Regenerate models
python main.py

# Verify outputs/models/ has all 5 files:
# - Logistic Regression.pkl
# - Decision Tree.pkl
# - Random Forest.pkl
# - XGBoost.pkl
# - scaler.pkl
# - feature_names.pkl
```

---

## Performance Tips

- **First load:** Slower (model loading)
- **Subsequent predictions:** Fast (cached)
- **Batch predictions:** Contact maintainers

---

## Security Notes

- Models are pre-trained (no data exposure)
- Dashboard is local by default
- Predictions stay on your machine
- No data sent to external services

---

## What to Share

After getting it running:

1. **Screenshot of Dashboard Home**
   - Shows the UI and key stats

2. **Sample Prediction Result**
   - Shows a low-risk and high-risk applicant

3. **SHAP Explainability**
   - Shows transparent AI decision-making

4. **GitHub Link**
   - https://github.com/YOUR-USERNAME/CodeAlpha_CreditScoring

5. **LinkedIn Post**
   - Share your achievement with @CodeAlpha

---

## Support

Need help?
- Check README.md for detailed docs
- Review VIDEO_SCRIPT.md for project explanation
- Check GITHUB_SETUP.md for GitHub info
- Read DEPLOYMENT.md for deployment options
- Open issue on GitHub

---

**Congratulations!** 🎉

You now have a production-ready credit scoring system running. 

Next: Deploy it online and share with your network!

