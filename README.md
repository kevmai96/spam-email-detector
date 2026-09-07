# Spam email detector

# What is it
This is a script that detects spam vs legitimate emails using machine learning. It trains four models:

  - Naive Bayes
  - Logistic Regression
  - Random Forest
  - Linear SVM

The script compares their performance, shows the top words that trigger spam detection, and lets you test your own emails interactively.

# How to test:
1. Setup environment
python3.12 -m venv env
source env/bin/activate

2. Install requirements:
pip install -r requirements.txt
    
3. Run: python app.py

# IMPORTANT DISCLAIMER
This project is for practice and learning purposes only.
The dataset is small and synthetic; results may not reflect real-world email detection.
Do not rely on this script for actual spam filtering, security, or financial decisions.
Accuracy may vary significantly if used with real emails.

The developers are not responsible for any consequences arising from using this tool on actual emails.