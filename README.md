# Adversarial Attacks on Bandits

# Repository setup

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.py

To run experiments:

    python3 jun_comparison.py
    python3 liu_comparison.py

Parameters can be adjusted from `config.py` and specific experiment file.

For compatibility with mpl

    pip install tikzplotlib=='0.10.1'