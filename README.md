# Adversarial Attacks on Bandits

# Repository setup

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.py

# Repository structure
.
├── libmab
│   ├── attackers.py
│   ├── envs
│   │   ├── combinatorial.py:
│   │   └── stochastic.py
│   ├── learners.py
│   ├── utils.py
│   └── visualization.py
├── LICENSE
├── README.md
└── requirements.txt

# TODO
- [ ] env reward seed for round t
- [ ] parallel execution
- [ ] saving experiment
- [ ] division (stochastic, adversarial)
- [ ] refactor envs
- [ ] add adversarial bandits