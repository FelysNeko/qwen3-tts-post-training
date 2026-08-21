"""Scorer worker entrypoint. The real logic lives in scorer.serve; this thin
launcher lets ScorerClient spawn `python main.py` without path hacks.
"""

from scorer.serve import main

if __name__ == "__main__":
    main()
