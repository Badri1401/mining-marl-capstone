# Contributing

Thanks for your interest in this project! Contributions, issues, and suggestions
are welcome.

## Getting set up

```bash
git clone https://github.com/Badri1401/mining-marl-capstone.git
cd mining-marl-capstone
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Ways to contribute

- **Report a bug** — open an issue using the *Bug report* template.
- **Request a feature** — open an issue using the *Feature request* template.
- **Submit code** — fork, branch, and open a pull request.

## Pull request checklist

1. Create a topic branch: `git checkout -b feature/short-description`.
2. Keep changes focused; match the existing code style.
3. Make sure everything still byte-compiles:
   ```bash
   python -m compileall -q src analysis simulation
   ```
4. If you touched the environment, training, or analysis, verify a short run and
   note what you checked in the PR description.
5. Open the PR against `main` and fill in the template.

## Code style

- Python 3.10, 4-space indentation.
- Keep functions small and readable; prefer clear names over cleverness.
- `ruff` is used for linting (see `pyproject.toml`); run `ruff check src analysis`
  before pushing.

## Reporting security issues

Please do not file public issues for security-sensitive problems — see
[`SECURITY.md`](SECURITY.md) if present, or contact the maintainers directly.
