# Contributing

Thank you for your interest in contributing to Crypto-transformer-predictor!

## Getting Started

1. Fork the repository and create your feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Install dependencies and run tests locally:
   ```bash
   pip install -r requirements.txt
   pytest
   ```

## Coding Standards

We follow these principles to ensure clean, maintainable code:

1. **Clear Naming** – Use intent-revealing names. Avoid abbreviations and magic numbers.
2. **Single Responsibility** – Each module, class, and function should do one thing.
3. **Open/Closed** – Extend behavior via abstractions; do not modify existing code when adding features.
4. **Liskov Substitution** – Subclasses must honor contracts of their base classes.
5. **Interface Segregation** – Define small, focused interfaces.
6. **Dependency Inversion** – Depend on abstractions, not concretes. Use dependency injection.
7. **DRY** – Eliminate duplication; one representation of every piece of logic.
8. **Small Functions** – Functions should be ≤20 lines, few parameters, no flag arguments, and no side effects.
9. **Self‑Documenting Code** – Write clear, formatted code per Python conventions; minimize comments.
10. **Error Handling** – Raise meaningful exceptions; do not suppress errors silently.
11. **Testing** – Write automated tests for every unit of logic. All tests must pass before merging.
12. **Continuous Refactoring** – Leave the code base cleaner than you found it.

## Pull Request Process

1. Ensure your branch is up to date with `main`.
2. Run `pytest` and linters (`flake8`, `black`) to confirm no errors.
3. Open a Pull Request with a clear title and description of your changes.
4. Address review comments and make any requested changes.
5. Once approved and all checks pass, your PR will be merged.

## Code Reviews

- Reviewers may suggest changes for code style, test coverage, or architectural improvements.
- Discussions should be constructive and focused on the code, not the author.

## Thank You

We appreciate your contributions! Feel free to ask questions if you need guidance.