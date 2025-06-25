# Contributing to EarthDaily Python Client

This document will guide you through the steps required to set up your development environment, develop, test, and contribute to the project.

## Prerequisites

Ensure you have the following installed on your machine:
- [Python 3.10+](https://www.python.org/downloads/)
- [Poetry](https://python-poetry.org/docs/#installation) (for dependency management)
- [Tox](https://tox.readthedocs.io/en/latest/) (for running tests in multiple environments)
- [Pre-commit](https://pre-commit.com/) (for managing pre-commit hooks)

## Setting Up the Development Environment

1. **Clone the repository**:
   ```bash
   git clone git@github.com:earthdaily/earthdaily-python-client.git
   cd earthdaily-python-client
   ```

2. **Install dependencies using Poetry**:
   ```bash
   poetry install --all-extras
   ```

3. **Set up pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## Development Workflow

1. **Create a new branch for your feature or bugfix**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes and commit them**:
   ```bash
   git add .
   git commit -m "your commit message"
   ```

3. **Run tests locally**:
   ```bash
   poetry run tox
   ```

4. **Push your changes and create a merge request**:
   ```bash
   git push origin feature/your-feature-name
   ```
   Then, go to the GitHub repository and create a new pull request.

## Running Tests

We use `tox` to run tests across multiple Python versions. To run all tests:

```bash
poetry run tox
```

To run tests for a specific Python version:

```bash
poetry run tox -e py310  # for Python 3.10
poetry run tox -e py311  # for Python 3.11
poetry run tox -e py312  # for Python 3.12
```

To run specific test environments:

```bash
poetry run tox -e lint     # Run linting
poetry run tox -e format   # Check formatting
poetry run tox -e mypy     # Run type checking
```

## Code Style and Linting

We use `ruff` for code linting and formatting. Pre-commit hooks are configured to run automatically.

To check your code manually:

```bash
poetry run tox -e lint
```

To check formatting:

```bash
poetry run tox -e format
```

To run type checking:

```bash
poetry run tox -e mypy
```

## Building the Package

To build the package, run:

```bash
poetry build
```

This will create both wheel and source distributions in the `dist/` directory.

## Publishing the Package

Package publishing is handled by our CI/CD pipeline. Manual publishing should only be done if absolutely necessary and after consulting with the team.

## Continuous Integration

Our project uses GitHub Actions for continuous integration. The configuration can be found in `.github/workflows/` at the root of the repository. It automatically runs tests, linting, and builds for each pull request.

## Documentation

When adding new features or making significant changes, please update the documentation accordingly. This includes updating the README.md file and any relevant docstrings in the code.

## Reporting Issues

If you encounter any bugs or have feature requests, please create an issue in our GitHub repository. Provide as much detail as possible, including steps to reproduce for bugs, and clear descriptions for feature requests.

## Code of Conduct

Please note that this project is released with a Contributor Code of Conduct. By participating in this project you agree to abide by its terms.

Thank you for contributing to EarthDaily Python Client!
