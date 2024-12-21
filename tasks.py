from invoke.context import Context
from invoke.tasks import task

@task
def test(ctx: Context) -> None:
    """Run all tests with pytest."""
    ctx.run("poetry run pytest src/tests/")

@task
def lint(ctx: Context) -> None:
    """Run ruff to lint and format the code."""
    ctx.run("ruff check .")

@task
def format(ctx: Context) -> None:
    """Run ruff to format the code."""
    ctx.run("ruff check . --fix")

@task
def type(ctx: Context) -> None:
    """Check the types with mypy."""
    ctx.run("mypy src")

@task
def docs(ctx: Context) -> None:
    """Generate HTML documentation with pdoc."""
    ctx.run("pdoc email_classifier --output-dir docs")


