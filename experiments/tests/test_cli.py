from typer.testing import CliRunner

from lpdp.cli import app

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "experiment pipeline" in result.output.lower() or "Usage" in result.output
