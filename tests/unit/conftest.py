"""Test configuration for pytest."""


def pytest_configure(config):
    """Configure pytest options."""
    # Instead of using addinivalue_line, we'll set the option directly
    # This won't try to modify an existing list option
    config._inicache["asyncio_default_fixture_loop_scope"] = "function"
