import pytest


def _option_exists(parser, name):
    """Check if a pytest option is already registered."""
    try:
        for opt in parser._anonymous.options:
            if name in opt.names():
                return True
        for group in parser._groups:
            for opt in group.options:
                if name in opt.names():
                    return True
    except AttributeError:
        pass
    return False


def pytest_addoption(parser):
    # Guard against duplicate registration when collected with test/conftest.py
    if not _option_exists(parser, "--has-dolfin"):
        parser.addoption("--has-dolfin", type=int, default=0)
    if not _option_exists(parser, "--has-dolfinx"):
        parser.addoption("--has-dolfinx", type=int, default=0)
    if not _option_exists(parser, "--has-exafmm"):
        parser.addoption("--has-exafmm", type=int, default=0)
    if not _option_exists(parser, "--dolfin-books-only"):
        parser.addoption("--dolfin-books-only", type=int, default=0)


@pytest.fixture
def has_dolfin(request):
    return request.config.getoption("--has-dolfin") > 0


@pytest.fixture
def has_dolfinx(request):
    return request.config.getoption("--has-dolfinx") > 0


@pytest.fixture
def has_exafmm(request):
    return request.config.getoption("--has-exafmm") > 0


@pytest.fixture
def dolfin_books_only(request):
    return request.config.getoption("--dolfin-books-only") > 0
