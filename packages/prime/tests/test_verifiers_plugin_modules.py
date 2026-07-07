"""The plugin's module paths must exist in the installed verifiers.

Prime shells out to these modules by name; a rename on the verifiers side otherwise
surfaces only as a ModuleNotFoundError at `prime gepa run` / `prime eval` time.
"""

import importlib.util

import pytest
from prime_cli.verifiers_plugin import PrimeVerifiersPlugin

pytest.importorskip("verifiers")


def test_plugin_modules_resolve_in_installed_verifiers():
    plugin = PrimeVerifiersPlugin()
    modules = (
        plugin.eval_module,
        plugin.init_module,
        plugin.validate_module,
        plugin.gepa_module,
    )
    missing = [module for module in modules if importlib.util.find_spec(module) is None]
    assert not missing, f"plugin points at modules verifiers no longer ships: {missing}"
