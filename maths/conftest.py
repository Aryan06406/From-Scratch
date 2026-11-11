"""
Shared pytest / Hypothesis configuration.
"""
import os
from hypothesis import HealthCheck, settings

settings.register_profile(
    "adv", max_examples=150, deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    print_blob=True,
)
settings.register_profile(
    "thorough", max_examples=1500, deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    print_blob=True,
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "adv"))

def pytest_configure(config):
    markers = {
        "unit": "plain example based unit tests",
        "boundary": "boundary value / edge case tests",
        "property": "Hypothesis property based tests",
        "invariance": "invariance / symmetry / metamorphic tests",
        "numeric": "floating point precision & tolerance tests",
        "reference": "differential tests against numpy / scipy / exact Fractions",
        "scaling": "asymptotic complexity / convergence order tests",
        "stress": "ill conditioned / extreme magnitude / large size tests",
        "performance": "performance and algorithmic scaling tests",
    }
    for name, desc in markers.items():
        config.addinivalue_line(
            "markers",
            f"{name}: {desc}"
        )
