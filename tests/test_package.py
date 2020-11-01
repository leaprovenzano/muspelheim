# These tests can be run on an almost empty package.
# Useful for testing that the basics work before anything else exists.
import re

import muspelheim


def is_valid_semver(version: str) -> bool:
    """return True if a value is a valid semantic version
    """
    match = re.match(r'^[0-9]+\.[0-9]+\.[0-9]+(-([0-9a-z]+(\.[0-9a-z]+)*))?$', version)
    return match is not None


def test_version_is_valid_semver():
    assert is_valid_semver(muspelheim.__version__)
