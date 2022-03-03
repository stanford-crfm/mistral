from tests import run_tests
from src.util.paths import get_file_extension


def test_get_file_extension():
    assert get_file_extension("some.json") == "json"
    assert get_file_extension("some/path/some.txt") == "txt"


if __name__ == "__main__":
    run_tests()
