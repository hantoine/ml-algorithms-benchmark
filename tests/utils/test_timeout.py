import time
from utils.timeout import set_timeout, TimeoutError


def function():
    i = 1
    while True:
        i = (i + 1) % 20


def test_set_timeout():
    function_w_timeout = set_timeout(function, 5)
    start_time = time.time()
    try:
        function_w_timeout()
    except TimeoutError:
        pass
    elapsed_time = time.time() - start_time
    assert 4 < elapsed_time < 6
