import time
import pytest
from multiprocessing import Process
from zeroband.utils.shared_int_deque import SharedIntDeque


@pytest.mark.parametrize("size", [3, 5, 10])
def test_append(size):
    deque = SharedIntDeque(max_size=size)
    for i in range(size):
        deque.append(i)
    deque.append(size)  # This should evict the oldest element when the deque is full
    assert deque[0] == 1  # The first item should be the second element added if size is 3 or more
    assert deque[-1] == size
    assert len(deque) == size


@pytest.mark.parametrize("size", [3, 5, 10])
def test_appendleft(size):
    deque = SharedIntDeque(max_size=size)
    for i in range(size):
        deque.appendleft(i)
    deque.appendleft(size)  # This should evict the oldest element when the deque is full
    assert deque[0] == size  # The first item should be the last added
    assert deque[-1] == 1  # The last item should be the second-to-last item added
    assert len(deque) == size


@pytest.mark.parametrize("size", [3, 5, 10])
def test_pop(size):
    deque = SharedIntDeque(max_size=size)
    for i in range(size):
        deque.append(i)
    item = deque.pop()
    assert item == size - 1
    assert len(deque) == size - 1
    assert deque[-1] == size - 2


@pytest.mark.parametrize("size", [3, 5, 10])
def test_popleft(size):
    deque = SharedIntDeque(max_size=size)
    for i in range(size):
        deque.append(i)
    item = deque.popleft()
    assert item == 0
    assert len(deque) == size - 1
    assert deque[0] == 1


@pytest.mark.parametrize("size", [3, 5, 10])
def test_index_error_on_empty_pop(size):
    deque = SharedIntDeque(max_size=size)
    with pytest.raises(IndexError):
        deque.pop()


@pytest.mark.parametrize("size", [3, 5, 10])
def test_index_error_on_empty_popleft(size):
    deque = SharedIntDeque(max_size=size)
    with pytest.raises(IndexError):
        deque.popleft()


@pytest.mark.parametrize("size", [3, 5, 10])
def test_getitem_out_of_range(size):
    deque = SharedIntDeque(max_size=size)
    deque.append(1)
    with pytest.raises(IndexError):
        _ = deque[1]


@pytest.mark.parametrize("size", [3, 5, 10])
def test_setitem(size):
    deque = SharedIntDeque(max_size=size)
    deque.append(1)
    deque[0] = 10
    assert deque[0] == 10


@pytest.mark.parametrize("size", [3, 5, 10])
def test_mp_append(size):
    deque = SharedIntDeque(max_size=size)

    def foo(_deque):
        for i in range(size):
            _deque.append(i)
        print(f"meow {_deque}")
        time.sleep(1)
        print(f"woof {_deque}")
        assert _deque[0] == 1  # The first item should be the second element added if size is 3 or more
        assert _deque[-1] == size

    proc = Process(target=foo, args=(deque,))
    proc.start()
    time.sleep(0.5)
    assert len(deque) == size

    deque.append(size)  # This should evict the oldest element when the deque is full
    assert deque[0] == 1  # The first item should be the second element added if size is 3 or more
    assert deque[-1] == size
    assert len(deque) == size
    proc.join()
