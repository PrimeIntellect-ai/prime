from multiprocessing import Value, Array, Lock
from multiprocessing.sharedctypes import Synchronized, SynchronizedArray
from typing import List
from collections.abc import MutableSequence


class SharedIntDeque(MutableSequence):
    """
    A thread-safe, shared deque implementation using multiprocessing primitives
    that can be safely accessed by multiple processes.

    Args:
        max_size (int): The maximum number of elements the deque can hold.

    Example:
        >>> deque = SharedIntDeque(4)
        >>> deque.append(1)
        >>> deque.append(2)
        >>> deque.append(3)
        >>> deque.append(4)
        >>> deque.append(5)
        >>> deque
        SharedIntDeque([2, 3, 4, 5])
        >>> deque.popleft()
        2
        >>> deque.pop()
        5
        >>> deque
        SharedIntDeque([3, 4])
    """

    def __init__(self, max_size: int) -> None:
        """
        Initializes a SharedDeque instance with the specified maximum size.

        Args:
            max_size (int): The maximum number of elements the deque can hold.
        """
        self.max_size: int = max_size
        self.array: SynchronizedArray = Array("i", [0] * max_size)
        self.size: Synchronized = Value("i", 0)
        self.front: Synchronized = Value("i", 0)
        self.rear: Synchronized = Value("i", 0)
        self.lock = Lock()

    def append(self, item: int) -> None:
        """
        Adds an item to the end of the deque. If the deque is full, evicts the oldest value.

        Args:
            item (int): The item to be added.
        """
        with self.lock:
            if self.size.value == self.max_size:
                # Evict the oldest value by moving the front forward
                self.front.value = (self.front.value + 1) % self.max_size
                self.size.value -= 1
            self.array[self.rear.value] = item
            # Inline increment
            self.rear.value = (self.rear.value + 1) % self.max_size
            self.size.value += 1

    def appendleft(self, item: int) -> None:
        """
        Adds an item to the front of the deque. If the deque is full, evicts the oldest value.

        Args:
            item (int): The item to be added.
        """
        with self.lock:
            if self.size.value == self.max_size:
                # Evict the oldest value by moving the rear backward
                self.rear.value = (self.rear.value - 1 + self.max_size) % self.max_size
                self.size.value -= 1
            # Inline decrement
            self.front.value = (self.front.value - 1 + self.max_size) % self.max_size
            self.array[self.front.value] = item
            self.size.value += 1

    def pop(self) -> int:
        """
        Removes and returns an item from the end of the deque.

        Returns:
            int: The item removed from the end.
        Raises:
            IndexError: If the deque is empty.
        """
        with self.lock:
            if self.size.value == 0:
                raise IndexError("Deque is empty")
            # Inline decrement
            self.rear.value = (self.rear.value - 1 + self.max_size) % self.max_size
            item: int = self.array[self.rear.value]
            self.size.value -= 1
            return item

    def popleft(self) -> int:
        """
        Removes and returns an item from the front of the deque.

        Returns:
            int: The item removed from the front.
        Raises:
            IndexError: If the deque is empty.
        """
        with self.lock:
            if self.size.value == 0:
                raise IndexError("Deque is empty")
            item: int = self.array[self.front.value]
            # Inline increment
            self.front.value = (self.front.value + 1) % self.max_size
            self.size.value -= 1
            return item

    def locked_peek(self, indexes: List[int]) -> List[int]:
        """
        Returns a list of items at the specified indexes in the deque.

        Args:
            indexes (List[int]): The list of indexes to peek at.
        Returns:
            List[int]: The list of items at the specified indexes.
        Raises:
            IndexError: If any of the indexes are out of range.
        """
        with self.lock:
            return [self[i] for i in indexes]

    def __len__(self) -> int:
        with self.lock:
            return self.size.value

    def __getitem__(self, index: int) -> int:
        if index < 0:
            index += self.size.value
        if not 0 <= index < self.size.value:
            raise IndexError("Index out of range")
        real_index: int = (self.front.value + index) % self.max_size
        return self.array[real_index]

    def __setitem__(self, index: int, value: int) -> None:
        with self.lock:
            if index < 0:
                index += self.size.value
            if not 0 <= index < self.size.value:
                raise IndexError("Index out of range")
            real_index: int = (self.front.value + index) % self.max_size
            self.array[real_index] = value

    def __delitem__(self, index: int) -> None:
        raise NotImplementedError("Deletion by index is not supported for this deque")

    def insert(self, index: int, value: int) -> None:
        raise NotImplementedError("Insertion by index is not supported for this deque")

    def __repr__(self) -> str:
        items: List[int] = [self[i] for i in range(len(self))]
        return f"{self.__class__.__name__}({items})"
