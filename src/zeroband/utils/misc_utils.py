
class IntRef:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"IntRef({self.value})"

    def __iadd__(self, other):
        self.value += other
        return self

    def __add__(self, other):
        return IntRef(self.value + other)

    def __isub__(self, other):
        self.value -= other
        return self

    def __sub__(self, other):
        return IntRef(self.value - other)

    def __int__(self):
        return self.value

    def __gt__(self, other):
        return self.value > other

    def __lt__(self, other):
        return self.value < other

    def __eq__(self, other):
        return self.value == other

    def __ne__(self, other):
        return self.value != other

    def __ge__(self, other):
        return self.value >= other

    def __le__(self, other):
        return self.value <= other