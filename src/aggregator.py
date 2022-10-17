from enum import Enum, unique, auto


@unique
class Aggregator(Enum):
    SUM = auto()
    MAX = auto()
    MIN = auto()
    SUB = auto()
    SUM_PROD = auto()
    PROD = auto()
    COUNT = auto()
    DISTINCT_COUNT = auto()
    IDENTITY = auto()

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other

        if isinstance(other, Aggregator):
            return self is other

        return False


@unique
class Annotation(Enum):
    NULL = auto()
    NOT_NULL = auto()
    NOT_GREATER = auto()
    GREATER = auto()
    DISTINCT = auto()
    NOT_DISTINCT = auto()
    IN = auto()
    NOT_IN = auto()

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other

        if isinstance(other, Annotation):
            return self is other

        return False


if __name__ == "__main__":
    a = (1, Aggregator.SUM)
    db = Aggregator
    print(a)
    print(a[1] == Aggregator.SUM)
