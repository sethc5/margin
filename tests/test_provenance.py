from margin.provenance import new_id, are_correlated, merge


class TestNewId:
    def test_returns_string(self):
        assert isinstance(new_id(), str)

    def test_unique(self):
        ids = {new_id() for _ in range(100)}
        assert len(ids) == 100

    def test_length(self):
        assert len(new_id()) == 8


class TestAreCorrelated:
    def test_shared_ancestor(self):
        shared = new_id()
        assert are_correlated([shared, "a"], [shared, "b"]) is True

    def test_no_shared_ancestor(self):
        assert are_correlated(["a", "b"], ["c", "d"]) is False

    def test_empty_lists(self):
        assert are_correlated([], []) is False

    def test_one_empty(self):
        assert are_correlated(["a"], []) is False


class TestMerge:
    def test_contains_both(self):
        m = merge(["a"], ["b"])
        assert "a" in m
        assert "b" in m

    def test_adds_new_id(self):
        m = merge(["a"], ["b"])
        assert len(m) == 3  # a, b, new

    def test_deduplicates(self):
        m = merge(["a", "b"], ["b", "c"])
        assert sorted([x for x in m if x in ("a", "b", "c")]) == ["a", "b", "c"]
