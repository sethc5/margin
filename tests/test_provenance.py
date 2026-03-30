from margin.provenance import new_id, are_correlated, merge, ProvenanceGraph


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


class TestProvenanceGraphCompress:
    def test_compress_noop_when_under_limit(self):
        pg = ProvenanceGraph()
        for _ in range(10):
            pg.create_root("step")
        pg.compress(max_nodes=20)
        assert len(pg.nodes) == 10

    def test_compress_prunes_to_max(self):
        pg = ProvenanceGraph()
        for _ in range(100):
            pg.create_root("step")
        pg.compress(max_nodes=50)
        assert len(pg.nodes) == 50

    def test_compress_keeps_newest(self):
        pg = ProvenanceGraph()
        ids = [pg.create_root(f"step:{i}") for i in range(10)]
        pg.compress(max_nodes=5)
        for node_id in ids[-5:]:
            assert node_id in pg.nodes
        for node_id in ids[:5]:
            assert node_id not in pg.nodes

    def test_compress_cleans_dangling_source_ids(self):
        pg = ProvenanceGraph()
        root = pg.create_root("root")
        child = pg.derive("child", [root])
        pg.compress(max_nodes=1)  # prune root, keep child
        assert root not in pg.nodes
        assert child in pg.nodes
        assert pg.nodes[child].source_ids == []  # dangling ref removed

    def test_compress_returns_self(self):
        pg = ProvenanceGraph()
        for _ in range(10):
            pg.create_root("step")
        result = pg.compress(max_nodes=5)
        assert result is pg
