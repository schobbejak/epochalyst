from epochlib.pipeline import Block, Base, SequentialSystem, ParallelSystem
from tests.pipeline.util import remove_cache_files
from pathlib import Path


class Test_Base:
    def test_init(self):
        base = Base()
        assert base is not None

    def test_set_hash(self):
        base = Base()
        prev_hash = base.get_hash()
        base.set_hash("prev_hash")
        assert base.get_hash() != prev_hash

    def test_get_children(self):
        base = Base()
        assert base.get_children() == []

    def test_get_parent(self):
        base = Base()
        assert base.get_parent() is None

    def test__set_parent(self):
        base = Base()
        base.set_parent(base)
        assert base.get_parent() == base

    def test__set_children(self):
        base = Base()
        base.set_children([base])
        assert base.get_children() == [base]

    def test__repr_html_(self):
        base = Base()
        assert (
            base._repr_html_()
            == "<div style='border: 1px solid black; padding: 10px;'><p><strong>Class:</strong> Base</p><ul><li><strong>Hash:</strong> a00a595206d7eefcf0e87acf6e2e22ee</li><li><strong>Parent:</strong> None</li><li><strong>Children:</strong> None</li></ul></div>"
        )

    def test_save_to_html(self):
        html_path = Path("./tests/cache/test_html.html")
        Path("./tests/cache/").mkdir(parents=True, exist_ok=True)
        base = Base()
        base.save_to_html(html_path)
        assert Path.exists(html_path)
        remove_cache_files()


class TestBlock:
    def test_block_init(self):
        block = Block()
        assert block is not None

    def test_block_set_hash(self):
        block = Block()
        block.set_hash("")
        hash1 = block.get_hash()
        assert hash1 != ""
        block.set_hash(hash1)
        hash2 = block.get_hash()
        assert hash2 != ""
        assert hash1 != hash2

    def test_block_get_hash(self):
        block = Block()
        block.set_hash("")
        hash1 = block.get_hash()
        assert hash1 != ""

    def test__repr_html_(self):
        block_instance = Block()

        html_representation = block_instance._repr_html_()

        assert html_representation is not None


class TestSequentialSystem:
    def test_system_init(self):
        system = SequentialSystem()
        assert system is not None

    def test_system_hash_no_steps(self):
        system = SequentialSystem()
        assert system.get_hash() == ""

    def test_system_hash_with_1_step(self):
        block1 = Block()

        system = SequentialSystem([block1])
        assert system.get_hash() != ""
        assert block1.get_hash() == system.get_hash()

    def test_system_hash_with_2_steps(self):
        block1 = Block()
        block2 = Block()

        system = SequentialSystem([block1, block2])
        assert system.get_hash() != block1.get_hash()
        assert (
            system.get_hash() == block2.get_hash() != ""
        )

    def test_system_hash_with_3_steps(self):
        block1 = Block()
        block2 = Block()
        block3 = Block()

        system = SequentialSystem([block1, block2, block3])
        assert system.get_hash() != block1.get_hash()
        assert system.get_hash() != block2.get_hash()
        assert block1.get_hash() != block2.get_hash()
        assert (
            system.get_hash() == block3.get_hash() != ""
        )

    def test__repr_html_(self):
        block_instance = Block()
        system_instance = SequentialSystem([block_instance, block_instance])
        html_representation = system_instance._repr_html_()

        assert html_representation is not None


class TestParallelSystem:
    def test_parallel_system_init(self):
        parallel_system = ParallelSystem()
        assert parallel_system is not None

    def test_parallel_system_hash_no_steps(self):
        system = ParallelSystem()
        assert system.get_hash() == ""

    def test_parallel_system_hash_with_1_step(self):
        block1 = Block()

        system = ParallelSystem([block1])
        assert system.get_hash() != ""
        assert block1.get_hash() == system.get_hash()

    def test_parallel_system_hash_with_2_steps(self):
        block1 = Block()
        block2 = Block()

        system = ParallelSystem([block1, block2])
        assert system.get_hash() != block1.get_hash()
        assert block1.get_hash() == block2.get_hash()
        assert system.get_hash() != block2.get_hash()
        assert system.get_hash() != ""

    def test_parallel_system_hash_with_3_steps(self):
        block1 = Block()
        block2 = Block()
        block3 = Block()

        system = ParallelSystem([block1, block2, block3])
        assert system.get_hash() != block1.get_hash()
        assert system.get_hash() != block2.get_hash()
        assert system.get_hash() != block3.get_hash()
        assert block1.get_hash() == block2.get_hash() == block3.get_hash()
        assert system.get_hash() != ""

    def test_parallel_system__repr_html_(self):
        block_instance = Block()
        system_instance = ParallelSystem([block_instance, block_instance])
        html_representation = system_instance._repr_html_()

        assert html_representation is not None
