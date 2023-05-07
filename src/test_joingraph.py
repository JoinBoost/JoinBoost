import unittest
import pandas as pd
from joinboost.joingraph import JoinGraph, JoinGraphException
import test_utils

class TestJoingraph(unittest.TestCase):
    def test_cycle(self):
        R = pd.DataFrame(columns=["A", "B"])
        S = pd.DataFrame(columns=["B", "C"])
        T = pd.DataFrame(columns=["A", "C"])

        dataset = JoinGraph()
        try:
            dataset.add_relation("R", ["B", "E"], relation_address=R)
            raise JoinGraphException("Attribute not in the relation but is allowed!")
        except:
            pass
        dataset.add_relation("R", ["B", "A"], y="B", relation_address=R)
        dataset.add_relation("S", ["B", "C"], relation_address=S)
        dataset.add_relation("T", ["A", "C"], relation_address=T)

        try:
            dataset.check_acyclic()
            raise Exception("Disjoint join graph is allowed!")
        except JoinGraphException:
            pass

        dataset.add_join("R", "S", ["B"], ["B"])
        dataset.add_join("S", "T", ["C"], ["C"])
        dataset.check_acyclic()

        # TODO: check join keys available
        # dataset.add_join("R", "S", ["A"], ["A"])

        dataset.add_join("R", "T", ["A"], ["A"])
        try:
            dataset.check_acyclic()
            raise Exception("Cyclic join graph is allowed!")
        except JoinGraphException:
            pass

    def test_multiplicity_for_many_to_many(self):
        cjt = test_utils.initialize_synthetic_many_to_many()

        self.assertGreater(cjt.get_multiplicity('R','S'), 1)
        self.assertGreater(cjt.get_multiplicity('S','R'), 1)
        self.assertGreater(cjt.get_multiplicity('S','T'), 1)
        self.assertGreater(cjt.get_multiplicity('T','S'), 1)

        self.assertEqual(cjt.get_missing_keys('S','R'), 1)
        self.assertEqual(cjt.get_missing_keys('R','S'), 0)
        self.assertEqual(cjt.get_missing_keys('T','S'), 0)
        self.assertEqual(cjt.get_missing_keys('S','T'), 1)

    def test_multiplicity_for_one_to_many(self):
        cjt = test_utils.initialize_synthetic_one_to_many()

        self.assertGreater(cjt.get_multiplicity('R','T'), 1)
        self.assertEqual(cjt.get_multiplicity('T','R'), 1)
        self.assertGreater(cjt.get_multiplicity('S','T'), 1)
        self.assertEqual(cjt.get_multiplicity('T','S'), 1)

        self.assertEqual(cjt.get_missing_keys('S','T'), 1)
        self.assertEqual(cjt.get_missing_keys('T','S'), 0)
        self.assertEqual(cjt.get_missing_keys('T','S'), 0)
        self.assertEqual(cjt.get_missing_keys('S','T'), 1)


if __name__ == '__main__':
    unittest.main()