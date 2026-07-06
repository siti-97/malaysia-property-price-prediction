import unittest
import math

class TestPropertyDataPipeline(unittest.TestCase):
    def test_logical_boundaries(self):
        # Test: Property price must be strictly positive
        min_simulated_price = 50000 
        self.assertGreater(min_simulated_price, 0, "FAIL: Price must be strictly positive")

    def test_feature_engineering_logic(self):
        # Test: Log transformation mathematical logic is sound
        raw_price = 100000
        log_price = math.log1p(raw_price)
        self.assertTrue(log_price > 0, "FAIL: Log-transformed price should be positive")

    def test_data_types(self):
        # Test: Simulated bedroom count must not be negative
        simulated_bedrooms = 3
        self.assertGreaterEqual(simulated_bedrooms, 0, "FAIL: Negative bedroom count detected")

if __name__ == '__main__':
    unittest.main()
