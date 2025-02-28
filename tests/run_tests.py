"""
Test runner for MemoryWeave tests.
"""

import unittest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def run_tests():
    """Run all tests."""
    # Discover and run tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(start_dir=os.path.dirname(__file__), pattern="test_*.py")
    
    # Run tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Return exit code based on test results
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
