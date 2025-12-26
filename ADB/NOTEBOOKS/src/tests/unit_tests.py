# Databricks notebook source
"""
Unit Tests for MLOps Pipeline

This notebook contains unit tests to validate the deployment pipeline.
It includes:
- Custom test runner with result tracking
- Sample functions for testing
- Basic validation tests
- Test summary reporting

The tests must pass before deployment can proceed.
"""

# COMMAND ----------

# Standard library imports for testing framework
import sys
import traceback

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Suite

# COMMAND ----------

class TestRunner:
    """
    Custom test runner that tracks test results and provides summary reporting.
    Provides simple assertion methods similar to standard testing frameworks.
    """

    def __init__(self):
        """Initialize test runner with empty counters and results list"""
        self.passed_tests = 0
        self.failed_tests = 0
        self.test_results = []

    def assert_equal(self, actual, expected, test_name):
        """Assert that actual equals expected value"""
        try:
            assert actual == expected, f"Expected {expected}, but got {actual}"
            self.passed_tests += 1
            self.test_results.append(f"✓ PASSED: {test_name}")
            print(f"✓ PASSED: {test_name}")
            return True
        except AssertionError as e:
            self.failed_tests += 1
            self.test_results.append(f"✗ FAILED: {test_name} - {str(e)}")
            print(f"✗ FAILED: {test_name} - {str(e)}")
            return False

    def assert_true(self, condition, test_name):
        """Assert that condition evaluates to True"""
        try:
            assert condition, f"Condition evaluated to False"
            self.passed_tests += 1
            self.test_results.append(f"✓ PASSED: {test_name}")
            print(f"✓ PASSED: {test_name}")
            return True
        except AssertionError as e:
            self.failed_tests += 1
            self.test_results.append(f"✗ FAILED: {test_name} - {str(e)}")
            print(f"✗ FAILED: {test_name} - {str(e)}")
            return False

    def print_summary(self):
        """Print comprehensive test summary with pass/fail counts"""
        print("\n" + "="*50)
        print("TEST SUMMARY")
        print("="*50)
        print(f"Total Tests: {self.passed_tests + self.failed_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        print("="*50)

        # Show detailed failure information if any tests failed
        if self.failed_tests > 0:
            print("\nFailed Tests:")
            for result in self.test_results:
                if "FAILED" in result:
                    print(result)

        return self.failed_tests == 0

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sample Unit Tests

# COMMAND ----------

# Sample functions to demonstrate testing capabilities
# These would be replaced with actual project functions in a real deployment

def add_numbers(a, b):
    """Add two numbers together"""
    return a + b

def multiply_numbers(a, b):
    """Multiply two numbers together"""
    return a * b

def is_even(number):
    """Check if a number is even"""
    return number % 2 == 0

def capitalize_string(text):
    """Convert string to uppercase"""
    return text.upper()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Tests

# COMMAND ----------

# Initialize test runner
runner = TestRunner()

print("Starting Unit Tests...")
print("="*50)

# Basic arithmetic tests
runner.assert_equal(add_numbers(2, 3), 5, "Test Addition: 2 + 3 = 5")
# runner.assert_equal(multiply_numbers(4, 5), 20, "Test Multiplication: 4 * 5 = 20")
# runner.assert_equal(add_numbers(0, 0), 0, "Test Zero Addition: 0 + 0 = 0")
# runner.assert_equal(add_numbers(-5, 3), -2, "Test Negative Addition: -5 + 3 = -2")

# # Boolean logic tests
# runner.assert_true(is_even(4), "Test Even Number: 4 is even")
# runner.assert_true(not is_even(5), "Test Odd Number: 5 is not even")

# # String manipulation tests
# runner.assert_equal(capitalize_string("hello"), "HELLO", "Test String Capitalization")
# runner.assert_equal(len("test"), 4, "Test String Length")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Print Summary and Exit

# COMMAND ----------

# Generate and display test summary
all_passed = runner.print_summary()

# Determine final result and exit appropriately
if not all_passed:
    print("\n❌ Unit tests FAILED! Deployment should not proceed.")
    raise Exception("Unit tests failed. Stopping execution.")
else:
    print("\n✅ All unit tests PASSED! Deployment can proceed.")

# COMMAND ----------

# Exit notebook with success/failure status for pipeline integration
dbutils.notebook.exit("success" if all_passed else "failed")
