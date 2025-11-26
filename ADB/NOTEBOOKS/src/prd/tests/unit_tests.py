# Databricks notebook source
"""
Unit Tests for MLOps Pipeline
This notebook runs basic unit tests to validate the deployment
"""

# COMMAND ----------

import sys
import traceback

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test Suite

# COMMAND ----------

class TestRunner:
    def __init__(self):
        self.passed_tests = 0
        self.failed_tests = 0
        self.test_results = []
    
    def assert_equal(self, actual, expected, test_name):
        """Custom assert equal with test tracking"""
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
        """Custom assert true with test tracking"""
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
        """Print test summary"""
        print("\n" + "="*50)
        print("TEST SUMMARY")
        print("="*50)
        print(f"Total Tests: {self.passed_tests + self.failed_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        print("="*50)
        
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

def add_numbers(a, b):
    """Sample function to test - add two numbers"""
    return a + b

def multiply_numbers(a, b):
    """Sample function to test - multiply two numbers"""
    return a * b

def is_even(number):
    """Sample function to test - check if number is even"""
    return number % 2 == 0

def capitalize_string(text):
    """Sample function to test - capitalize string"""
    return text.upper()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Tests

# COMMAND ----------

# Initialize test runner
runner = TestRunner()

print("Starting Unit Tests...")
print("="*50)

# Test 1: Addition
runner.assert_equal(add_numbers(2, 3), 5, "Test Addition: 2 + 3 = 5")

# Test 2: Multiplication
runner.assert_equal(multiply_numbers(4, 5), 20, "Test Multiplication: 4 * 5 = 20")

# Test 3: Even number check
runner.assert_true(is_even(4), "Test Even Number: 4 is even")

# Test 4: Odd number check
runner.assert_true(not is_even(5), "Test Odd Number: 5 is not even")

# Test 5: String capitalization
runner.assert_equal(capitalize_string("hello"), "HELLO", "Test String Capitalization")

# Test 6: Zero addition
runner.assert_equal(add_numbers(0, 0), 0, "Test Zero Addition: 0 + 0 = 0")

# Test 7: Negative numbers
runner.assert_equal(add_numbers(-5, 3), -2, "Test Negative Addition: -5 + 3 = -2")

# Test 8: String operations
runner.assert_equal(len("test"), 4, "Test String Length")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Print Summary and Exit

# COMMAND ----------

# Print summary
all_passed = runner.print_summary()

# Exit with appropriate code
if not all_passed:
    print("\n❌ Unit tests FAILED! Deployment should not proceed.")
    raise Exception("Unit tests failed. Stopping execution.")
else:
    print("\n✅ All unit tests PASSED! Deployment can proceed.")

# COMMAND ----------

dbutils.notebook.exit("success" if all_passed else "failed")

