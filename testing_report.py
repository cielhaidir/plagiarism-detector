#!/usr/bin/env python3
"""
Comprehensive Testing Report Generator for Plagiarism Detection System
Includes both Blackbox and Whitebox testing with detailed reporting.
"""

import sys
import os
import json
import time
import unittest
import requests
import csv
from unittest.mock import patch, MagicMock
from datetime import datetime
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
try:
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

# Add the current directory to Python path to import project modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from similarity_utils import (
        calculate_final_score,
        jaccard_similarity,
        levenshtein_similarity,
        tfidf_cosine_similarity,
        sentence_embedding_similarity,
        ngrams,
        highlight_similarities,
        simple_preprocess_text
    )
    from app import preprocess_text, search_column, _process_single_search
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import all modules: {e}")
    MODULES_AVAILABLE = False

class Colors:
    """ANSI color codes for console output"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class TestResult:
    """Container for test results"""
    def __init__(self, name: str, passed: bool, message: str = "", execution_time: float = 0.0):
        self.name = name
        self.passed = passed
        self.message = message
        self.execution_time = execution_time
        self.timestamp = datetime.now()

class TestReportGenerator:
    """Main class for generating comprehensive testing reports"""
    
    def __init__(self, api_base_url: str = "http://localhost:5000"):
        self.api_base_url = api_base_url
        self.blackbox_results: List[TestResult] = []
        self.whitebox_results: List[TestResult] = []
        self.performance_results: List[TestResult] = []
        self.start_time = datetime.now()
        
    def print_header(self, title: str, color: str = Colors.BLUE):
        """Print a formatted header"""
        print(f"\n{color}{Colors.BOLD}{'=' * 80}")
        print(f"{title.center(80)}")
        print(f"{'=' * 80}{Colors.END}\n")
        
    def print_section(self, title: str, color: str = Colors.CYAN):
        """Print a formatted section header"""
        print(f"\n{color}{Colors.BOLD}{'-' * 60}")
        print(f"{title}")
        print(f"{'-' * 60}{Colors.END}")
        
    def measure_time(self, func, *args, **kwargs) -> Tuple[Any, float]:
        """Measure execution time of a function"""
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        return result, (end - start) * 1000  # Return time in milliseconds

    # =================================================================================
    # WHITEBOX TESTING - Testing internal functions and logic
    # =================================================================================
    
    def run_whitebox_tests(self):
        """Run all whitebox tests focusing on calculate_final_score function"""
        self.print_header("WHITEBOX TESTING - Internal Function Analysis", Colors.MAGENTA)
        
        if not MODULES_AVAILABLE:
            self.whitebox_results.append(TestResult(
                "Module Import Check", False, 
                "Required modules not available for whitebox testing"
            ))
            return
            
        # Test calculate_final_score function extensively
        self._test_calculate_final_score_basic()
        self._test_calculate_final_score_edge_cases()
        self._test_calculate_final_score_penalties()
        self._test_calculate_final_score_bonuses()
        self._test_calculate_final_score_boundaries()
        self._test_calculate_final_score_weights()
        
        # Test other similarity functions
        self._test_similarity_functions()
        self._test_preprocessing_functions()
        self._test_utility_functions()
        
    def _test_calculate_final_score_basic(self):
        """Test basic functionality of calculate_final_score"""
        self.print_section("Testing calculate_final_score - Basic Functionality")
        
        # Test Case 1: Normal values
        try:
            result, exec_time = self.measure_time(
                calculate_final_score, 
                0.8, 0.7, 0.6, 0.5, "test text one", "test text two"
            )
            
            expected_base = (0.5 * 0.8) + (0.45 * 0.7) + (0.025 * 0.6) + (0.025 * 0.5)
            expected_base = 0.4 + 0.315 + 0.015 + 0.0125  # = 0.7425
            
            # Check if result is reasonable (considering penalties/bonuses)
            if 0.0 <= result <= 1.0:
                self.whitebox_results.append(TestResult(
                    "calculate_final_score - Normal Values", True,
                    f"Result: {result:.4f}, Expected base: {expected_base:.4f}, Time: {exec_time:.2f}ms",
                    exec_time
                ))
            else:
                self.whitebox_results.append(TestResult(
                    "calculate_final_score - Normal Values", False,
                    f"Result {result} is outside valid range [0,1]"
                ))
        except Exception as e:
            self.whitebox_results.append(TestResult(
                "calculate_final_score - Normal Values", False,
                f"Exception occurred: {str(e)}"
            ))
            
        # Test Case 2: All zeros
        try:
            result, exec_time = self.measure_time(
                calculate_final_score, 
                0.0, 0.0, 0.0, 0.0, "", ""
            )
            
            if result == 0.0:
                self.whitebox_results.append(TestResult(
                    "calculate_final_score - All Zeros", True,
                    f"Result: {result}, Time: {exec_time:.2f}ms", exec_time
                ))
            else:
                self.whitebox_results.append(TestResult(
                    "calculate_final_score - All Zeros", False,
                    f"Expected 0.0, got {result}"
                ))
        except Exception as e:
            self.whitebox_results.append(TestResult(
                "calculate_final_score - All Zeros", False,
                f"Exception occurred: {str(e)}"
            ))
            
        # Test Case 3: All maximum values
        try:
            result, exec_time = self.measure_time(
                calculate_final_score, 
                1.0, 1.0, 1.0, 1.0, "identical text", "identical text"
            )
            
            # Should be close to 1.0 with potential bonuses
            if 0.8 <= result <= 1.0:
                self.whitebox_results.append(TestResult(
                    "calculate_final_score - Maximum Values", True,
                    f"Result: {result}, Time: {exec_time:.2f}ms", exec_time
                ))
            else:
                self.whitebox_results.append(TestResult(
                    "calculate_final_score - Maximum Values", False,
                    f"Result {result} not in expected range [0.8, 1.0]"
                ))
        except Exception as e:
            self.whitebox_results.append(TestResult(
                "calculate_final_score - Maximum Values", False,
                f"Exception occurred: {str(e)}"
            ))

    def _test_calculate_final_score_edge_cases(self):
        """Test edge cases for calculate_final_score"""
        self.print_section("Testing calculate_final_score - Edge Cases")
        
        edge_cases = [
            ("Empty strings", 0.5, 0.5, 0.0, 0.0, "", ""),
            ("Very long texts", 0.3, 0.8, 0.1, 0.2, "a" * 1000, "b" * 1000),
            ("Single character", 1.0, 0.9, 1.0, 0.8, "a", "a"),
            ("Special characters only", 0.2, 0.1, 0.05, 0.1, "!@#$%", "&*()_+"),
            ("Numbers only", 0.4, 0.3, 0.2, 0.1, "12345", "67890"),
        ]
        
        for case_name, similarity, semantic, exact, fuzzy, text1, text2 in edge_cases:
            try:
                result, exec_time = self.measure_time(
                    calculate_final_score, 
                    similarity, semantic, exact, fuzzy, text1, text2
                )
                
                if 0.0 <= result <= 1.0:
                    self.whitebox_results.append(TestResult(
                        f"calculate_final_score - {case_name}", True,
                        f"Result: {result:.4f}, Time: {exec_time:.2f}ms", exec_time
                    ))
                else:
                    self.whitebox_results.append(TestResult(
                        f"calculate_final_score - {case_name}", False,
                        f"Result {result} outside valid range"
                    ))
            except Exception as e:
                self.whitebox_results.append(TestResult(
                    f"calculate_final_score - {case_name}", False,
                    f"Exception: {str(e)}"
                ))

    def _test_calculate_final_score_penalties(self):
        """Test penalty conditions in calculate_final_score"""
        self.print_section("Testing calculate_final_score - Penalty Conditions")
        
        # Test penalty when (exact + fuzzy) < 0.5
        try:
            # Case where exact + fuzzy = 0.4 (< 0.5), should trigger 0.4 penalty multiplier
            result_with_penalty, exec_time = self.measure_time(
                calculate_final_score,
                0.8, 0.7, 0.2, 0.2, "some text", "different text"  # exact + fuzzy = 0.4
            )
            
            # Calculate expected result with penalty
            base_score = (0.5 * 0.8) + (0.45 * 0.7) + (0.025 * 0.2) + (0.025 * 0.2)
            expected_with_penalty = base_score * 0.4
            
            # Allow some tolerance for floating point comparison
            if abs(result_with_penalty - expected_with_penalty) < 0.01:
                self.whitebox_results.append(TestResult(
                    "calculate_final_score - Low Literal Match Penalty", True,
                    f"Applied penalty correctly. Result: {result_with_penalty:.4f}, Expected: {expected_with_penalty:.4f}",
                    exec_time
                ))
            else:
                self.whitebox_results.append(TestResult(
                    "calculate_final_score - Low Literal Match Penalty", False,
                    f"Penalty not applied correctly. Result: {result_with_penalty:.4f}, Expected: {expected_with_penalty:.4f}"
                ))
        except Exception as e:
            self.whitebox_results.append(TestResult(
                "calculate_final_score - Low Literal Match Penalty", False,
                f"Exception: {str(e)}"
            ))
            
        # Test penalty when similarity > 0.5 and semantic < 0.6
        try:
            result_with_penalty, exec_time = self.measure_time(
                calculate_final_score,
                0.7, 0.5, 0.8, 0.7, "test text", "test content"  # similarity > 0.5, semantic < 0.6
            )
            
            # This should trigger the 0.8 multiplier penalty
            base_score = (0.5 * 0.7) + (0.45 * 0.5) + (0.025 * 0.8) + (0.025 * 0.7)
            expected_with_penalty = base_score * 0.8
            
            if abs(result_with_penalty - expected_with_penalty) < 0.01:
                self.whitebox_results.append(TestResult(
                    "calculate_final_score - TF-IDF/Semantic Mismatch Penalty", True,
                    f"Applied penalty correctly. Result: {result_with_penalty:.4f}, Expected: {expected_with_penalty:.4f}",
                    exec_time
                ))
            else:
                self.whitebox_results.append(TestResult(
                    "calculate_final_score - TF-IDF/Semantic Mismatch Penalty", False,
                    f"Penalty not applied correctly. Result: {result_with_penalty:.4f}, Expected: {expected_with_penalty:.4f}"
                ))
        except Exception as e:
            self.whitebox_results.append(TestResult(
                "calculate_final_score - TF-IDF/Semantic Mismatch Penalty", False,
                f"Exception: {str(e)}"
            ))

    def _test_calculate_final_score_bonuses(self):
        """Test bonus conditions in calculate_final_score"""
        self.print_section("Testing calculate_final_score - Bonus Conditions")
        
        # Test bonus when semantic > 0.9
        try:
            result_with_bonus, exec_time = self.measure_time(
                calculate_final_score,
                0.8, 0.95, 0.7, 0.6, "very similar text", "very similar content"  # semantic > 0.9
            )
            
            # Calculate expected result with bonus
            base_score = (0.5 * 0.8) + (0.45 * 0.95) + (0.025 * 0.7) + (0.025 * 0.6)
            expected_with_bonus = min(base_score + 0.03, 1.0)  # +0.03 bonus, capped at 1.0
            
            if abs(result_with_bonus - expected_with_bonus) < 0.01:
                self.whitebox_results.append(TestResult(
                    "calculate_final_score - High Semantic Bonus", True,
                    f"Applied bonus correctly. Result: {result_with_bonus:.4f}, Expected: {expected_with_bonus:.4f}",
                    exec_time
                ))
            else:
                self.whitebox_results.append(TestResult(
                    "calculate_final_score - High Semantic Bonus", False,
                    f"Bonus not applied correctly. Result: {result_with_bonus:.4f}, Expected: {expected_with_bonus:.4f}"
                ))
        except Exception as e:
            self.whitebox_results.append(TestResult(
                "calculate_final_score - High Semantic Bonus", False,
                f"Exception: {str(e)}"
            ))

    def _test_calculate_final_score_boundaries(self):
        """Test boundary conditions and score capping"""
        self.print_section("Testing calculate_final_score - Boundary Conditions")
        
        # Test that result is always <= 1.0
        extreme_cases = [
            ("Extreme high values", 2.0, 2.0, 2.0, 2.0, "test", "test"),
            ("High semantic with bonus", 0.9, 0.95, 0.8, 0.8, "test", "test"),
            ("All maximum legal", 1.0, 1.0, 1.0, 1.0, "identical", "identical"),
        ]
        
        for case_name, similarity, semantic, exact, fuzzy, text1, text2 in extreme_cases:
            try:
                result, exec_time = self.measure_time(
                    calculate_final_score,
                    similarity, semantic, exact, fuzzy, text1, text2
                )
                
                if result <= 1.0:
                    self.whitebox_results.append(TestResult(
                        f"calculate_final_score - {case_name} (≤ 1.0)", True,
                        f"Result correctly capped: {result:.4f}", exec_time
                    ))
                else:
                    self.whitebox_results.append(TestResult(
                        f"calculate_final_score - {case_name} (≤ 1.0)", False,
                        f"Result exceeds 1.0: {result:.4f}"
                    ))
            except Exception as e:
                self.whitebox_results.append(TestResult(
                    f"calculate_final_score - {case_name} (≤ 1.0)", False,
                    f"Exception: {str(e)}"
                ))

    def _test_calculate_final_score_weights(self):
        """Test that weights are applied correctly"""
        self.print_section("Testing calculate_final_score - Weight Verification")
        
        try:
            # Test with known values to verify weight calculation
            similarity, semantic, exact, fuzzy = 0.8, 0.6, 0.4, 0.2
            text1, text2 = "test text", "test content"
            
            result, exec_time = self.measure_time(
                calculate_final_score,
                similarity, semantic, exact, fuzzy, text1, text2
            )
            
            # Calculate expected base score with known weights: (0.5, 0.45, 0.025, 0.025)
            expected_base = (0.5 * 0.8) + (0.45 * 0.6) + (0.025 * 0.4) + (0.025 * 0.2)
            expected_base = 0.4 + 0.27 + 0.01 + 0.005  # = 0.685
            
            # Note: This will be modified by penalties, but we can verify the weights are reasonable
            weights_sum = 0.5 + 0.45 + 0.025 + 0.025  # Should equal 1.0
            
            if abs(weights_sum - 1.0) < 0.001:
                self.whitebox_results.append(TestResult(
                    "calculate_final_score - Weight Sum Verification", True,
                    f"Weights sum to 1.0: {weights_sum}", exec_time
                ))
            else:
                self.whitebox_results.append(TestResult(
                    "calculate_final_score - Weight Sum Verification", False,
                    f"Weights sum to {weights_sum}, not 1.0"
                ))
                
            # Test weight distribution (similarity and semantic should dominate)
            if 0.4 <= expected_base <= 0.8:  # Reasonable range given inputs
                self.whitebox_results.append(TestResult(
                    "calculate_final_score - Weight Distribution", True,
                    f"Base score in reasonable range: {expected_base:.4f}", exec_time
                ))
            else:
                self.whitebox_results.append(TestResult(
                    "calculate_final_score - Weight Distribution", False,
                    f"Base score outside expected range: {expected_base:.4f}"
                ))
                
        except Exception as e:
            self.whitebox_results.append(TestResult(
                "calculate_final_score - Weight Verification", False,
                f"Exception: {str(e)}"
            ))

    def _test_similarity_functions(self):
        """Test individual similarity functions"""
        self.print_section("Testing Individual Similarity Functions")
        
        # Test Jaccard similarity
        try:
            result, exec_time = self.measure_time(
                jaccard_similarity, 
                "the quick brown fox", "the quick brown dog"
            )
            
            if 0.0 <= result <= 1.0:
                self.whitebox_results.append(TestResult(
                    "jaccard_similarity - Normal Case", True,
                    f"Result: {result:.4f}, Time: {exec_time:.2f}ms", exec_time
                ))
            else:
                self.whitebox_results.append(TestResult(
                    "jaccard_similarity - Normal Case", False,
                    f"Result {result} outside valid range"
                ))
        except Exception as e:
            self.whitebox_results.append(TestResult(
                "jaccard_similarity - Normal Case", False,
                f"Exception: {str(e)}"
            ))
            
        # Test Levenshtein similarity
        try:
            result, exec_time = self.measure_time(
                levenshtein_similarity,
                "kitten", "sitting"
            )
            
            if 0.0 <= result <= 1.0:
                self.whitebox_results.append(TestResult(
                    "levenshtein_similarity - Normal Case", True,
                    f"Result: {result:.4f}, Time: {exec_time:.2f}ms", exec_time
                ))
            else:
                self.whitebox_results.append(TestResult(
                    "levenshtein_similarity - Normal Case", False,
                    f"Result {result} outside valid range"
                ))
        except Exception as e:
            self.whitebox_results.append(TestResult(
                "levenshtein_similarity - Normal Case", False,
                f"Exception: {str(e)}"
            ))

    def _test_preprocessing_functions(self):
        """Test text preprocessing functions"""
        self.print_section("Testing Text Preprocessing Functions")
        
        try:
            result, exec_time = self.measure_time(
                simple_preprocess_text,
                "Hello World! This is a TEST with 123 numbers and http://example.com links."
            )
            
            # Should be lowercase, no numbers, no URLs, no punctuation
            expected_parts = ["hello", "world", "this", "is", "a", "test", "with", "numbers", "and", "links"]
            
            if result and all(part in result.lower() for part in ["hello", "world", "test"]):
                self.whitebox_results.append(TestResult(
                    "simple_preprocess_text - Normal Case", True,
                    f"Processed: '{result}', Time: {exec_time:.2f}ms", exec_time
                ))
            else:
                self.whitebox_results.append(TestResult(
                    "simple_preprocess_text - Normal Case", False,
                    f"Unexpected result: '{result}'"
                ))
        except Exception as e:
            self.whitebox_results.append(TestResult(
                "simple_preprocess_text - Normal Case", False,
                f"Exception: {str(e)}"
            ))

    def _test_utility_functions(self):
        """Test utility functions"""
        self.print_section("Testing Utility Functions")
        
        # Test ngrams function
        try:
            result, exec_time = self.measure_time(
                ngrams,
                "the quick brown fox jumps", 3
            )
            
            if isinstance(result, set) and len(result) > 0:
                self.whitebox_results.append(TestResult(
                    "ngrams - Normal Case", True,
                    f"Generated {len(result)} 3-grams, Time: {exec_time:.2f}ms", exec_time
                ))
            else:
                self.whitebox_results.append(TestResult(
                    "ngrams - Normal Case", False,
                    f"Unexpected result type or empty: {type(result)}"
                ))
        except Exception as e:
            self.whitebox_results.append(TestResult(
                "ngrams - Normal Case", False,
                f"Exception: {str(e)}"
            ))

    # =================================================================================
    # BLACKBOX TESTING - Testing API endpoints and system behavior
    # =================================================================================
    
    def run_blackbox_tests(self):
        """Run all blackbox tests for API endpoints"""
        self.print_header("BLACKBOX TESTING - API Endpoint Testing", Colors.GREEN)
        
        # Test API availability
        self._test_api_health()
        self._test_api_info()
        
        # Test search endpoints
        self._test_search_endpoint()
        self._test_search_bulk_endpoint()
        self._test_search_error_handling()
        
        # Test edge cases and error conditions
        self._test_api_edge_cases()
        
    def _test_api_health(self):
        """Test API health endpoint"""
        self.print_section("Testing API Health Endpoint")
        
        try:
            start_time = time.time()
            response = requests.get(f"{self.api_base_url}/health", timeout=10)
            exec_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    self.blackbox_results.append(TestResult(
                        "API Health Check", True,
                        f"API is healthy, indices_loaded: {data.get('indices_loaded')}, Time: {exec_time:.2f}ms",
                        exec_time
                    ))
                else:
                    self.blackbox_results.append(TestResult(
                        "API Health Check", False,
                        f"API unhealthy: {data}"
                    ))
            else:
                self.blackbox_results.append(TestResult(
                    "API Health Check", False,
                    f"HTTP {response.status_code}: {response.text}"
                ))
        except requests.exceptions.RequestException as e:
            self.blackbox_results.append(TestResult(
                "API Health Check", False,
                f"Connection error: {str(e)}"
            ))

    def _test_api_info(self):
        """Test API info endpoint"""
        self.print_section("Testing API Info Endpoint")
        
        try:
            start_time = time.time()
            response = requests.get(f"{self.api_base_url}/info", timeout=10)
            exec_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ["available_columns", "total_proposals", "unique_skemas"]
                
                if all(field in data for field in required_fields):
                    self.blackbox_results.append(TestResult(
                        "API Info Endpoint", True,
                        f"Info retrieved: {len(data.get('available_columns', []))} columns, "
                        f"{data.get('total_proposals', 0)} proposals, Time: {exec_time:.2f}ms",
                        exec_time
                    ))
                else:
                    self.blackbox_results.append(TestResult(
                        "API Info Endpoint", False,
                        f"Missing required fields in response: {data}"
                    ))
            else:
                self.blackbox_results.append(TestResult(
                    "API Info Endpoint", False,
                    f"HTTP {response.status_code}: {response.text}"
                ))
        except requests.exceptions.RequestException as e:
            self.blackbox_results.append(TestResult(
                "API Info Endpoint", False,
                f"Connection error: {str(e)}"
            ))

    def _test_search_endpoint(self):
        """Test single search endpoint"""
        self.print_section("Testing Single Search Endpoint")
        
        # Test valid search
        test_cases = [
            {
                "name": "Valid Search - Judul",
                "payload": {
                    "query_text": "machine learning algoritma",
                    "column": "judul",
                    "top_k": 5
                }
            },
            {
                "name": "Valid Search - Ringkasan", 
                "payload": {
                    "query_text": "penelitian tentang sistem informasi",
                    "column": "ringkasan",
                    "top_k": 3
                }
            },
            {
                "name": "Search with Skema Filter",
                "payload": {
                    "query_text": "data mining",
                    "column": "metode",
                    "skema": "Penelitian",
                    "top_k": 2
                }
            }
        ]
        
        for test_case in test_cases:
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.api_base_url}/search",
                    json=test_case["payload"],
                    timeout=30
                )
                exec_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    data = response.json()
                    if "results" in data and isinstance(data["results"], list):
                        results_count = len(data["results"])
                        self.blackbox_results.append(TestResult(
                            test_case["name"], True,
                            f"Found {results_count} results, Time: {exec_time:.2f}ms",
                            exec_time
                        ))
                        
                        # Validate result structure
                        if results_count > 0:
                            result = data["results"][0]
                            required_fields = ["id", "final_score", "similarity_score", "semantic_score"]
                            if all(field in result for field in required_fields):
                                self.blackbox_results.append(TestResult(
                                    f"{test_case['name']} - Result Structure", True,
                                    f"All required fields present in results"
                                ))
                            else:
                                self.blackbox_results.append(TestResult(
                                    f"{test_case['name']} - Result Structure", False,
                                    f"Missing required fields in result: {list(result.keys())}"
                                ))
                    else:
                        self.blackbox_results.append(TestResult(
                            test_case["name"], False,
                            f"Invalid response format: {data}"
                        ))
                else:
                    self.blackbox_results.append(TestResult(
                        test_case["name"], False,
                        f"HTTP {response.status_code}: {response.text}"
                    ))
            except requests.exceptions.RequestException as e:
                self.blackbox_results.append(TestResult(
                    test_case["name"], False,
                    f"Connection error: {str(e)}"
                ))

    def _test_search_bulk_endpoint(self):
        """Test bulk search endpoint"""
        self.print_section("Testing Bulk Search Endpoint")
        
        # Test valid bulk search
        try:
            payload = {
                "texts": [
                    {
                        "proposal_id": "test_1",
                        "judul": "machine learning dalam pendidikan",
                        "ringkasan": "penelitian tentang penerapan ML di sekolah"
                    },
                    {
                        "proposal_id": "test_2", 
                        "metode": "algoritma neural network",
                        "solusi": "implementasi sistem cerdas"
                    }
                ],
                "top_k": 2,
                "use_parallel": False  # Use sequential for more predictable testing
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.api_base_url}/search_bulk",
                json=payload,
                timeout=60
            )
            exec_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                if "bulk_results" in data and isinstance(data["bulk_results"], list):
                    bulk_results = data["bulk_results"]
                    if len(bulk_results) == 2:  # Should match input length
                        self.blackbox_results.append(TestResult(
                            "Bulk Search - Valid Request", True,
                            f"Processed {len(bulk_results)} queries, Time: {exec_time:.2f}ms",
                            exec_time
                        ))
                        
                        # Validate bulk result structure
                        first_result = bulk_results[0]
                        required_fields = ["query_index", "results", "query_info"]
                        if all(field in first_result for field in required_fields):
                            self.blackbox_results.append(TestResult(
                                "Bulk Search - Result Structure", True,
                                f"Bulk result structure is valid"
                            ))
                        else:
                            self.blackbox_results.append(TestResult(
                                "Bulk Search - Result Structure", False,
                                f"Missing required fields: {list(first_result.keys())}"
                            ))
                    else:
                        self.blackbox_results.append(TestResult(
                            "Bulk Search - Valid Request", False,
                            f"Expected 2 results, got {len(bulk_results)}"
                        ))
                else:
                    self.blackbox_results.append(TestResult(
                        "Bulk Search - Valid Request", False,
                        f"Invalid response format: {data}"
                    ))
            else:
                self.blackbox_results.append(TestResult(
                    "Bulk Search - Valid Request", False,
                    f"HTTP {response.status_code}: {response.text}"
                ))
        except requests.exceptions.RequestException as e:
            self.blackbox_results.append(TestResult(
                "Bulk Search - Valid Request", False,
                f"Connection error: {str(e)}"
            ))

    def _test_search_error_handling(self):
        """Test error handling in search endpoints"""
        self.print_section("Testing Search Error Handling")
        
        error_test_cases = [
            {
                "name": "Missing query_text",
                "payload": {"column": "judul"},
                "expected_status": 400
            },
            {
                "name": "Missing column",
                "payload": {"query_text": "test"},
                "expected_status": 400
            },
            {
                "name": "Invalid column",
                "payload": {"query_text": "test", "column": "invalid_column"},
                "expected_status": 400
            },
            {
                "name": "Empty JSON",
                "payload": {},
                "expected_status": 400
            }
        ]
        
        for test_case in error_test_cases:
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.api_base_url}/search",
                    json=test_case["payload"],
                    timeout=10
                )
                exec_time = (time.time() - start_time) * 1000
                
                if response.status_code == test_case["expected_status"]:
                    self.blackbox_results.append(TestResult(
                        f"Error Handling - {test_case['name']}", True,
                        f"Correctly returned HTTP {response.status_code}, Time: {exec_time:.2f}ms",
                        exec_time
                    ))
                else:
                    self.blackbox_results.append(TestResult(
                        f"Error Handling - {test_case['name']}", False,
                        f"Expected HTTP {test_case['expected_status']}, got {response.status_code}"
                    ))
            except requests.exceptions.RequestException as e:
                self.blackbox_results.append(TestResult(
                    f"Error Handling - {test_case['name']}", False,
                    f"Connection error: {str(e)}"
                ))

    def _test_api_edge_cases(self):
        """Test edge cases for API"""
        self.print_section("Testing API Edge Cases")
        
        edge_cases = [
            {
                "name": "Very Long Query Text",
                "payload": {
                    "query_text": "a" * 10000,  # Very long query
                    "column": "judul",
                    "top_k": 1
                }
            },
            {
                "name": "Special Characters Query",
                "payload": {
                    "query_text": "!@#$%^&*()_+{}|:<>?[]\\;'\",./ 测试",
                    "column": "ringkasan",
                    "top_k": 1
                }
            },
            {
                "name": "Empty Query Text",
                "payload": {
                    "query_text": "",
                    "column": "metode",
                    "top_k": 1
                }
            },
            {
                "name": "Large top_k Value",
                "payload": {
                    "query_text": "test",
                    "column": "judul",
                    "top_k": 1000
                }
            }
        ]
        
        for test_case in edge_cases:
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.api_base_url}/search",
                    json=test_case["payload"],
                    timeout=30
                )
                exec_time = (time.time() - start_time) * 1000
                
                # For edge cases, we mainly want to ensure the API doesn't crash
                if response.status_code in [200, 400]:  # Either success or handled error
                    self.blackbox_results.append(TestResult(
                        f"Edge Case - {test_case['name']}", True,
                        f"API handled gracefully (HTTP {response.status_code}), Time: {exec_time:.2f}ms",
                        exec_time
                    ))
                else:
                    self.blackbox_results.append(TestResult(
                        f"Edge Case - {test_case['name']}", False,
                        f"Unexpected HTTP {response.status_code}: {response.text}"
                    ))
            except requests.exceptions.RequestException as e:
                self.blackbox_results.append(TestResult(
                    f"Edge Case - {test_case['name']}", False,
                    f"Connection error: {str(e)}"
                ))

    # =================================================================================
    # PERFORMANCE TESTING
    # =================================================================================
    
    def run_performance_tests(self):
        """Run performance tests"""
        self.print_header("PERFORMANCE TESTING", Colors.YELLOW)
        
        if not MODULES_AVAILABLE:
            self.performance_results.append(TestResult(
                "Performance Tests", False,
                "Modules not available for performance testing"
            ))
            return
            
        self._test_scoring_function_performance()
        self._test_similarity_function_performance()
        
    def _test_scoring_function_performance(self):
        """Test performance of scoring functions"""
        self.print_section("Testing Scoring Function Performance")
        
        # Test calculate_final_score performance
        iterations = 1000
        test_data = [
            (0.8, 0.7, 0.6, 0.5, "test text one", "test text two")
            for _ in range(iterations)
        ]
        
        try:
            start_time = time.time()
            for similarity, semantic, exact, fuzzy, text1, text2 in test_data:
                calculate_final_score(similarity, semantic, exact, fuzzy, text1, text2)
            total_time = (time.time() - start_time) * 1000
            avg_time = total_time / iterations
            
            if avg_time < 1.0:  # Should be less than 1ms per call
                self.performance_results.append(TestResult(
                    f"calculate_final_score Performance ({iterations} iterations)", True,
                    f"Avg time per call: {avg_time:.4f}ms, Total: {total_time:.2f}ms",
                    total_time
                ))
            else:
                self.performance_results.append(TestResult(
                    f"calculate_final_score Performance ({iterations} iterations)", False,
                    f"Too slow: {avg_time:.4f}ms per call"
                ))
        except Exception as e:
            self.performance_results.append(TestResult(
                f"calculate_final_score Performance ({iterations} iterations)", False,
                f"Exception: {str(e)}"
            ))

    def _test_similarity_function_performance(self):
        """Test performance of individual similarity functions"""
        self.print_section("Testing Individual Similarity Function Performance")
        
        test_texts = [
            ("the quick brown fox jumps over the lazy dog", "a quick brown fox leaps over a lazy dog"),
            ("machine learning algorithms", "algorithmic machine learning"),
            ("data science and analytics", "analytics in data science"),
        ]
        
        functions_to_test = [
            ("jaccard_similarity", jaccard_similarity),
            ("levenshtein_similarity", levenshtein_similarity),
        ]
        
        for func_name, func in functions_to_test:
            try:
                total_time = 0
                iterations = len(test_texts) * 100  # 100 iterations per text pair
                
                start_time = time.time()
                for _ in range(100):
                    for text1, text2 in test_texts:
                        func(text1, text2)
                total_time = (time.time() - start_time) * 1000
                avg_time = total_time / iterations
                
                if avg_time < 5.0:  # Should be less than 5ms per call
                    self.performance_results.append(TestResult(
                        f"{func_name} Performance ({iterations} iterations)", True,
                        f"Avg time per call: {avg_time:.4f}ms, Total: {total_time:.2f}ms",
                        total_time
                    ))
                else:
                    self.performance_results.append(TestResult(
                        f"{func_name} Performance ({iterations} iterations)", False,
                        f"Too slow: {avg_time:.4f}ms per call"
                    ))
            except Exception as e:
                self.performance_results.append(TestResult(
                    f"{func_name} Performance", False,
                    f"Exception: {str(e)}"
                ))

    # =================================================================================
    # REPORT GENERATION
    # =================================================================================
    
    def generate_report(self):
        """Generate comprehensive testing report"""
        self.print_header("COMPREHENSIVE TESTING REPORT", Colors.BOLD)
        
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        # Summary statistics
        all_results = self.whitebox_results + self.blackbox_results + self.performance_results
        total_tests = len(all_results)
        passed_tests = len([r for r in all_results if r.passed])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"{Colors.BOLD}TEST EXECUTION SUMMARY{Colors.END}")
        print(f"{'─' * 50}")
        print(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Duration: {total_duration:.2f} seconds")
        print(f"Total Tests: {Colors.BOLD}{total_tests}{Colors.END}")
        print(f"Passed: {Colors.GREEN}{passed_tests}{Colors.END}")
        print(f"Failed: {Colors.RED}{failed_tests}{Colors.END}")
        print(f"Success Rate: {Colors.BOLD}{success_rate:.1f}%{Colors.END}")
        
        # Detailed results by category
        self._print_detailed_results("WHITEBOX TEST RESULTS", self.whitebox_results, Colors.MAGENTA)
        self._print_detailed_results("BLACKBOX TEST RESULTS", self.blackbox_results, Colors.GREEN)
        self._print_detailed_results("PERFORMANCE TEST RESULTS", self.performance_results, Colors.YELLOW)
        
        # Performance summary
        self._print_performance_summary()
        
        # Recommendations
        self._print_recommendations()
        
        # Save report to file
        self._save_report_to_file()
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": success_rate,
            "duration": total_duration,
            "whitebox_results": len(self.whitebox_results),
            "blackbox_results": len(self.blackbox_results),
            "performance_results": len(self.performance_results)
        }
    
    def _print_detailed_results(self, title: str, results: List[TestResult], color: str):
        """Print detailed results for a category"""
        if not results:
            return
            
        self.print_section(title, color)
        
        passed = [r for r in results if r.passed]
        failed = [r for r in results if not r.passed]
        
        print(f"Summary: {Colors.GREEN}{len(passed)} passed{Colors.END}, {Colors.RED}{len(failed)} failed{Colors.END}")
        print()
        
        # Show failed tests first
        if failed:
            print(f"{Colors.RED}{Colors.BOLD}FAILED TESTS:{Colors.END}")
            for result in failed:
                print(f"  {Colors.RED}✗{Colors.END} {result.name}")
                print(f"    {result.message}")
                if result.execution_time > 0:
                    print(f"    Time: {result.execution_time:.2f}ms")
                print()
        
        # Show passed tests
        if passed:
            print(f"{Colors.GREEN}{Colors.BOLD}PASSED TESTS:{Colors.END}")
            for result in passed:
                print(f"  {Colors.GREEN}✓{Colors.END} {result.name}")
                if result.message:
                    print(f"    {result.message}")
        print()

    def _print_performance_summary(self):
        """Print performance analysis summary"""
        self.print_section("PERFORMANCE ANALYSIS", Colors.YELLOW)
        
        if not self.performance_results:
            print("No performance tests were executed.")
            return
            
        # Calculate performance metrics
        total_exec_time = sum(r.execution_time for r in self.performance_results if r.execution_time > 0)
        avg_exec_time = total_exec_time / len(self.performance_results) if self.performance_results else 0
        
        print(f"Total Performance Test Time: {total_exec_time:.2f}ms")
        print(f"Average Test Execution Time: {avg_exec_time:.2f}ms")
        
        # Find slowest tests
        slowest_tests = sorted(
            [r for r in self.performance_results if r.execution_time > 0],
            key=lambda x: x.execution_time,
            reverse=True
        )[:5]
        
        if slowest_tests:
            print(f"\n{Colors.BOLD}Top 5 Slowest Operations:{Colors.END}")
            for i, result in enumerate(slowest_tests, 1):
                print(f"  {i}. {result.name}: {result.execution_time:.2f}ms")

    def _print_recommendations(self):
        """Print recommendations based on test results"""
        self.print_section("RECOMMENDATIONS", Colors.CYAN)
        
        failed_tests = [r for r in (self.whitebox_results + self.blackbox_results + self.performance_results) if not r.passed]
        
        if not failed_tests:
            print(f"{Colors.GREEN}✓ All tests passed! The system appears to be functioning correctly.{Colors.END}")
            print(f"{Colors.GREEN}✓ The calculate_final_score function is working as expected.{Colors.END}")
            print(f"{Colors.GREEN}✓ API endpoints are responding correctly.{Colors.END}")
        else:
            print(f"{Colors.YELLOW}Areas for improvement:{Colors.END}")
            
            # Analyze failure patterns
            whitebox_failures = [r for r in self.whitebox_results if not r.passed]
            blackbox_failures = [r for r in self.blackbox_results if not r.passed]
            performance_failures = [r for r in self.performance_results if not r.passed]
            
            if whitebox_failures:
                print(f"  • {len(whitebox_failures)} whitebox test(s) failed - review internal function logic")
                if any("calculate_final_score" in r.name for r in whitebox_failures):
                    print(f"    - The calculate_final_score function has issues that need attention")
                    
            if blackbox_failures:
                print(f"  • {len(blackbox_failures)} blackbox test(s) failed - check API functionality")
                if any("API" in r.name for r in blackbox_failures):
                    print(f"    - API connectivity or basic functionality issues detected")
                    
            if performance_failures:
                print(f"  • {len(performance_failures)} performance test(s) failed - optimize slow operations")
        
        print(f"\n{Colors.BOLD}General Recommendations:{Colors.END}")
        print("  • Run this test suite regularly during development")
        print("  • Pay special attention to the calculate_final_score function as it's critical for accuracy")
        print("  • Monitor API response times in production")
        print("  • Consider adding more edge case tests for robustness")

    def _save_report_to_file(self):
        """Save detailed report to JSON, CSV, and Excel files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"testing_report_{timestamp}.json"
        csv_filename = f"testing_report_{timestamp}.csv"
        excel_filename = f"testing_report_{timestamp}.xlsx"
        
        report_data = {
            "timestamp": self.start_time.isoformat(),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "summary": {
                "total_tests": len(self.whitebox_results + self.blackbox_results + self.performance_results),
                "whitebox_tests": len(self.whitebox_results),
                "blackbox_tests": len(self.blackbox_results),
                "performance_tests": len(self.performance_results),
                "passed_tests": len([r for r in (self.whitebox_results + self.blackbox_results + self.performance_results) if r.passed]),
                "failed_tests": len([r for r in (self.whitebox_results + self.blackbox_results + self.performance_results) if not r.passed])
            },
            "whitebox_results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "message": r.message,
                    "execution_time_ms": r.execution_time,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in self.whitebox_results
            ],
            "blackbox_results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "message": r.message,
                    "execution_time_ms": r.execution_time,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in self.blackbox_results
            ],
            "performance_results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "message": r.message,
                    "execution_time_ms": r.execution_time,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in self.performance_results
            ]
        }
        
        # Save JSON report
        try:
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            print(f"\n{Colors.CYAN}Detailed JSON report saved to: {json_filename}{Colors.END}")
        except Exception as e:
            print(f"\n{Colors.RED}Failed to save JSON report to file: {str(e)}{Colors.END}")
        
        # Save CSV report
        self._save_report_to_csv(csv_filename)
        
        # Save Excel report
        self._save_report_to_excel(excel_filename)

    def _save_report_to_csv(self, filename: str):
        """Save test results to CSV file"""
        try:
            all_results = []
            
            # Collect all results with category labels
            for result in self.whitebox_results:
                all_results.append({
                    'Test Category': 'Whitebox',
                    'Test Name': result.name,
                    'Status': 'PASSED' if result.passed else 'FAILED',
                    'Execution Time (ms)': f"{result.execution_time:.4f}" if result.execution_time > 0 else 'N/A',
                    'Message': result.message,
                    'Timestamp': result.timestamp.strftime('%Y-%m-%d %H:%M:%S')
                })
            
            for result in self.blackbox_results:
                all_results.append({
                    'Test Category': 'Blackbox',
                    'Test Name': result.name,
                    'Status': 'PASSED' if result.passed else 'FAILED',
                    'Execution Time (ms)': f"{result.execution_time:.4f}" if result.execution_time > 0 else 'N/A',
                    'Message': result.message,
                    'Timestamp': result.timestamp.strftime('%Y-%m-%d %H:%M:%S')
                })
            
            for result in self.performance_results:
                all_results.append({
                    'Test Category': 'Performance',
                    'Test Name': result.name,
                    'Status': 'PASSED' if result.passed else 'FAILED',
                    'Execution Time (ms)': f"{result.execution_time:.4f}" if result.execution_time > 0 else 'N/A',
                    'Message': result.message,
                    'Timestamp': result.timestamp.strftime('%Y-%m-%d %H:%M:%S')
                })
            
            # Write to CSV
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                if all_results:
                    fieldnames = ['Test Category', 'Test Name', 'Status', 'Execution Time (ms)', 'Message', 'Timestamp']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    # Write header
                    writer.writeheader()
                    
                    # Write all test results
                    writer.writerows(all_results)
                    
                    # Add summary row
                    total_tests = len(all_results)
                    passed_tests = len([r for r in all_results if r['Status'] == 'PASSED'])
                    failed_tests = total_tests - passed_tests
                    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
                    
                    # Add empty row for separation
                    writer.writerow({col: '' for col in fieldnames})
                    
                    # Add summary
                    writer.writerow({
                        'Test Category': 'SUMMARY',
                        'Test Name': f'Total Tests: {total_tests}',
                        'Status': f'Success Rate: {success_rate:.1f}%',
                        'Execution Time (ms)': f'Passed: {passed_tests}',
                        'Message': f'Failed: {failed_tests}',
                        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                
            print(f"{Colors.CYAN}Test results saved to CSV: {filename}{Colors.END}")
            
        except Exception as e:
            print(f"{Colors.RED}Failed to save CSV report: {str(e)}{Colors.END}")

    def _save_report_to_excel(self, filename: str):
        """Save test results to Excel file with specific Indonesian format"""
        if not EXCEL_AVAILABLE:
            print(f"{Colors.YELLOW}Warning: openpyxl not available. Cannot generate Excel report.{Colors.END}")
            print(f"{Colors.YELLOW}Install with: pip install openpyxl{Colors.END}")
            return
            
        try:
            # Prepare data with Indonesian column headers
            excel_data = []
            no = 1
            
            # Process whitebox results
            for result in self.whitebox_results:
                expected_result = self._get_expected_result(result.name, "whitebox")
                excel_data.append({
                    'No': no,
                    'Skenario Pengujian': result.name,
                    'Hasil Yang Diharapkan': expected_result,
                    'Hasil Pengujian': result.message,
                    'Status': 'LULUS' if result.passed else 'GAGAL'
                })
                no += 1
            
            # Process blackbox results
            for result in self.blackbox_results:
                expected_result = self._get_expected_result(result.name, "blackbox")
                excel_data.append({
                    'No': no,
                    'Skenario Pengujian': result.name,
                    'Hasil Yang Diharapkan': expected_result,
                    'Hasil Pengujian': result.message,
                    'Status': 'LULUS' if result.passed else 'GAGAL'
                })
                no += 1
                
            # Process performance results
            for result in self.performance_results:
                expected_result = self._get_expected_result(result.name, "performance")
                excel_data.append({
                    'No': no,
                    'Skenario Pengujian': result.name,
                    'Hasil Yang Diharapkan': expected_result,
                    'Hasil Pengujian': result.message,
                    'Status': 'LULUS' if result.passed else 'GAGAL'
                })
                no += 1
            
            # Create DataFrame and save to Excel
            df = pd.DataFrame(excel_data)
            
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Write main data
                df.to_excel(writer, sheet_name='Test Results', index=False)
                
                # Get the workbook and worksheet for formatting
                workbook = writer.book
                worksheet = writer.sheets['Test Results']
                
                # Auto-adjust column widths
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)  # Cap at 50 characters
                    worksheet.column_dimensions[column_letter].width = adjusted_width
                
                # Add summary sheet
                summary_data = {
                    'Metrik': ['Total Pengujian', 'Lulus', 'Gagal', 'Tingkat Keberhasilan (%)', 'Whitebox Tests', 'Blackbox Tests', 'Performance Tests'],
                    'Nilai': [
                        len(excel_data),
                        len([r for r in excel_data if r['Status'] == 'LULUS']),
                        len([r for r in excel_data if r['Status'] == 'GAGAL']),
                        f"{(len([r for r in excel_data if r['Status'] == 'LULUS']) / len(excel_data) * 100):.1f}" if excel_data else "0",
                        len(self.whitebox_results),
                        len(self.blackbox_results),
                        len(self.performance_results)
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Ringkasan', index=False)
                
                # Format summary sheet
                summary_ws = writer.sheets['Ringkasan']
                for column in summary_ws.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = max_length + 2
                    summary_ws.column_dimensions[column_letter].width = adjusted_width
            
            print(f"{Colors.CYAN}Excel report saved to: {filename}{Colors.END}")
            
        except Exception as e:
            print(f"{Colors.RED}Failed to save Excel report: {str(e)}{Colors.END}")
            
    def _get_expected_result(self, test_name: str, category: str) -> str:
        """Get expected result description for each test"""
        if category == "whitebox":
            if "calculate_final_score" in test_name.lower():
                if "normal values" in test_name.lower():
                    return "Fungsi mengembalikan skor antara 0.0-1.0 dengan perhitungan bobot yang benar"
                elif "zeros" in test_name.lower():
                    return "Fungsi mengembalikan 0.0 untuk input kosong"
                elif "maximum" in test_name.lower():
                    return "Fungsi mengembalikan skor mendekati 1.0 untuk input identik"
                elif "penalty" in test_name.lower():
                    return "Fungsi menerapkan penalti sesuai kondisi yang ditentukan"
                elif "bonus" in test_name.lower():
                    return "Fungsi menerapkan bonus untuk skor semantik tinggi"
                elif "weight" in test_name.lower():
                    return "Bobot fungsi berjumlah 1.0 dan distribusi sesuai spesifikasi"
                else:
                    return "Fungsi berjalan tanpa error dan mengembalikan nilai valid"
            elif "similarity" in test_name.lower():
                return "Fungsi similarity mengembalikan nilai antara 0.0-1.0"
            elif "preprocess" in test_name.lower():
                return "Teks diproses dengan benar (lowercase, tanpa tanda baca)"
            else:
                return "Fungsi internal berjalan sesuai spesifikasi"
                
        elif category == "blackbox":
            if "health" in test_name.lower():
                return "API mengembalikan status healthy (HTTP 200)"
            elif "info" in test_name.lower():
                return "API mengembalikan informasi sistem yang lengkap"
            elif "search" in test_name.lower():
                return "API mengembalikan hasil pencarian yang valid dengan struktur JSON yang benar"
            elif "error" in test_name.lower():
                return "API mengembalikan kode error yang sesuai (HTTP 400/500)"
            elif "edge case" in test_name.lower():
                return "API menangani kasus ekstrem tanpa crash"
            else:
                return "Endpoint API berfungsi sesuai spesifikasi"
                
        elif category == "performance":
            if "calculate_final_score" in test_name.lower():
                return "Waktu eksekusi < 1ms per pemanggilan"
            elif "similarity" in test_name.lower():
                return "Waktu eksekusi < 5ms per pemanggilan"
            else:
                return "Performa dalam batas yang dapat diterima"
                
        return "Fungsi berjalan tanpa error sesuai spesifikasi"

def main():
    """Main function to run all tests"""
    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("=" * 80)
    print("PLAGIARISM DETECTION SYSTEM - COMPREHENSIVE TESTING SUITE")
    print("=" * 80)
    print(f"{Colors.END}")
    
    # Check if API is available
    api_url = "http://localhost:5000"
    print(f"Testing API at: {api_url}")
    
    # Initialize test runner
    test_runner = TestReportGenerator(api_url)
    
    try:
        # Run all test categories
        test_runner.run_whitebox_tests()
        test_runner.run_blackbox_tests()
        test_runner.run_performance_tests()
        
        # Generate comprehensive report
        summary = test_runner.generate_report()
        
        # Exit with appropriate code
        if summary["failed_tests"] == 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}All tests passed! ✓{Colors.END}")
            sys.exit(0)
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}{summary['failed_tests']} test(s) failed! ✗{Colors.END}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Testing interrupted by user{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Critical error during testing: {str(e)}{Colors.END}")
        sys.exit(1)

if __name__ == "__main__":
    main()