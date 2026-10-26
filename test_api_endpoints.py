"""
Automated API endpoint tests for Industrial Sensor AI Demo
Tests all critical endpoints and methods
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def test_gap_benchmark_info():
    """Test gap benchmark info endpoint"""
    print("Testing gap benchmark info...")
    response = requests.get(f"{BASE_URL}/api/v1/sensor/gap-benchmark/info")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert "available_gap_lengths" in data, "Missing available_gap_lengths"
    # The endpoint returns a list of objects with gap_length details
    gap_lengths = [item["gap_length"] for item in data["available_gap_lengths"]]
    assert gap_lengths == [10, 20, 50], f"Expected [10, 20, 50], got {gap_lengths}"
    print("✓ Gap benchmark info test passed")

def test_gap_benchmark_windows():
    """Test gap benchmark windows endpoint"""
    print("Testing gap benchmark windows...")
    response = requests.get(f"{BASE_URL}/api/v1/sensor/gap-benchmark/10/windows")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert "windows" in data, "Missing windows"
    assert len(data["windows"]) > 0, "No windows returned"
    print(f"✓ Gap benchmark windows test passed ({len(data['windows'])} windows)")

def test_linear_interpolation():
    """Test linear interpolation reconstruction"""
    print("Testing linear interpolation...")
    body = {
        "gap_length": 10,
        "filename": "test_03_M01_DC_test.npz",
        "window_index": 0,
        "method": "linear_interpolation"
    }
    response = requests.post(f"{BASE_URL}/api/v1/sensor/gap-benchmark/reconstruct-window", json=body)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["method"] == "linear_interpolation", f"Expected linear_interpolation, got {data['method']}"
    assert "predictions" in data, "Missing predictions"
    assert "ground_truth" in data, "Missing ground_truth"
    assert "metrics" in data, "Missing metrics"
    print("✓ Linear interpolation test passed")

def test_residual_lstm_gap_10():
    """Test residual LSTM for gap 10"""
    print("Testing residual LSTM (gap 10)...")
    body = {
        "gap_length": 10,
        "filename": "test_03_M01_DC_test.npz",
        "window_index": 0,
        "method": "residual_lstm"
    }
    response = requests.post(f"{BASE_URL}/api/v1/sensor/gap-benchmark/reconstruct-window", json=body)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["method"] == "residual_lstm", f"Expected residual_lstm, got {data['method']}"
    assert "predictions" in data, "Missing predictions"
    print("✓ Residual LSTM (gap 10) test passed")

def test_residual_lstm_gap_20():
    """Test residual LSTM for gap 20"""
    print("Testing residual LSTM (gap 20)...")
    body = {
        "gap_length": 20,
        "filename": "test_03_M01_DC_test.npz",
        "window_index": 0,
        "method": "residual_lstm"
    }
    response = requests.post(f"{BASE_URL}/api/v1/sensor/gap-benchmark/reconstruct-window", json=body)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["gap_length"] == 20, f"Expected gap_length 20, got {data['gap_length']}"
    print("✓ Residual LSTM (gap 20) test passed")

def test_residual_lstm_gap_50():
    """Test residual LSTM for gap 50"""
    print("Testing residual LSTM (gap 50)...")
    body = {
        "gap_length": 50,
        "filename": "test_03_M01_DC_test.npz",
        "window_index": 0,
        "method": "residual_lstm"
    }
    response = requests.post(f"{BASE_URL}/api/v1/sensor/gap-benchmark/reconstruct-window", json=body)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["gap_length"] == 50, f"Expected gap_length 50, got {data['gap_length']}"
    print("✓ Residual LSTM (gap 50) test passed")

def test_hybrid_reconstruction():
    """Test hybrid reconstruction"""
    print("Testing hybrid reconstruction...")
    body = {
        "gap_length": 10,
        "filename": "test_03_M01_DC_test.npz",
        "window_index": 0,
        "method": "hybrid"
    }
    response = requests.post(f"{BASE_URL}/api/v1/sensor/gap-benchmark/reconstruct-window", json=body)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["method"] == "hybrid", f"Expected hybrid, got {data['method']}"
    assert "predictions" in data, "Missing predictions"
    print("✓ Hybrid reconstruction test passed")

def test_evaluation_summary():
    """Test evaluation summary endpoint"""
    print("Testing evaluation summary...")
    response = requests.get(f"{BASE_URL}/api/v1/sensor/gap-benchmark/10/evaluation-summary")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert "gap_length" in data, "Missing gap_length"
    assert "overall_normalized" in data, "Missing overall_normalized"
    assert "total_sequences" in data, "Missing total_sequences"
    print("✓ Evaluation summary test passed")

def test_epoch_comparison():
    """Test epoch comparison endpoint"""
    print("Testing epoch comparison...")
    response = requests.get(f"{BASE_URL}/api/v1/sensor/model/epoch-comparison")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert "best_epoch" in data, "Missing best_epoch"
    assert "epoch_data" in data, "Missing epoch_data"
    assert len(data["epoch_data"]) == 4, f"Expected 4 epoch data points, got {len(data['epoch_data'])}"
    print("✓ Epoch comparison test passed")

def test_invalid_method():
    """Test invalid method validation"""
    print("Testing invalid method validation...")
    body = {
        "gap_length": 10,
        "filename": "test_03_M01_DC_test.npz",
        "window_index": 0,
        "method": "invalid_method"
    }
    response = requests.post(f"{BASE_URL}/api/v1/sensor/gap-benchmark/reconstruct-window", json=body)
    assert response.status_code == 400, f"Expected 400 for invalid method, got {response.status_code}"
    print("✓ Invalid method validation test passed")

def test_invalid_gap_length():
    """Test invalid gap length validation"""
    print("Testing invalid gap length validation...")
    body = {
        "gap_length": 100,
        "filename": "test_03_M01_DC_test.npz",
        "window_index": 0,
        "method": "residual_lstm"
    }
    response = requests.post(f"{BASE_URL}/api/v1/sensor/gap-benchmark/reconstruct-window", json=body)
    assert response.status_code == 404, f"Expected 404 for invalid gap length, got {response.status_code}"
    print("✓ Invalid gap length validation test passed")

def main():
    print("=" * 70)
    print("AUTOMATED API ENDPOINT TESTS")
    print("=" * 70)
    print()
    
    tests = [
        test_gap_benchmark_info,
        test_gap_benchmark_windows,
        test_linear_interpolation,
        test_residual_lstm_gap_10,
        test_residual_lstm_gap_20,
        test_residual_lstm_gap_50,
        test_hybrid_reconstruction,
        test_evaluation_summary,
        test_epoch_comparison,
        test_invalid_method,
        test_invalid_gap_length,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            failed += 1
        print()
    
    print("=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
