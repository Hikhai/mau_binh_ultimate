"""
Chạy tất cả tests cho core modules
"""
import sys
import traceback


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("🧪 RUNNING ALL CORE TESTS")
    print("=" * 60)
    
    tests = [
        ("card.py", "card"),
        ("hand_types.py", "hand_types"),
        ("evaluator.py", "evaluator"),
    ]
    
    passed = 0
    failed = 0
    
    for filename, module_name in tests:
        print(f"\n{'='*60}")
        print(f"Testing {filename}...")
        print(f"{'='*60}")
        
        try:
            # Import và chạy
            module = __import__(module_name)
            
            # Chạy các test functions
            if hasattr(module, 'test_card'):
                module.test_card()
            if hasattr(module, 'test_deck'):
                module.test_deck()
            if hasattr(module, 'test_hand_rank'):
                module.test_hand_rank()
            if hasattr(module, 'test_evaluator_3_cards'):
                module.test_evaluator_3_cards()
            if hasattr(module, 'test_evaluator_5_cards'):
                module.test_evaluator_5_cards()
            if hasattr(module, 'test_compare'):
                module.test_compare()
            
            print(f"✅ {filename} - ALL TESTS PASSED")
            passed += 1
            
        except Exception as e:
            print(f"❌ {filename} - TESTS FAILED")
            print(f"Error: {e}")
            traceback.print_exc()
            failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)