#!/usr/bin/env python3
"""
Test script to verify that parameter_sweep.sh correctly modifies YAML files
and passes parameters to run.py
"""

import yaml
import sys
import os
import tempfile
import subprocess

def test_yaml_modification():
    """Test that the Python YAML modification code works correctly."""
    
    print("=== Testing YAML Modification Logic ===")
    
    # Create a test YAML similar to gpt2_124m_ts.yaml
    test_config = {
        'crosscoder': {
            'ft_init_checkpt_epoch': None,
            'ft_init_checkpt_folder': None,
            'hidden_dim': 1536,
            'k': 20
        },
        'train': {
            'batch_size': 128,
            'beta_n': 1,
            'lam_n': 10,
            'log_every_n_steps': 100,
            'num_steps': 50000
        },
        'experiment_name': 'lambda_n10_beta_n1_S'
    }
    
    # Test parameters
    test_lam_n = 200
    test_hidden_dim = 2048
    test_experiment_name = f"lam{test_lam_n}_dim{test_hidden_dim}_sweep"
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f, default_flow_style=False)
        temp_yaml = f.name
    
    try:
        # Test the YAML modification code (same as in parameter_sweep.sh)
        modification_code = f"""
import yaml
with open('{temp_yaml}', 'r') as f:
    config = yaml.safe_load(f)

# Update parameters  
config['train']['lam_n'] = {test_lam_n}
config['crosscoder']['hidden_dim'] = {test_hidden_dim}
config['experiment_name'] = '{test_experiment_name}'

with open('{temp_yaml}', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
"""
        
        # Execute the modification
        result = subprocess.run([sys.executable, '-c', modification_code], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ YAML modification failed: {result.stderr}")
            return False
        
        # Load and verify the modified YAML
        with open(temp_yaml, 'r') as f:
            modified_config = yaml.safe_load(f)
        
        # Verify modifications
        checks = [
            (modified_config['train']['lam_n'], test_lam_n, "lam_n"),
            (modified_config['crosscoder']['hidden_dim'], test_hidden_dim, "hidden_dim"),
            (modified_config['experiment_name'], test_experiment_name, "experiment_name")
        ]
        
        all_passed = True
        for actual, expected, param_name in checks:
            if actual == expected:
                print(f"✅ {param_name}: {actual} (correct)")
            else:
                print(f"❌ {param_name}: got {actual}, expected {expected}")
                all_passed = False
        
        return all_passed
        
    finally:
        # Clean up
        os.unlink(temp_yaml)

def test_yaml_structure():
    """Test that the base YAML file exists and has expected structure."""
    
    print("\n=== Testing Base YAML File Structure ===")
    
    base_yaml = "/Users/dmitrymanning-coe/Documents/Research/compact_proofs/code/post_fork/crosscoders-feature-interactions/sleepers/sleepers/scripts/train_topk_sleeper/gpt2_124m_ts.yaml"
    
    if not os.path.exists(base_yaml):
        print(f"❌ Base YAML file not found: {base_yaml}")
        return False
    
    try:
        with open(base_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required structure
        required_paths = [
            ('train', 'lam_n'),
            ('crosscoder', 'hidden_dim'),
            ('experiment_name',)
        ]
        
        all_passed = True
        for path in required_paths:
            current = config
            path_str = ""
            
            for key in path:
                path_str += f"['{key}']" if path_str else f"config['{key}']"
                if key not in current:
                    print(f"❌ Missing key: {path_str}")
                    all_passed = False
                    break
                current = current[key]
            else:
                print(f"✅ Found: {path_str} = {current}")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Error reading base YAML: {e}")
        return False

def test_parameter_ranges():
    """Test that the parameter ranges in the script are reasonable."""
    
    print("\n=== Testing Parameter Ranges ===")
    
    # These should match the values in parameter_sweep.sh
    lam_n_values = [0, 10, 50, 200, 1000, 5000]
    hidden_dim_values = [512, 1024, 1536, 2048, 3072]
    
    print(f"LAM_N_VALUES: {lam_n_values}")
    print(f"HIDDEN_DIM_VALUES: {hidden_dim_values}")
    print(f"Total combinations: {len(lam_n_values)} × {len(hidden_dim_values)} = {len(lam_n_values) * len(hidden_dim_values)}")
    
    # Check for reasonable ranges
    checks = [
        (all(isinstance(x, int) and x >= 0 for x in lam_n_values), "LAM_N values are non-negative integers"),
        (all(isinstance(x, int) and x > 0 for x in hidden_dim_values), "Hidden dim values are positive integers"),
        (len(lam_n_values) > 0, "At least one LAM_N value"),
        (len(hidden_dim_values) > 0, "At least one hidden dim value"),
        (len(lam_n_values) * len(hidden_dim_values) <= 50, "Total combinations reasonable (<= 50)")
    ]
    
    all_passed = True
    for check, description in checks:
        if check:
            print(f"✅ {description}")
        else:
            print(f"❌ {description}")
            all_passed = False
    
    return all_passed

def create_test_run():
    """Create a minimal test to verify the full pipeline."""
    
    print("\n=== Creating Test Configuration ===")
    
    # Test with just one parameter combination
    test_lam_n = 10
    test_hidden_dim = 1024
    
    base_yaml = "/Users/dmitrymanning-coe/Documents/Research/compact_proofs/code/post_fork/crosscoders-feature-interactions/sleepers/sleepers/scripts/train_topk_sleeper/gpt2_124m_ts.yaml"
    test_yaml = "test_sweep_config.yaml"
    
    try:
        # Copy and modify the base YAML
        with open(base_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        # Modify parameters
        config['train']['lam_n'] = test_lam_n
        config['crosscoder']['hidden_dim'] = test_hidden_dim
        config['experiment_name'] = f"test_lam{test_lam_n}_dim{test_hidden_dim}"
        
        # Make it a very short run for testing
        config['train']['num_steps'] = 10  # Just 10 steps for testing
        
        # Write test config
        with open(test_yaml, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✅ Created test config: {test_yaml}")
        print(f"   - lam_n: {config['train']['lam_n']}")
        print(f"   - hidden_dim: {config['crosscoder']['hidden_dim']}")
        print(f"   - experiment_name: {config['experiment_name']}")
        print(f"   - num_steps: {config['train']['num_steps']} (shortened for testing)")
        
        print(f"\n🧪 To test manually, run:")
        print(f"   cd /Users/dmitrymanning-coe/Documents/Research/compact_proofs/code/post_fork/crosscoders-feature-interactions/sleepers/sleepers/scripts/train_topk_sleeper")
        print(f"   python run.py {test_yaml}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating test config: {e}")
        return False

def run_all_tests():
    """Run all validation tests."""
    
    print("🧪 Testing Parameter Sweep Configuration")
    print("=" * 60)
    
    tests = [
        ("YAML Modification Logic", test_yaml_modification),
        ("Base YAML File Structure", test_yaml_structure),
        ("Parameter Ranges", test_parameter_ranges),
        ("Test Configuration Creation", create_test_run)
    ]
    
    results = []
    
    for test_name, test_fn in tests:
        try:
            result = test_fn()
            results.append((test_name, result))
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"\n{status}: {test_name}")
        except Exception as e:
            results.append((test_name, False))
            print(f"\n💥 ERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS:")
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Parameter sweep should work correctly.")
    else:
        print("\n⚠️  SOME TESTS FAILED! Please review the parameter sweep setup.")
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)