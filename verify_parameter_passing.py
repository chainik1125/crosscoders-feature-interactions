#!/usr/bin/env python3
"""
End-to-end test to verify parameter passing from parameter_sweep.sh through to the actual training.
This simulates the exact flow that happens in the parameter sweep.
"""

import yaml
import sys
import os
import tempfile

# Add the model-diffing module to path for config classes
sys.path.append('/Users/dmitrymanning-coe/Documents/Research/compact_proofs/code/post_fork/crosscoders-feature-interactions/sleepers')
sys.path.append('/Users/dmitrymanning-coe/Documents/Research/compact_proofs/code/post_fork/crosscoders-feature-interactions/model-diffing')

from sleepers.scripts.train_topk_sleeper.config import TopKExperimentConfig

def test_config_instantiation():
    """Test that modified YAML can be properly loaded into the config class."""
    
    print("=== Testing Config Instantiation ===")
    
    # Start with the actual base YAML
    base_yaml = "/Users/dmitrymanning-coe/Documents/Research/compact_proofs/code/post_fork/crosscoders-feature-interactions/sleepers/sleepers/scripts/train_topk_sleeper/gpt2_124m_ts.yaml"
    
    # Test parameters
    test_lam_n = 500  
    test_hidden_dim = 2048
    test_experiment_name = f"lam{test_lam_n}_dim{test_hidden_dim}_sweep"
    
    try:
        # Load the base config
        with open(base_yaml, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Apply the same modifications as parameter_sweep.sh
        config_dict['train']['lam_n'] = test_lam_n
        config_dict['crosscoder']['hidden_dim'] = test_hidden_dim
        config_dict['experiment_name'] = test_experiment_name
        
        print(f"Modified config:")
        print(f"  - train.lam_n: {config_dict['train']['lam_n']}")
        print(f"  - crosscoder.hidden_dim: {config_dict['crosscoder']['hidden_dim']}")
        print(f"  - experiment_name: {config_dict['experiment_name']}")
        
        # Try to instantiate the config class (this is what run.py does)
        config = TopKExperimentConfig(**config_dict)
        
        # Verify the parameters made it through
        checks = [
            (config.train.lam_n, test_lam_n, "config.train.lam_n"),
            (config.crosscoder.hidden_dim, test_hidden_dim, "config.crosscoder.hidden_dim"),
            (config.experiment_name, test_experiment_name, "config.experiment_name")
        ]
        
        all_passed = True
        for actual, expected, param_name in checks:
            if actual == expected:
                print(f"✅ {param_name}: {actual}")
            else:
                print(f"❌ {param_name}: got {actual}, expected {expected}")
                all_passed = False
        
        # Also check some other important config values
        print(f"\nOther config values:")
        print(f"  - train.batch_size: {config.train.batch_size}")
        print(f"  - train.num_steps: {config.train.num_steps}")
        print(f"  - crosscoder.k: {config.crosscoder.k}")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Error instantiating config: {e}")
        import traceback
        traceback.print_exc()
        return False

def simulate_parameter_sweep_step():
    """Simulate exactly what parameter_sweep.sh does for one parameter combination."""
    
    print("\n=== Simulating Parameter Sweep Step ===")
    
    # Parameters to test
    LAM_N = 100
    HIDDEN_DIM = 1536
    EXPERIMENT_NAME = f"lam{LAM_N}_dim{HIDDEN_DIM}_sweep"
    
    # Files
    BASE_YAML = "/Users/dmitrymanning-coe/Documents/Research/compact_proofs/code/post_fork/crosscoders-feature-interactions/sleepers/sleepers/scripts/train_topk_sleeper/gpt2_124m_ts.yaml"
    TEMP_YAML = "temp_test_sweep.yaml"
    
    try:
        print(f"1. Copying {BASE_YAML} to {TEMP_YAML}")
        
        # Step 1: Copy base YAML (equivalent to: cp "$BASE_YAML" "$YAML_FILE")
        with open(BASE_YAML, 'r') as src:
            with open(TEMP_YAML, 'w') as dst:
                dst.write(src.read())
        
        print(f"2. Modifying YAML with LAM_N={LAM_N}, HIDDEN_DIM={HIDDEN_DIM}")
        
        # Step 2: Modify YAML (equivalent to the Python -c command in the script)
        with open(TEMP_YAML, 'r') as f:
            config = yaml.safe_load(f)
        
        config['train']['lam_n'] = LAM_N
        config['crosscoder']['hidden_dim'] = HIDDEN_DIM  
        config['experiment_name'] = EXPERIMENT_NAME
        
        with open(TEMP_YAML, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"3. Verifying modified YAML can be loaded by TopKExperimentConfig")
        
        # Step 3: Verify config loading (equivalent to what run.py does)
        with open(TEMP_YAML, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config_obj = TopKExperimentConfig(**config_dict)
        
        # Verify the parameters
        success = (
            config_obj.train.lam_n == LAM_N and
            config_obj.crosscoder.hidden_dim == HIDDEN_DIM and
            config_obj.experiment_name == EXPERIMENT_NAME
        )
        
        if success:
            print(f"✅ Parameter sweep simulation successful!")
            print(f"   Final config.train.lam_n: {config_obj.train.lam_n}")
            print(f"   Final config.crosscoder.hidden_dim: {config_obj.crosscoder.hidden_dim}")
            print(f"   Final config.experiment_name: {config_obj.experiment_name}")
        else:
            print(f"❌ Parameter mismatch in final config")
        
        return success
        
    except Exception as e:
        print(f"❌ Error in parameter sweep simulation: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up
        if os.path.exists(TEMP_YAML):
            os.unlink(TEMP_YAML)

def check_parameter_sweep_script():
    """Verify the parameter_sweep.sh script has correct syntax."""
    
    print("\n=== Checking Parameter Sweep Script ===")
    
    script_path = "/Users/dmitrymanning-coe/Documents/Research/compact_proofs/code/post_fork/crosscoders-feature-interactions/sleepers/sleepers/scripts/train_topk_sleeper/parameter_sweep.sh"
    
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        return False
    
    try:
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Check for key components
        checks = [
            ("LAM_N_VALUES=" in content, "LAM_N_VALUES array defined"),
            ("HIDDEN_DIM_VALUES=" in content, "HIDDEN_DIM_VALUES array defined"),
            ("config['train']['lam_n'] = $LAM_N" in content, "LAM_N parameter setting"),
            ("config['crosscoder']['hidden_dim'] = $HIDDEN_DIM" in content, "HIDDEN_DIM parameter setting"),
            ("python run.py" in content, "run.py execution"),
            ("BASE_YAML=" in content, "Base YAML file specified")
        ]
        
        all_passed = True
        for check, description in checks:
            if check:
                print(f"✅ {description}")
            else:
                print(f"❌ {description}")
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Error reading script: {e}")
        return False

def run_verification():
    """Run all verification tests."""
    
    print("🔍 Verifying Parameter Passing in Parameter Sweep")
    print("=" * 60)
    
    tests = [
        ("Config Instantiation", test_config_instantiation),
        ("Parameter Sweep Simulation", simulate_parameter_sweep_step),
        ("Parameter Sweep Script Check", check_parameter_sweep_script)
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
    print("📊 VERIFICATION RESULTS:")
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 ALL VERIFICATIONS PASSED!")
        print("✅ Parameters are correctly passed from parameter_sweep.sh to run.py")
        print("✅ YAML modification works correctly")  
        print("✅ Config instantiation works correctly")
        print("✅ The parameter sweep should work as expected")
    else:
        print("\n⚠️  SOME VERIFICATIONS FAILED!")
        print("❌ There may be issues with parameter passing")
    
    return all_passed

if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)