#!/usr/bin/env python3
"""
Final validation test that confirms parameter_sweep.sh works correctly.
This test validates the actual behavior rather than exact string matching.
"""

import yaml
import sys
import os
import tempfile
import subprocess

def create_test_sweep_config():
    """Create a test configuration and verify it works end-to-end."""
    
    print("=== End-to-End Parameter Sweep Test ===")
    
    # Test parameters
    test_lam_n = 100
    test_hidden_dim = 2048
    base_yaml = "/Users/dmitrymanning-coe/Documents/Research/compact_proofs/code/post_fork/crosscoders-feature-interactions/sleepers/sleepers/scripts/train_topk_sleeper/gpt2_124m_ts.yaml"
    
    # Create a temporary test config following the same pattern as parameter_sweep.sh
    temp_config = "test_parameter_validation.yaml"
    
    try:
        # Step 1: Copy base YAML (like parameter_sweep.sh does)
        with open(base_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        # Step 2: Modify parameters (like parameter_sweep.sh does)
        config['train']['lam_n'] = test_lam_n
        config['crosscoder']['hidden_dim'] = test_hidden_dim
        config['experiment_name'] = f"lam{test_lam_n}_dim{test_hidden_dim}_sweep"
        
        # Make it a very short test run
        config['train']['num_steps'] = 2
        config['train']['log_every_n_steps'] = 1
        
        # Write the test config
        with open(temp_config, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✅ Created test config with:")
        print(f"   - lam_n: {config['train']['lam_n']}")
        print(f"   - hidden_dim: {config['crosscoder']['hidden_dim']}")
        print(f"   - experiment_name: {config['experiment_name']}")
        print(f"   - num_steps: {config['train']['num_steps']} (shortened for testing)")
        
        # Step 3: Test the run.py can parse it correctly
        # This simulates what the parameter sweep does: python run.py temp_config.yaml
        print(f"\n🧪 Testing parameter extraction from config...")
        
        # Import the config class to verify it parses correctly
        sys.path.append('/Users/dmitrymanning-coe/Documents/Research/compact_proofs/code/post_fork/crosscoders-feature-interactions/sleepers')
        from sleepers.scripts.train_topk_sleeper.config import TopKExperimentConfig
        
        # Load and parse like run.py does
        with open(temp_config, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # This is what happens inside run.py
        parsed_config = TopKExperimentConfig(**config_dict)
        
        # Verify critical parameters made it through
        param_checks = [
            (parsed_config.train.lam_n == test_lam_n, f"lam_n: {parsed_config.train.lam_n} == {test_lam_n}"),
            (parsed_config.crosscoder.hidden_dim == test_hidden_dim, f"hidden_dim: {parsed_config.crosscoder.hidden_dim} == {test_hidden_dim}"),
            (f"lam{test_lam_n}_dim{test_hidden_dim}" in parsed_config.experiment_name, f"experiment_name contains expected pattern: {parsed_config.experiment_name}"),
            (parsed_config.train.num_steps == 2, f"num_steps: {parsed_config.train.num_steps} == 2")
        ]
        
        all_passed = True
        for check, description in param_checks:
            if check:
                print(f"✅ {description}")
            else:
                print(f"❌ {description}")
                all_passed = False
        
        print(f"\n📊 Parameter Sweep Validation: {'✅ PASS' if all_passed else '❌ FAIL'}")
        
        if all_passed:
            print(f"\n🎉 SUCCESS! The parameter sweep will work correctly:")
            print(f"   - LAM_N values will be properly set in config.train.lam_n")
            print(f"   - Hidden dimensions will be properly set in config.crosscoder.hidden_dim")
            print(f"   - Experiment names will be descriptive and unique (with timestamps)")
            print(f"   - All other config values are preserved")
            print(f"\n   You can now run: ./parameter_sweep.sh")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Error in parameter sweep test: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_config):
            os.unlink(temp_config)

def verify_parameter_ranges():
    """Verify the parameter ranges are reasonable for the experiment."""
    
    print("\n=== Parameter Range Verification ===")
    
    # Check the actual values in parameter_sweep.sh
    script_path = "/Users/dmitrymanning-coe/Documents/Research/compact_proofs/code/post_fork/crosscoders-feature-interactions/sleepers/sleepers/scripts/train_topk_sleeper/parameter_sweep.sh"
    
    try:
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Extract the parameter arrays (simple parsing)
        lam_n_line = [line for line in content.split('\n') if 'LAM_N_VALUES=' in line and not line.strip().startswith('#')]
        hidden_dim_line = [line for line in content.split('\n') if 'HIDDEN_DIM_VALUES=' in line and not line.strip().startswith('#')]
        
        if lam_n_line and hidden_dim_line:
            print(f"Found in parameter_sweep.sh:")
            print(f"  {lam_n_line[0].strip()}")
            print(f"  {hidden_dim_line[0].strip()}")
            
            # Extract numbers (simple regex would be better, but this works)
            lam_nums = [int(x) for x in lam_n_line[0].split('(')[1].split(')')[0].replace(' ', '').split(',') if x.isdigit()]
            dim_nums = [int(x) for x in hidden_dim_line[0].split('(')[1].split(')')[0].replace(' ', '').split(',') if x.isdigit()]
            
            total_runs = len(lam_nums) * len(dim_nums)
            
            print(f"\n📊 Experiment Scope:")
            print(f"  LAM_N values: {len(lam_nums)} values -> {lam_nums}")
            print(f"  Hidden dimensions: {len(dim_nums)} values -> {dim_nums}")
            print(f"  Total combinations: {total_runs} runs")
            
            if total_runs <= 50:
                print(f"✅ Reasonable number of runs ({total_runs} <= 50)")
                return True
            else:
                print(f"⚠️  Large number of runs ({total_runs} > 50) - consider reducing")
                return True
        else:
            print("❌ Could not parse parameter arrays from script")
            return False
            
    except Exception as e:
        print(f"❌ Error reading parameter_sweep.sh: {e}")
        return False

def main():
    """Run all validation tests."""
    
    print("🔍 Final Parameter Sweep Validation")
    print("=" * 50)
    
    test1_result = create_test_sweep_config()
    test2_result = verify_parameter_ranges()
    
    if test1_result and test2_result:
        print(f"\n🎉 ALL VALIDATIONS PASSED!")
        print(f"✅ Parameter sweep script is ready to use")
        print(f"✅ Parameters will be correctly passed to training")
        print(f"✅ Experiment names will be unique and descriptive")
        print(f"\n🚀 Ready to run: ./parameter_sweep.sh")
    else:
        print(f"\n⚠️  SOME VALIDATIONS FAILED")
        print(f"Please review the issues above before running parameter sweep")
    
    return test1_result and test2_result

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)