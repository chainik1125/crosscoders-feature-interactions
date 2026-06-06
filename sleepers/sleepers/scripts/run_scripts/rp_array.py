#!/usr/bin/env python3
import subprocess
import os
import argparse
import time
import pathlib

# Your RunPod SSH config alias
RUNPOD_ALIAS = "rp_a40"

# Default settings
default_repo = "https://github.com/JasonGross/crosscoders-feature-interactions"
default_branch = "dev_nobias"
default_dir = "crosscoders-feature-interactions"
default_script_dir = "sleepers/sleepers/scripts/train_topk_sleeper"
default_script = "run.py"
default_yaml_dir = "sleepers/sleepers/scripts/train_topk_sleeper"

def run_ssh_command(command, timeout=30, debug=False):
    """Run a command over SSH using the configured alias."""
    ssh_cmd = ["ssh", "-o", "ForwardAgent=yes", RUNPOD_ALIAS, command]
    if debug:
        print(f"Running SSH command: {' '.join(ssh_cmd)}")
    try:
        result = subprocess.run(
            ssh_cmd, 
            capture_output=True, 
            text=True,
            timeout=timeout
        )
        if result.returncode != 0:
            print(f"Command failed with error: {result.stderr}")
            return "", result.returncode
        return result.stdout.strip(), result.returncode
    except subprocess.TimeoutExpired:
        print(f"SSH command timed out after {timeout} seconds")
        return "", 1

def scp_file_to_pod(local_path, remote_path):
    """Copy a file to the pod using SCP."""
    try:
        scp_cmd = ["scp", local_path, f"{RUNPOD_ALIAS}:{remote_path}"]
        result = subprocess.run(scp_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"SCP failed: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Error copying file: {e}")
        return False

def convert_https_to_ssh(url):
    """Convert HTTPS GitHub URLs to SSH format for authentication."""
    if not url or not isinstance(url, str):
        return url
    
    # Check if it's already SSH format
    if url.startswith('git@'):
        return url
    
    # Convert from HTTPS to SSH format
    if url.startswith('https://github.com/'):
        return url.replace('https://github.com/', 'git@github.com:')
    
    return url

def parse_args():
    parser = argparse.ArgumentParser(description="Run a job on RunPod")
    parser.add_argument("yaml_file", help="YAML configuration file for the job (local or relative to repo)")
    parser.add_argument("--repo", help=f"Git repository to clone (default: {default_repo})", default=default_repo)
    parser.add_argument("--branch", help=f"Git branch to use (default: {default_branch})", default=default_branch)
    parser.add_argument("--base-dir", help=f"Base directory on pod for the repository (default: /root/{default_dir})", 
                       default=f"/root/{default_dir}")
    parser.add_argument("--script-dir", help=f"Directory containing the script (default: {default_script_dir})", 
                       default=default_script_dir)
    parser.add_argument("--script", help=f"Script to run (default: {default_script})", 
                       default=default_script)
    parser.add_argument("--yaml-dir", help=f"Directory for yaml file on pod (default: {default_yaml_dir})", 
                       default=default_yaml_dir)
    parser.add_argument("--local-yaml", action="store_true", 
                       help="If set, yaml_file is a local file to copy. Otherwise, it's a path relative to yaml-dir")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Handle different YAML file scenarios
    if args.local_yaml:
        # Check if local yaml file exists
        if not os.path.exists(args.yaml_file):
            print(f"Error: Local YAML file '{args.yaml_file}' not found")
            return
        yaml_is_local = True
    else:
        # Check if it's a local file anyway (for convenience)
        if os.path.exists(args.yaml_file):
            print(f"Found '{args.yaml_file}' locally. Treating as local file to upload.")
            yaml_is_local = True
        else:
            print(f"YAML file '{args.yaml_file}' will be used from the repository.")
            yaml_is_local = False
    
    # Convert HTTPS GitHub URLs to SSH format for authentication
    args.repo = convert_https_to_ssh(args.repo)
    print(f"Using repository URL: {args.repo}")
    print(f"Using branch: {args.branch}")
    
    # Ensure base_dir is an absolute path
    if not args.base_dir.startswith('/'):
        base_dir = f"/root/{args.base_dir}"
    else:
        base_dir = args.base_dir
        
    print(f"Using absolute base directory path: {base_dir}")
    
    # Build the full paths
    repo_script_dir = os.path.join(base_dir, args.script_dir)
    full_script_path = os.path.join(args.script_dir, args.script)
    repo_yaml_dir = os.path.join(base_dir, args.yaml_dir)
    
    print(f"Base directory: {base_dir}")
    print(f"Script directory: {repo_script_dir}")
    print(f"YAML directory: {repo_yaml_dir}")
    
    # Step 1: Create directory if it doesn’t exist
    run_ssh_command(f"mkdir -p {base_dir}")
    
    # Step 2: Clone or pull the repository with specific branch
    print(f"Updating code repository: {args.repo} (branch: {args.branch})")
    
    # Check if directory exists with git repo
    check_dir, _ = run_ssh_command(f"if [ -d {base_dir}/.git ]; then echo 'exists'; else echo 'not_exists'; fi")
    print(f"Git repository exists check result: {check_dir}")
    
    if check_dir.strip() == 'exists':
        # Set the remote URL to SSH to ensure SSH authentication
        print(f"Setting remote URL to: {args.repo}")
        run_ssh_command(f"cd {base_dir} && git remote set-url origin {args.repo}")
        
        # Repository exists, fetch changes and checkout specified branch
        update_cmd = f"cd {base_dir} && git fetch && git checkout {args.branch} && git pull origin {args.branch}"
        print(f"Running update command: {update_cmd}")
        output, return_code = run_ssh_command(update_cmd)
        if return_code != 0:
            print(f"Failed to update repository: {output}")
            return
        print("Repository updated successfully")
    else:
        # Clone the repository with specific branch
        clone_cmd = f"git clone -b {args.branch} {args.repo} {base_dir}"
        print(f"Running clone command: {clone_cmd}")
        output, return_code = run_ssh_command(clone_cmd)
        if return_code != 0:
            print(f"Failed to clone repository: {output}")
            # Try cloning without branch specification, then checkout the branch
            print("Trying to clone main branch and then checkout...")
            clone_main_cmd = f"git clone {args.repo} {base_dir} && cd {base_dir} && git checkout {args.branch}"
            output, return_code = run_ssh_command(clone_main_cmd)
            if return_code != 0:
                print(f"Failed to clone repository: {output}")
                return
        print(f"Repository cloned successfully and switched to branch {args.branch}")
    print(f"Using repository URL: {args.repo}")
    print(f"Using branch: {args.branch}")
    
    # Ensure base_dir is an absolute path
    if not args.base_dir.startswith('/'):
        base_dir = f"/root/{args.base_dir}"
    else:
        base_dir = args.base_dir
        
    print(f"Using absolute base directory path: {base_dir}")
    
    # Build the full paths
    repo_script_dir = os.path.join(base_dir, args.script_dir)
    full_script_path = os.path.join(args.script_dir, args.script)
    repo_yaml_dir = os.path.join(base_dir, args.yaml_dir)
    
    print(f"Base directory: {base_dir}")
    print(f"Script directory: {repo_script_dir}")
    print(f"YAML directory: {repo_yaml_dir}")
    
    # Step 1: Create directory if it doesn't exist
    run_ssh_command(f"mkdir -p {base_dir}")
    
    # Step 2: Clone or pull the repository with specific branch
    print(f"Updating code repository: {args.repo} (branch: {args.branch})")
    
    # Check if directory exists with git repo
    check_dir, _ = run_ssh_command(f"if [ -d {base_dir}/.git ]; then echo 'exists'; else echo 'not_exists'; fi")
    print(f"Git repository exists check result: {check_dir}")
    
    if check_dir.strip() == 'exists':
        # Repository exists, fetch changes and checkout specified branch
        update_cmd = f"cd {base_dir} && git fetch && git checkout {args.branch} && git pull origin {args.branch}"
        print(f"Running update command: {update_cmd}")
        output, return_code = run_ssh_command(update_cmd)
        if return_code != 0:
            print(f"Failed to update repository: {output}")
            return
        print("Repository updated successfully")
    else:
        # Clone the repository with specific branch
        clone_cmd = f"git clone -b {args.branch} {args.repo} {base_dir}"
        print(f"Running clone command: {clone_cmd}")
        output, return_code = run_ssh_command(clone_cmd)
        if return_code != 0:
            print(f"Failed to clone repository: {output}")
            # Try cloning without branch specification, then checkout the branch
            print("Trying to clone main branch and then checkout...")
            clone_main_cmd = f"git clone {args.repo} {base_dir} && cd {base_dir} && git checkout {args.branch}"
            output, return_code = run_ssh_command(clone_main_cmd)
            if return_code != 0:
                print(f"Failed to clone repository: {output}")
                return
        print(f"Repository cloned successfully and switched to branch {args.branch}")
    
    # Make sure the script and yaml directories exist
    run_ssh_command(f"mkdir -p {repo_script_dir}")
    run_ssh_command(f"mkdir -p {repo_yaml_dir}")
    
    # Step 3: Handle YAML file (copy if local, use from repo if not)
    yaml_filename = os.path.basename(args.yaml_file)
    if yaml_is_local:
        remote_yaml_path = f"{repo_yaml_dir}/{yaml_filename}"
        print(f"Copying YAML file to pod: {remote_yaml_path}")
        if not scp_file_to_pod(args.yaml_file, remote_yaml_path):
            print("Failed to copy YAML file to pod")
            return
    else:
        # Use the YAML file from the repository
        yaml_filename = args.yaml_file
        remote_yaml_path = f"{repo_yaml_dir}/{yaml_filename}"
        print(f"Using YAML file from repository: {remote_yaml_path}")
        # Check if the file exists
        check_yaml, _ = run_ssh_command(f"if [ -f {remote_yaml_path} ]; then echo 'exists'; else echo 'not_exists'; fi")
        if check_yaml.strip() != 'exists':
            print(f"Error: YAML file '{remote_yaml_path}' not found in repository")
            return
    
    # Step 4: Run the script
    # Construct the command to run
    run_cmd = f"cd {base_dir} && python {full_script_path} {remote_yaml_path}"
    print(f"Running command: {run_cmd}")
    
    # Create log directory
    log_dir = os.path.join(base_dir, "logs")
    run_ssh_command(f"mkdir -p {log_dir}")
    log_file = f"{log_dir}/job_output_{int(time.time())}.log"
    
    # Submit the job to run in the background
    job_cmd = f"cd {base_dir} && nohup {run_cmd} > {log_file} 2>&1 & echo $!"
    job_id, return_code = run_ssh_command(job_cmd)
    
    if return_code != 0 or not job_id:
        print("Failed to start job")
        return
    
    print(f"Job started with PID: {job_id}")
    
    # Step 5: Show initial output
    time.sleep(2)  # Give job time to start
    output, _ = run_ssh_command(f"tail -10 {log_file}")
    print("\nInitial job output:")
    print(output)
    
    print(f"\nTo check job status: ssh {RUNPOD_ALIAS} ps -p {job_id}")
    print(f"To view job output: ssh {RUNPOD_ALIAS} tail -f {log_file}")

if __name__ == "__main__":
    main()