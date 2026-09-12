"""Audit backend code for GitHub push and RunPod deployment."""
from pathlib import Path
import subprocess


def audit_backend_for_github():
    """Audit the backend/ folder for GitHub push and RunPod deployment readiness."""
    print("=" * 60)
    print("TEST 11: Backend Code Audit for GitHub Push")
    print("=" * 60)
    
    backend_path = Path("backend")
    
    if not backend_path.exists():
        print("  FAIL: backend/ folder not found")
        return
    
    # Check required files
    required_files = [
        "Dockerfile",
        "README.md",
        "api/app/main.py",
        "api/app/__init__.py",
        "api/requirements.txt",
        "worker/handler.py",
        "worker/requirements.txt",
        "worker/Dockerfile",
        ".dockerignore",
        ".github/workflows/deploy-runpod.yml",
    ]
    
    print("  Checking required files:")
    all_present = True
    for file in required_files:
        filepath = backend_path / file
        exists = filepath.exists()
        size = filepath.stat().st_size if exists else 0
        status = "OK" if exists and size > 0 else "MISSING/EMPTY"
        print(f"    {status}: {file} ({size} bytes)")
        if not exists or size == 0:
            all_present = False
    
    # Check for .pyc files that should be in .gitignore
    pyc_files = list(backend_path.rglob("__pycache__"))
    if pyc_files:
        print(f"\n  ⚠ Found {len(pyc_files)} __pycache__ dirs (should be in .gitignore)")
        for pyc in pyc_files[:3]:
            print(f"    {pyc.relative_to('backend')}")
    
    # Check if git is initialized
    git_dir = Path(".git")
    if git_dir.exists():
        print("\n  OK: Git repository initialized")
        
        # Check git status
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                cwd="."
            )
            if result.stdout.strip():
                print(f"\n  ⚠ Git has uncommitted changes:")
                for line in result.stdout.strip().split('\n')[:10]:
                    print(f"    {line}")
            else:
                print("\n  OK: All changes committed")
        except Exception:
            print("\n  Could not check git status")
    else:
        print("\n  ⚠ No .git directory found")
    
    # Check Dockerfiles
    print("\n  Checking Dockerfile configurations:")
    
    # Root Dockerfile (for worker)
    root_docker = backend_path / "Dockerfile"
    if root_docker.exists():
        content = root_docker.read_text()
        print(f"    Root Dockerfile: {len(content)} bytes")
        if "FROM" in content and "handler.py" in content:
            print("      OK: Properly configured for worker deployment")
        else:
            print("      ⚠ May need review for worker deployment")
    
    # API Dockerfile
    api_docker = backend_path / "api" / "Dockerfile"
    if api_docker.exists():
        content = api_docker.read_text()
        print(f"    API Dockerfile: {len(content)} bytes")
        if "FROM" in content and "uvicorn" in content:
            print("      OK: Properly configured for API deployment")
        else:
            print("      ⚠ May need review for API deployment")
    
    # Worker Dockerfile
    worker_docker = backend_path / "worker" / "Dockerfile"
    if worker_docker.exists():
        content = worker_docker.read_text()
        print(f"    Worker Dockerfile: {len(content)} bytes")
        if "FROM" in content:
            print("      OK: Has proper base image")
        else:
            print("      ⚠ May need review")
    
    # Check GitHub Actions workflow
    workflow = backend_path / ".github" / "workflows" / "deploy-runpod.yml"
    if workflow.exists():
        content = workflow.read_text()
        print(f"\n    GitHub Actions workflow: {len(content)} bytes")
        if "push" in content and "runpod" in content.lower():
            print("      OK: Configured for automatic deployment")
        else:
            print("      ⚠ May need review")
    else:
        print("\n  ⚠ No GitHub Actions workflow found")
    
    # Check for secrets/hardcoded credentials
    print("\n  Checking for hardcoded credentials:")
    secret_patterns = ["password", "api_key", "secret", "token"]
    for py_file in backend_path.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        try:
            content = py_file.read_text()
            for pattern in secret_patterns:
                if pattern in content.lower() and "=" in content:
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if pattern.lower() in line.lower() and '=' in line:
                            if 'os.environ' not in line and 'os.getenv' not in line:
                                print(f"    ⚠ {py_file.relative_to('backend')}:{i+1} - Potential hardcoded {pattern}")
        except Exception:
            pass
    
    print("\n  ✅ Backend code audit complete")
    print("  📋 Summary:")
    print("    - All required files present" if all_present else "    - ❌ Some files missing!")
    print("    - Ready for git push" if all_present else "    - ❌ Fix issues before push")
    print()


if __name__ == "__main__":
    audit_backend_for_github()
