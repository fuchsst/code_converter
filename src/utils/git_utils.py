# src/utils/git_utils.py
import subprocess
import os
from src.logger_setup import get_logger

logger = get_logger(__name__)

def create_git_commit(repo_path: str, commit_message: str) -> bool:
    """
    Stages all changes and creates a git commit in the specified repository.

    Args:
        repo_path: The absolute path to the git repository.
        commit_message: The message for the git commit.

    Returns:
        True if the commit was successful, False otherwise.
    """
    if not os.path.isdir(os.path.join(repo_path, '.git')):
        logger.error(f"Repository path '{repo_path}' is not a git repository or .git directory not found.")
        return False

    try:
        # Stage all changes
        add_process = subprocess.run(
            ['git', 'add', '.'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=False  # Don't raise exception on non-zero exit
        )
        if add_process.returncode != 0:
            logger.error(f"Failed to stage changes in '{repo_path}'. Error: {add_process.stderr}")
            return False
        logger.info(f"Successfully staged changes in '{repo_path}'.")

        # Commit changes
        commit_process = subprocess.run(
            ['git', 'commit', '-m', commit_message],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=False # Don't raise exception on non-zero exit
        )
        if commit_process.returncode != 0:
            # It's possible commit fails if there are no changes to commit after 'git add .'
            # (e.g., if all files were already staged and committed, or .gitignore ignores everything new)
            # We check stderr for "nothing to commit" or similar messages.
            if "nothing to commit" in commit_process.stdout.lower() or \
               "nothing to commit" in commit_process.stderr.lower() or \
               "no changes added to commit" in commit_process.stdout.lower() or \
               "no changes added to commit" in commit_process.stderr.lower():
                logger.info(f"No changes to commit in '{repo_path}' for message: '{commit_message}'.")
                return True # Considered a success as there's nothing to do
            logger.error(f"Failed to commit changes in '{repo_path}'. Message: '{commit_message}'. Error: {commit_process.stderr} Stdout: {commit_process.stdout}")
            return False
        
        logger.info(f"Successfully committed changes in '{repo_path}' with message: '{commit_message}'.")
        return True

    except FileNotFoundError:
        logger.error("Git command not found. Please ensure Git is installed and in your system's PATH.")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during git operation in '{repo_path}': {e}", exc_info=True)
        return False

#if __name__ == '__main__':
#    # Example usage (for testing purposes)
#    # Create a dummy repo:
#    # mkdir temp_repo
#    # cd temp_repo
#    # git init
#    # echo "hello" > test.txt
#    # python ../src/utils/git_utils.py
#
#    dummy_repo_path = os.path.abspath("./temp_repo_for_git_utils_test") # More specific name
#    if not os.path.exists(dummy_repo_path):
#        os.makedirs(dummy_repo_path)
#        subprocess.run(['git', 'init'], cwd=dummy_repo_path, check=True)
#        logger.info(f"Created and initialized dummy git repo at: {dummy_repo_path}")
#
#    # Test 1: Create a file and commit
#    with open(os.path.join(dummy_repo_path, "test_file1.txt"), "w") as f:
#        f.write("Initial content for git_utils test.")
#    
#    logger.info("Attempting first commit...")
#    success1 = create_git_commit(dummy_repo_path, "Test commit 1: Added test_file1.txt")
#    logger.info(f"First commit success: {success1}")
#
#    # Test 2: No changes, try to commit
#    logger.info("Attempting second commit (no changes)...")
#    success2 = create_git_commit(dummy_repo_path, "Test commit 2: No changes")
#    logger.info(f"Second commit (no changes) success: {success2}")
#
#    # Test 3: Modify the file and commit
#    with open(os.path.join(dummy_repo_path, "test_file1.txt"), "a") as f:
#        f.write("\nAppended content.")
#    
#    logger.info("Attempting third commit (modified file)...")
#    success3 = create_git_commit(dummy_repo_path, "Test commit 3: Modified test_file1.txt")
#    logger.info(f"Third commit success: {success3}")
#
#    # Test 4: Non-git directory
#    non_git_dir = os.path.abspath("./temp_non_git_dir_for_test")
#    if not os.path.exists(non_git_dir):
#        os.makedirs(non_git_dir)
#    logger.info(f"Attempting commit in non-git directory: {non_git_dir}")
#    success4 = create_git_commit(non_git_dir, "Test commit in non-git dir")
#    logger.info(f"Commit in non-git dir success: {success4} (expected False)")
#
#    # Clean up dummy repo (optional)
#    # import shutil
#    # if os.path.exists(dummy_repo_path):
#    #     shutil.rmtree(dummy_repo_path)
#    #     logger.info(f"Cleaned up dummy repo: {dummy_repo_path}")
#    # if os.path.exists(non_git_dir):
#    #     shutil.rmtree(non_git_dir)
#    #     logger.info(f"Cleaned up non-git dir: {non_git_dir}")
#