#!/usr/bin/env python3
"""
GitHub AI Agent - Autonomous repository updater

This script:
1. Clones a GitHub repository
2. Makes programmatic changes to the codebase
3. Commits the changes
4. Creates a pull request to the main branch
"""

import os
import sys
import tempfile
import subprocess
from github import Github
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GitHubAIAgent:
    def __init__(self, github_token, repo_url, branch_name="ai-automated-update"):
        """
        Initialize the GitHub AI Agent
        
        Args:
            github_token: Personal Access Token with repo permissions
            repo_url: GitHub repository URL (format: 'owner/repo')
            branch_name: Name for the new branch
        """
        self.github_token = github_token
        self.repo_url = repo_url
        self.branch_name = branch_name
        self.github_client = Github(github_token)
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Working directory: {self.temp_dir}")
        
    def clone_repository(self):
        """Clone the repository to a temporary directory"""
        logger.info(f"Cloning repository: {self.repo_url}")
        clone_url = f"https://{self.github_token}@github.com/{self.repo_url}.git"
        subprocess.run(["git", "clone", clone_url, self.temp_dir], check=True)
        os.chdir(self.temp_dir)
        
    def create_branch(self):
        """Create a new branch for the changes"""
        logger.info(f"Creating branch: {self.branch_name}")
        subprocess.run(["git", "checkout", "-b", self.branch_name], check=True)
        
    def make_changes(self):
        """
        Make changes to the repository files
        Override this method in subclasses to implement specific changes
        """
        # Example: Update dependency version in package.json
        logger.info("Making changes to the repository")
        if os.path.exists("package.json"):
            self._update_package_json()
        elif os.path.exists("requirements.txt"):
            self._update_requirements_txt()
        else:
            # Fallback: Create or update a README file
            self._update_readme()
            
    def _update_package_json(self):
        """Update dependency versions in package.json"""
        import json
        
        with open("package.json", "r") as f:
            package_data = json.load(f)
        
        # Example: Update a dependency to latest version
        if "dependencies" in package_data:
            if "lodash" in package_data["dependencies"]:
                package_data["dependencies"]["lodash"] = "^4.17.21"  # Example update
                
        with open("package.json", "w") as f:
            json.dump(package_data, f, indent=2)
            
    def _update_requirements_txt(self):
        """Update Python dependencies in requirements.txt"""
        with open("requirements.txt", "r") as f:
            requirements = f.readlines()
            
        updated_requirements = []
        for req in requirements:
            # Example: Update requests package to a specific version
            if req.startswith("requests"):
                updated_requirements.append("requests>=2.31.0\n")
            else:
                updated_requirements.append(req)
                
        with open("requirements.txt", "w") as f:
            f.writelines(updated_requirements)
    
    def _update_readme(self):
        """Create or update README.md file"""
        readme_path = "README.md"
        
        if os.path.exists(readme_path):
            with open(readme_path, "r") as f:
                content = f.read()
            
            # Add a badge or note at the top of the README
            updated_content = "[![Automated Updates](https://img.shields.io/badge/Automated-Updates-blue)]()\n\n" + content
            
            with open(readme_path, "w") as f:
                f.write(updated_content)
        else:
            # Create a simple README if none exists
            with open(readme_path, "w") as f:
                f.write("# Repository\n\nThis repository is maintained with automated updates.\n")
    
    def commit_changes(self, commit_message="Automated update by AI agent"):
        """Commit the changes to the new branch"""
        logger.info("Committing changes")
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        
    def push_changes(self):
        """Push the changes to GitHub"""
        logger.info(f"Pushing branch: {self.branch_name}")
        subprocess.run(["git", "push", "-u", "origin", self.branch_name], check=True)
        
    def create_pull_request(self, title=None, body=None):
        """Create a pull request from the new branch to main"""
        if title is None:
            title = "Automated updates by AI agent"
        if body is None:
            body = """
This PR was created automatically by the AI Agent.

Changes include:
- Updated dependencies
- Code formatting improvements
- Documentation updates

Please review the changes and merge if appropriate.
"""
        
        logger.info("Creating pull request")
        repo = self.github_client.get_repo(self.repo_url)
        base_branch = "main"  # or 'master' depending on the repository
        
        try:
            pr = repo.create_pull(
                title=title,
                body=body,
                base=base_branch,
                head=self.branch_name
            )
            logger.info(f"Pull request created successfully: {pr.html_url}")
            return pr.html_url
        except Exception as e:
            logger.error(f"Failed to create pull request: {str(e)}")
            return None
        
    def run(self):
        """Execute the full workflow"""
        try:
            self.clone_repository()
            self.create_branch()
            self.make_changes()
            self.commit_changes()
            self.push_changes()
            pr_url = self.create_pull_request()
            return pr_url
        except Exception as e:
            logger.error(f"Error in AI agent workflow: {str(e)}")
            return None
        finally:
            # Clean up temp directory
            logger.info(f"Cleaning up temporary directory: {self.temp_dir}")
            subprocess.run(["rm", "-rf", self.temp_dir], check=True)


# Example advanced agent class that performs specific code improvements
class CodeImprovementAgent(GitHubAIAgent):
    def make_changes(self):
        """Make specific code improvements"""
        logger.info("Analyzing code and making improvements")
        
        # Look for Python files to improve
        python_files = []
        for root, _, files in os.walk("."):
            for file in files:
                if file.endswith(".py"):
                    python_files.append(os.path.join(root, file))
        
        if python_files:
            self._improve_python_files(python_files)
        else:
            # Fallback to base class behavior
            super().make_changes()
    
    def _improve_python_files(self, python_files):
        """Apply automated improvements to Python files"""
        # Example: Sort imports using isort
        try:
            import isort
            for file_path in python_files:
                logger.info(f"Sorting imports in {file_path}")
                isort.file(file_path)
        except ImportError:
            logger.warning("isort not installed, skipping import sorting")
        
        # Example: Format code using black
        try:
            import black
            for file_path in python_files:
                logger.info(f"Formatting {file_path} with black")
                subprocess.run(["black", file_path], check=False)
        except (ImportError, FileNotFoundError):
            logger.warning("black not installed, skipping code formatting")


def main():
    """Main entry point"""
    # Configuration should come from environment variables for security
    github_token = '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'
    repo_url = 'https://github.com/saikittu332/testdbx'
    
    if not github_token:
        logger.error("GitHub token not provided. Set the GITHUB_TOKEN environment variable.")
        sys.exit(1)
    
    # Create and run the agent
    agent = CodeImprovementAgent(github_token, repo_url)
    pr_url = agent.run()
    
    if pr_url:
        logger.info(f"Success! Pull request created: {pr_url}")
        sys.exit(0)
    else:
        logger.error("Failed to create pull request")
        sys.exit(1)


if __name__ == "__main__":
    main()