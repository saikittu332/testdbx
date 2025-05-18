#!/usr/bin/env python3
"""
Test GitHub token authentication
"""

import logging
import os
import sys

from github import Github

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_token(token, repo_url=None):
    """Test if a GitHub token is valid"""
    try:
        logger.info(f"Testing GitHub token (length: {len(token)})")
        g = Github(token)

        # Test basic authentication
        user = g.get_user()
        logger.info(f"✓ Authentication successful as: {user.login}")

        # Test repo access if provided
        if repo_url:
            logger.info(f"Testing access to repository: {repo_url}")
            repo = g.get_repo(repo_url)
            logger.info(f"✓ Repository access successful: {repo.full_name}")

            # Test listing branches
            branches = list(repo.get_branches())
            logger.info(f"✓ Retrieved {len(branches)} branches")
            for branch in branches:
                logger.info(f"  - {branch.name}")

        return True
    except Exception as e:
        logger.error(f"✗ Token validation failed: {str(e)}")
        return False


def main():
    # Try to read token from .env file first
    token = "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"

    # If not found in .env, try environment variable
    if not token:
        token = os.environ.get("GITHUB_TOKEN")

    # If still not found, prompt user
    if not token:
        token = input("Enter your GitHub Personal Access Token: ")

    # Test the token
    repo_url = input(
        "Enter repository (owner/repo) to test access to (press Enter to skip): "
    )
    if not repo_url.strip():
        repo_url = None

    if test_token(token, repo_url):
        logger.info("✓ Token validation successful!")
    else:
        logger.error("✗ Token validation failed")


if __name__ == "__main__":
    main()
