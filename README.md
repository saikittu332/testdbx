# GitHub AI Agent

An automated agent that clones repositories, makes improvements, and creates pull requests without human intervention.

## Features

- Automatically clones GitHub repositories
- Makes programmatic changes to code
- Creates branches and pull requests
- Can be scheduled to run periodically

## Setup

1. Create a GitHub Personal Access Token with repository permissions
2. Set up environment variables:
   - `GITHUB_TOKEN`: Your Personal Access Token
   - `REPO_URL`: Target repository in format `owner/repo`

## Usage

### Local Execution

```bash
# Install dependencies
pip install -r requirements.txt

# Run the agent
GITHUB_TOKEN=your_token REPO_URL=owner/repo python -m src.agent