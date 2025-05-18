import requests
import sys

def test_token(token):
    """Simple test of a GitHub token using direct API calls"""
    
    # Remove any whitespace
    token = token.strip()
    
    print(f"Testing token (length: {len(token)})")
    
    # Test basic authentication with GitHub API
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    
    # Get authenticated user information
    response = requests.get('https://api.github.com/user', headers=headers)
    
    print(f"Status code: {response.status_code}")
    
    if response.status_code == 200:
        user_data = response.json()
        print(f"Authenticated as: {user_data['login']}")
        print("✓ Token is VALID")
        return True
    else:
        print(f"✗ Authentication failed: {response.json().get('message', 'Unknown error')}")
        
        if response.status_code == 401:
            print("\nPossible issues:")
            print("1. Token is incorrect or has expired")
            print("2. Token has been revoked")
            print("3. Token is malformed (check for extra characters)")
        
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Token provided as command line argument
        token = sys.argv[1]
    else:
        # Prompt for token
        token = input("Enter your GitHub Personal Access Token: ")
    
    test_token(token)
