#!/usr/bin/env python3
"""
ctrlX CORE Data Layer Access - FINAL CORRECT VERSION
Based on official Swagger API documentation
Server: https://{host}/automation/api/v2
"""

import requests
import urllib3
import time
from datetime import datetime, timedelta
import json

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class CtrlXDataLayer:
    """
    ctrlX Data Layer client using the correct API from Swagger docs
    Base URL: https://192.168.1.1/automation/api/v2
    """
    
    def __init__(self, ip, username, password):
        """
        Initialize Data Layer client
        
        Args:
            ip: ctrlX CORE IP address
            username: Login username  
            password: Login password
        """
        self.ip = ip
        self.username = username
        self.password = password
        self.base_url = f"https://{ip}"
        # Use v1 for write commands (v2 for reads works, v1 for writes based on web UI)
        self.api_base_read = f"{self.base_url}/automation/api/v2"
        self.api_base_write = f"{self.base_url}/automation/api/v1"
        self.token = None
        self.token_expiry = None
        
    def get_token(self):
        """Get authentication token"""
        url = f"{self.base_url}/identity-manager/api/v2/auth/token"
        
        payload = {
            "name": self.username,
            "password": self.password
        }
        
        try:
            response = requests.post(url, json=payload, verify=False, timeout=10)
            
            if response.status_code in [200, 201]:
                data = response.json()
                self.token = data.get('access_token')
                expires_in = data.get('expires_in', 3600)
                self.token_expiry = datetime.now() + timedelta(seconds=expires_in - 60)
                print(f"✓ Token obtained (expires in {expires_in}s)")
                return self.token
            else:
                print(f"Token request failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error getting token: {e}")
            return None
    
    def ensure_token(self):
        """Ensure we have a valid token"""
        if not self.token or (self.token_expiry and datetime.now() >= self.token_expiry):
            return self.get_token()
        return self.token
    
    def read_node(self, path):
        """
        Read a Data Layer node using GET /nodes/{Path}
        
        Args:
            path: Node path (e.g., "motion/axs" or "/motion/axs")
            
        Returns:
            Response data (typically contains 'type' and 'value' fields)
        """
        token = self.ensure_token()
        if not token:
            raise Exception("Failed to obtain authentication token")
        
        # Remove leading slash if present
        clean_path = path.lstrip('/')
        
        url = f"{self.api_base_read}/nodes/{clean_path}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json"
        }
        
        response = requests.get(url, headers=headers, verify=False, timeout=5)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Read failed ({response.status_code}): {response.text[:200]}")
    
    def write_node(self, path, value=None, value_type=None):
        """
        Write to a Data Layer node using POST (API v1 style from web UI)
        
        Args:
            path: Node path
            value: Value to write (optional for command triggers)
            value_type: Optional type string
        """
        token = self.ensure_token()
        if not token:
            raise Exception("Failed to obtain authentication token")
        
        # Add leading slash and URL encode the path (as web UI does)
        import urllib.parse
        if not path.startswith('/'):
            path = '/' + path
        encoded_path = urllib.parse.quote(path, safe='')
        
        # Use v1 API with POST method (as seen in web UI)
        url = f"{self.api_base_write}/{encoded_path}?format=json"
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        # Build payload - for commands, often empty or simple structure
        if value is None:
            # For command triggers without value, still need empty object
            payload = {}
        elif value_type:
            # When type is explicitly provided, use exact web UI format
            payload = {"value": value, "type": value_type}
        elif isinstance(value, dict):
            # Already a dictionary, use as-is
            payload = value
        else:
            # Simple value without type
            payload = {"value": value}
        
        # DEBUG: Print what we're sending
        print(f"DEBUG: URL = {url}")
        print(f"DEBUG: Payload = {payload}")
        
        response = requests.post(url, headers=headers, json=payload, verify=False, timeout=5)
        
        if response.status_code not in [200, 201, 204]:
            raise Exception(f"Write failed ({response.status_code}): {response.text[:200]}")


def test_data_layer():
    """Test the Data Layer access with correct API"""
    print("=" * 70)
    print("ctrlX CORE Data Layer Test - Using Swagger-Documented API")
    print("=" * 70)
    
    CTRLX_IP = "192.168.1.1"
    USERNAME = "boschrexroth"
    PASSWORD = "Muenchen81825"
    
    print(f"Target: {CTRLX_IP}")
    print(f"API Base: /automation/api/v2")
    print(f"Endpoint: GET /nodes/{{Path}}")
    print()
    
    dl = CtrlXDataLayer(CTRLX_IP, USERNAME, PASSWORD)
    
    # Get token
    if not dl.get_token():
        print("✗ Failed to get token!")
        return False
    
    print()
    
    # First, try to list all registered nodes
    print("Step 1: Listing registered Data Layer nodes...")
    try:
        data = dl.read_node("datalayer/nodes")
        print(f"✓ Successfully read datalayer/nodes!")
        print(f"Response: {json.dumps(data, indent=2)[:500]}")
        print()
    except Exception as e:
        print(f"⚠ Could not read datalayer/nodes: {e}")
        print()
    
    # Test reading motion axes
    print("Step 2: Reading motion/axs node...")
    try:
        data = dl.read_node("motion/axs")
        print(f"✓ Successfully read motion/axs node!")
        print(f"Full response: {json.dumps(data, indent=2)}")
        print()
        
        # Extract value
        if isinstance(data, dict):
            if 'value' in data:
                axes_value = data['value']
                print(f"Value type: {type(axes_value)}")
                print(f"Value: {axes_value}")
                
                if isinstance(axes_value, list):
                    if axes_value:
                        print(f"\n✓ Found {len(axes_value)} configured axis/axes:")
                        for axis in axes_value:
                            print(f"  - {axis}")
                        print()
                        print("=" * 70)
                        print("✓✓✓ SUCCESS! You can now control your motor!")
                        print("=" * 70)
                        print()
                        print("Next: Use motor_control_final.py with these axis names")
                        return True
                    else:
                        print("\n⚠ No axes configured")
                        print("Please configure axes in Motion app first")
                        return False
                else:
                    print(f"\nValue is not a list: {axes_value}")
            else:
                print(f"No 'value' field in response")
                print(f"Available fields: {data.keys()}")
                
    except Exception as e:
        print(f"✗ Error reading motion/axs: {e}")
        print()
        
        # Try alternative paths
        print("Trying alternative paths...")
        test_paths = [
            "motion",
            "framework",
            "datalayer/nodesrt",  # Realtime nodes
        ]
        
        for test_path in test_paths:
            try:
                print(f"\nTrying: {test_path}")
                data = dl.read_node(test_path)
                print(f"  ✓ Success!")
                print(f"  Response: {json.dumps(data, indent=2)[:300]}")
            except Exception as e2:
                print(f"  ✗ Failed: {str(e2)[:100]}")
    
    return False


if __name__ == "__main__":
    test_data_layer()
