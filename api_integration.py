import requests
import json
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image
import cv2

class APIIntegration:
    def __init__(self, api_base_url, api_key=None):
        # API Configuration
        self.api_base_url = api_base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'
        
        # User database
        self.user_database = {}

    def sync_users(self):
        """Fetch users from the API and update local cache"""
        try:
            # Fetch users from API
            response = requests.get(f"{self.api_base_url}/administration/warehouse_users", headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('statusCode') == 200 and 'result' in data:
                    users = data['result']['data']
                    
                    # Update user database
                    for user in users:
                        user_id = user['WAREHOUSE_USER_ID']
                        self.user_database[user_id] = {
                            'id': user_id,
                            'name': f"{user['PRENOM']} {user['NOM']}",
                            'email': user['EMAIL'],
                            'phone': user['TELEPHONE'],
                            'photo': user['PHOTO'],
                            'profile': user.get('profil', {}).get('DESCRIPTION_PROFIL', 'USER'),
                            'date_save': user['DATE_SAVE']
                        }
                    
                    print(f"Synced {len(users)} users from API")
                    return True
                else:
                    print(f"API Error: {data.get('message', 'Unknown error')}")
            else:
                print(f"Failed to fetch users: HTTP {response.status_code}")
                
        except requests.RequestException as e:
            print(f"Network error while syncing users: {e}")
        except Exception as e:
            print(f"Error syncing users: {e}")
        
        return False

    def log_access(self, user_id, image_frame, status=1):
        """Log access attempt to the API"""
        try:
            # Convert frame to JPEG image bytes
            _, buffer = cv2.imencode('.jpg', image_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            # Prepare multipart form data
            files = {
                'IMAGE': ('access_image.jpg', buffer.tobytes(), 'image/jpeg')
            }
            
            # Prepare form data (non-image fields)
            data = {
                'WAREHOUSE_USER_ID': str(user_id),
                'STATUT': str(status),  # 1 for success, 0 for failure
                'DATE_SAVE': datetime.now().isoformat()
            }
            
            # Prepare headers for multipart upload (remove Content-Type to let requests set it)
            upload_headers = self.headers.copy()
            if 'Content-Type' in upload_headers:
                del upload_headers['Content-Type']
            
            # Send to API with multipart form data
            response = requests.post(
                f"{self.api_base_url}/warehouse_acces/create", 
                files=files,
                data=data,
                headers=upload_headers
            )
            
            if response.status_code == 200:
                print("Access logged to API successfully")
                return True
            else:
                print(f"Failed to log access: HTTP {response.status_code}")
                if response.text:
                    print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"Error logging access to API: {e}")
        
        return False

    def get_user_info(self, user_id):
        """Get user information from local cache"""
        return self.user_database.get(user_id)