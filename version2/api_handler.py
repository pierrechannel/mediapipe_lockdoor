import requests
import json
import cv2
import numpy as np

class APIHandler:
    def __init__(self, api_base_url=None, headers=None):
        self.api_base_url = api_base_url
        self.headers = headers or {'Content-Type': 'application/json'}
    
    def get_system_mode_users(self):
        """Get registration requests with proper response handling
        
        Returns:
            dict: The API response data if successful, None otherwise
            (including when no data is found)
        """
        if not self.api_base_url:
            print("API base URL not configured")
            return None
            
        try:
            response = requests.get(
                f"{self.api_base_url}/administration/warehouse_users/checkmode/allregistration",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
                      
            # Validate response structure is a dictionary
            if not isinstance(data, dict):
                raise ValueError("Invalid API response format - expected a dictionary")
                         
            # Handle case when no data is found
            if data.get("message") == "Aucune donnée trouvée" and data.get("result") is None:
                print("No registration data found")
                return None
                         
            # Check status code in response
            if data.get('statusCode') != 200:
                error_msg = data.get('message', 'No error message provided')
                print(f"API Error: {error_msg}")
                return None
                         
            # Additional validation - check if expected data fields are present
            if 'result' not in data:
                print("API response missing expected 'result' field")
                return None
                         
            return data
                     
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {str(e)}")
            return None
        except ValueError as e:
            print(f"Invalid JSON response: {str(e)}")
            return None
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return None
   
    def log_access(self, user_id, image_frame, status=1, max_retries=3):
        """Log access attempt to the API with proper multipart form-data handling"""
        url = f"{self.api_base_url}/warehouse_acces/create" if self.api_base_url else "https://apps.mediabox.bi:26875/warehouse_acces/create"

        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt + 1} to log access for user {user_id}")

                # Convert image to JPEG with quality 85%
                success, buffer = cv2.imencode('.jpg', image_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not success:
                    print("Failed to encode image")
                    continue

                # Check and handle image size (max 2MB)
                image_bytes = buffer.tobytes()
                if len(image_bytes) > 2000000:
                    # Resize image to reduce size while maintaining aspect ratio
                    scale_factor = (3000000 / len(image_bytes)) ** 0.5
                    small_frame = cv2.resize(image_frame, None, fx=scale_factor, fy=scale_factor)
                    success, buffer = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not success:
                        print("Failed to encode resized image")
                        continue
                    image_bytes = buffer.tobytes()

                # Prepare the multipart form data
                files = {
                    'IMAGE': ('access.jpg', image_bytes, 'image/jpeg')
                }
                
                # Prepare form data with proper values
                data = {
                    'WAREHOUSE_USER_ID': str(user_id) if user_id else '0',
                    'STATUT': str(status)
                }

                # Make sure we're not sending JSON headers for form-data
                headers = {k: v for k, v in self.headers.items() if k.lower() != 'content-type'}

                # Debug output
                print(f"Sending to {url}")
                print(f"Data: {data}")
                print(f"Image size: {len(image_bytes)} bytes")

                # Send the request
                response = requests.post(
                    url,
                    data=data,
                    files=files,
                    headers=headers,
                    timeout=10
                )

                # Handle response
                print(f"Response status: {response.status_code}")
                
                if response.status_code == 200:
                    try:
                        response_data = response.json()
                        print("Success:", response_data.get('message', 'No message'))
                        if response_data.get('notification', {}).get('sent'):
                            print("Notification sent to clients")
                        return True
                    except ValueError:
                        print("Success (non-JSON response)")
                        return True
                
                elif response.status_code == 422:
                    try:
                        error_data = response.json()
                        print("Validation errors:", error_data.get('data', {}))
                    except ValueError:
                        print("Validation failed (no details)")
                    return False
                
                elif response.status_code == 400:
                    print("Bad request:", response.text[:500])
                    return False
                
                else:
                    print(f"Server error: {response.status_code} - {response.text[:500]}")
                    continue

            except requests.exceptions.Timeout:
                print("Request timed out")
                continue
            except requests.exceptions.RequestException as e:
                print(f"Network error: {str(e)}")
                continue
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                continue

        print("All attempts failed")
        return False
    
    def send_registration_status(self, user_id, status, face_encoding_data=None):
        """Send registration status to the API with proper error handling"""
        try:
            # Prepare the data payload according to API requirements
            data = {
                'WAREHOUSE_USER_ID': int(user_id),  # Ensure it's sent as number
                'REGISTRATION_STATUS': int(status),  # 1 = success, 2 = failure
                # DATE_SAVE will be handled server-side
            }
            
            # Include face encoding data if registration was successful
            if status == 1 and face_encoding_data is not None:
                if isinstance(face_encoding_data, (list, np.ndarray)):
                    # Convert numpy array to list if needed
                    if isinstance(face_encoding_data, np.ndarray):
                        face_encoding_data = face_encoding_data.tolist()
                    data['FACE_ENCODING_DATA'] = face_encoding_data
                else:
                    print("Warning: Invalid face encoding data format")
            
            # Make the API request
            response = requests.post(
                f"{self.api_base_url}/administration/warehouse_users/face/registration-status",
                json=data,
                headers=self.headers,
                timeout=10  # Add timeout to prevent hanging
            )
            
            # Process the response
            if response.status_code == 200:
                response_data = response.json()
                if response_data.get('statusCode') == 200:
                    status_text = "réussi" if status == 1 else "échec"
                    user_name = response_data.get('result', {}).get('USER_NAME', '')
                    print(f"Statut d'enregistrement {status_text} pour {user_name} (ID: {user_id})")
                    
                    # Additional success handling
                    if status == 1:
                        print("Face registration completed successfully")
                        if 'FACE_ENCODING_DATA' in data:
                            print("Face encoding data was sent to server")
                    return True
                else:
                    error_msg = response_data.get('message', 'Unknown error')
                    print(f"API returned error: {error_msg}")
                    return False
            elif response.status_code == 422:
                # Handle validation errors
                try:
                    error_data = response.json()
                    print("Validation failed:")
                    for field, errors in error_data.get('data', {}).items():
                        print(f" - {field}: {', '.join(errors)}")
                except ValueError:
                    print("Validation failed (no details)")
                return False
            elif response.status_code == 404:
                print("Error: User not found on server")
                return False
            else:
                print(f"API request failed with status {response.status_code}")
                try:
                    error_details = response.json()
                    print(f"Error details: {error_details}")
                except ValueError:
                    print(f"Raw response: {response.text[:500]}...")
                return False
                
        except requests.exceptions.Timeout:
            print("Error: Request timed out after 10 seconds")
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to the server")
        except requests.exceptions.RequestException as e:
            print(f"Network error: {str(e)}")
        except ValueError as e:
            print(f"Error parsing API response: {str(e)}")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
        
        return False