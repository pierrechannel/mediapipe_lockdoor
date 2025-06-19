import requests
import json
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image
import cv2
import time
import threading


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
        
        # Registration monitoring
        self.monitoring_registration = False

    def sync_users(self):
        """Fetch users from the API and update local cache"""
        try:
            # Updated endpoint to match your routes
            response = requests.get(f"{self.api_base_url}/administration/warehouse_users/", headers=self.headers)
            
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

    def check_registration_requests(self, user_id=None):
        """Check for registration requests (MODE = 2)"""
        try:
            # Updated endpoint to match your routes
            if user_id:
                response = requests.get(
                    f"{self.api_base_url}/administration/warehouse_users/face/registration-requests/{user_id}", 
                    headers=self.headers
                )
            else:
                # If no specific user_id, you might need to implement a different endpoint
                # or modify the backend to support getting all registration requests
                response = requests.get(
                    f"{self.api_base_url}/administration/warehouse_users/face/registration-requests/all", 
                    headers=self.headers
                )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('statusCode') == 200 and 'result' in data:
                    registration_requests = data['result']
                    
                    # Return registration requests where MODE = 2
                    if isinstance(registration_requests, list):
                        return [req for req in registration_requests if req.get('MODE') == 2]
                    elif isinstance(registration_requests, dict) and registration_requests.get('MODE') == 2:
                        return [registration_requests]
                    
        except requests.RequestException as e:
            print(f"Network error while checking registration requests: {e}")
        except Exception as e:
            print(f"Error checking registration requests: {e}")
        
        return []

    def send_registration_status(self, user_id, status):
        """Send registration status to the API"""
        try:
            data = {
                'WAREHOUSE_USER_ID': str(user_id),
                'REGISTRATION_STATUS': status,  # 1 = success, 2 = failure
                'DATE_SAVE': datetime.now().isoformat()
            }
            
            # Updated endpoint to match your routes
            response = requests.post(
                f"{self.api_base_url}/administration/warehouse_users/face/registration-status", 
                json=data,
                headers=self.headers
            )
            
            if response.status_code == 200:
                status_text = "réussi" if status == 1 else "échec"
                print(f"Statut d'enregistrement envoyé: {status_text} pour l'utilisateur {user_id}")
                return True
            else:
                print(f"Échec de l'envoi du statut d'enregistrement: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"Erreur lors de l'envoi du statut d'enregistrement: {e}")
        
        return False

    def request_face_registration(self, user_id):
        """Request facial registration for a user"""
        try:
            data = {
                'WAREHOUSE_USER_ID': str(user_id),
                'DATE_SAVE': datetime.now().isoformat()
            }
            
            # Updated endpoint to match your routes
            response = requests.post(
                f"{self.api_base_url}/administration/warehouse_users/face/request-registration", 
                json=data,
                headers=self.headers
            )
            
            if response.status_code == 200:
                print(f"Demande d'enregistrement facial envoyée pour l'utilisateur {user_id}")
                return True
            else:
                print(f"Échec de la demande d'enregistrement: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"Erreur lors de la demande d'enregistrement: {e}")
        
        return False

    def cancel_registration_request(self, user_id):
        """Cancel a pending facial registration request"""
        try:
            data = {
                'WAREHOUSE_USER_ID': str(user_id),
                'DATE_SAVE': datetime.now().isoformat()
            }
            
            # Updated endpoint to match your routes
            response = requests.post(
                f"{self.api_base_url}/administration/warehouse_users/face/cancel-registration", 
                json=data,
                headers=self.headers
            )
            
            if response.status_code == 200:
                print(f"Demande d'enregistrement annulée pour l'utilisateur {user_id}")
                return True
            else:
                print(f"Échec de l'annulation: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"Erreur lors de l'annulation: {e}")
        
        return False

    def change_system_mode(self, mode):
        """Change system mode (1: normal, 2: registration)"""
        try:
            data = {
                'MODE': mode,
                'DATE_SAVE': datetime.now().isoformat()
            }
            
            # Updated endpoint to match your routes
            response = requests.post(
                f"{self.api_base_url}/administration/warehouse_users/face/change-mode", 
                json=data,
                headers=self.headers
            )
            
            if response.status_code == 200:
                mode_text = "normal" if mode == 1 else "enregistrement"
                print(f"Mode système changé à: {mode_text}")
                return True
            else:
                print(f"Échec du changement de mode: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"Erreur lors du changement de mode: {e}")
        
        return False
    
    def get_system_mode(self):
        """Get current system mode (1: normal, 2: registration)"""
        try:
            # Endpoint pour récupérer le mode actuel
            response = requests.get(
                f"{self.api_base_url}/administration/warehouse_users/face/get-mode",
                headers=self.headers
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Supposons que l'API retourne {'mode': 1} ou {'mode': 2}
                current_mode = data.get('mode', 1)  # Par défaut mode normal
                mode_text = "normal" if current_mode == 1 else "enregistrement"
                
                print(f"Mode système actuel: {mode_text} (code: {current_mode})")
                return current_mode
                
            else:
                print(f"Échec de récupération du mode: HTTP {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Erreur de connexion lors de la récupération du mode: {e}")
            return None
        except Exception as e:
            print(f"Erreur lors de la récupération du mode: {e}")
            return None
    
    def get_user_for_face_system(self, user_id):
        """Get user information for facial recognition system"""
        try:
            # Updated endpoint to match your routes
            response = requests.get(
                f"{self.api_base_url}/administration/warehouse_users/face/user/{user_id}", 
                headers=self.headers
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('statusCode') == 200 and 'result' in data:
                    return data['result']
                    
        except Exception as e:
            print(f"Erreur lors de la récupération de l'utilisateur: {e}")
        
        return None

    def log_access(self, user_id, image_frame, status=1, max_retries=3):
        """Log access attempt to the API with retry logic"""

        url = "http://127.0.0.1:5000/warehouse_acces/create"  # 🔁 Remplacez ceci par l’URL réelle de votre API

        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt + 1} to log access for user {user_id}")

                # Convertir l'image en JPEG
                success, buffer = cv2.imencode('.jpg', image_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not success:
                    print("Failed to encode image")
                    return False

                image_bytes = buffer.tobytes()
                image_size = len(image_bytes)
                print(f"Image size: {image_size} bytes")

                # Préparer les données
                files = {
                    'IMAGE': ('access.jpg', image_bytes, 'image/jpeg')
                }
                data = {
                    'WAREHOUSE_USER_ID': str(user_id),
                    'STATUT': str(status),
                    'DATE_SAVE': datetime.now().isoformat()
                }

                print(f"Sending POST to {url}")
                print("DATA:", data)
                print("FILES: IMAGE (access.jpg)", f"{image_size} bytes")

                # Envoyer la requête
                response = requests.post(url, data=data, files=files, timeout=5)

                print("Status Code:", response.status_code)
                print("Response content (raw):", repr(response.content))

                if response.status_code == 200:
                    try:
                        result = response.json()
                        print("Access log success (JSON):", result)
                        return True
                    except ValueError as ve:
                        print("Erreur JSON (mais status 200):", ve)
                        print("Texte brut de la réponse:", response.text)
                        print("Accepté sans JSON.")
                        return True
                else:
                    print(f"HTTP error {response.status_code}: {response.text}")
                    continue

            except Exception as e:
                print("Request failed:", str(e))
                continue

        print("All attempts failed")
        return False

    def get_user_info(self, user_id):
        """Get user information from local cache"""
        return self.user_database.get(user_id)

    def start_registration_monitoring(self, face_recognition_instance, check_interval=5):
        """Start monitoring registration requests"""
        def monitor_loop():
            while self.monitoring_registration:
                try:
                    # Check for registration requests
                    registration_requests = self.check_registration_requests()
                    
                    for request in registration_requests:
                        user_id = request.get('WAREHOUSE_USER_ID')
                        if user_id and not face_recognition_instance.registration_mode:
                            print(f"Demande d'enregistrement reçue pour l'utilisateur {user_id}")
                            
                            # Start registration mode
                            if face_recognition_instance.start_registration_mode(user_id):
                                # Wait for registration to complete
                                timeout = 30  # 30 seconds timeout
                                start_time = time.time()
                                
                                while (face_recognition_instance.registration_mode and 
                                       time.time() - start_time < timeout):
                                    time.sleep(0.1)
                                
                                # Check result
                                if not face_recognition_instance.registration_mode:
                                    # Registration completed successfully
                                    self.send_registration_status(user_id, 1)
                                else:
                                    # Timeout - cancel registration
                                    face_recognition_instance.cancel_registration()
                                    self.send_registration_status(user_id, 2)
                    
                    time.sleep(check_interval)
                    
                except Exception as e:
                    print(f"Erreur dans la surveillance d'enregistrement: {e}")
                    time.sleep(check_interval)
        
        self.monitoring_registration = True
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        print("Surveillance des demandes d'enregistrement démarrée")

    def stop_registration_monitoring(self):
        """Stop registration monitoring"""
        self.monitoring_registration = False
        print("Surveillance des demandes d'enregistrement arrêtée")

    def get_single_user(self, user_id):
        """Get a single user by ID"""
        try:
            response = requests.get(f"{self.api_base_url}/administration/warehouse_users/{user_id}", headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('statusCode') == 200 and 'result' in data:
                    return data['result']
                    
        except Exception as e:
            print(f"Erreur lors de la récupération de l'utilisateur {user_id}: {e}")
        
        return None

    def update_user(self, user_id, user_data, photo_path=None):
        """Update a user by ID"""
        try:
            # Prepare headers for multipart upload if photo is included
            upload_headers = self.headers.copy()
            if 'Content-Type' in upload_headers:
                del upload_headers['Content-Type']
            
            files = {}
            if photo_path:
                files['PHOTO'] = open(photo_path, 'rb')
            
            response = requests.put(
                f"{self.api_base_url}/administration/warehouse_users/{user_id}",
                data=user_data,
                files=files if files else None,
                headers=upload_headers
            )
            
            if files:
                files['PHOTO'].close()
            
            if response.status_code == 200:
                print(f"Utilisateur {user_id} mis à jour avec succès")
                return True
            else:
                print(f"Échec de la mise à jour: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"Erreur lors de la mise à jour de l'utilisateur: {e}")
        
        return False

    def delete_users(self, user_ids):
        """Delete multiple users by IDs"""
        try:
            data = {'USER_IDS': user_ids}
            
            response = requests.delete(
                f"{self.api_base_url}/api/users",
                json=data,
                headers=self.headers
            )
            
            if response.status_code == 200:
                print(f"Utilisateurs supprimés avec succès: {user_ids}")
                return True
            else:
                print(f"Échec de la suppression: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"Erreur lors de la suppression des utilisateurs: {e}")
        
        return False