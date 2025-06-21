class Menu:
    @staticmethod
    def show_menu(door_lock):
        """Show system menu"""
        while True:
            print("\n" + "="*60)
            print("FACE RECOGNITION DOOR LOCK SYSTEM WITH API INTEGRATION")
            print("="*60)
            print("1. Sync Users from API")
            print("2. Add Authorized Face (Manual)")
            print("3. Register Face for API User")
            print("4. List Authorized Users")
            print("5. List API Users")
            print("6. Remove Authorized User")
            print("7. Start Door Lock System")
            print("8. Exit")
            print("="*60)
            
            choice = input("Enter your choice (1-8): ").strip()
            
            if choice == '1':
                door_lock.sync_api_users()
            
            elif choice == '2':
                name = input("Enter name for new authorized user: ").strip()
                if name:
                    door_lock.add_authorized_face(name)
                else:
                    print("Invalid name")
            
            elif choice == '3':
                if door_lock.api_users:
                    print("\nAPI Users needing face registration:")
                    available_users = []
                    for user_id, user_info in door_lock.api_users.items():
                        user_name = user_info['name']
                        if user_name not in door_lock.authorized_faces:
                            available_users.append((user_id, user_name))
                            print(f"{len(available_users)}. {user_name} (ID: {user_id})")
                    
                    if available_users:
                        try:
                            idx = int(input("Enter number to register face for: ")) - 1
                            if 0 <= idx < len(available_users):
                                user_id, user_name = available_users[idx]
                                door_lock.register_api_user_face(user_id)
                            else:
                                print("Invalid selection")
                        except ValueError:
                            print("Invalid input")
                    else:
                        print("All API users already have registered faces")
                else:
                    print("No API users found. Please sync users first.")
            
            elif choice == '4':
                if door_lock.authorized_faces:
                    print("\nAuthorized Users:")
                    for i, name in enumerate(door_lock.authorized_faces.keys(), 1):
                        print(f"{i}. {name}")
                else:
                    print("No authorized users found")
            
            elif choice == '5':
                if door_lock.api_users:
                    print("\nAPI Users:")
                    for user_id, user_info in door_lock.api_users.items():
                        user_name = user_info['name']
                        email = user_info.get('email', 'N/A')
                        phone = user_info.get('telephone', 'N/A')
                        has_face = "✓" if user_name in door_lock.authorized_faces else "✗"
                        print(f"ID: {user_id}")
                        print(f"  Name: {user_name}")
                        print(f"  Email: {email}")
                        print(f"  Phone: {phone}")
                        print(f"  Face Registered: {has_face}")
                        print(f"  Registration Date: {user_info.get('face_registration_date', 'N/A')}")
                        print("-" * 50)
                else:
                    print("No API users found. Please sync users first.")
            
            elif choice == '6':
                if door_lock.authorized_faces:
                    print("\nAuthorized Users:")
                    users = list(door_lock.authorized_faces.keys())
                    for i, name in enumerate(users, 1):
                        print(f"{i}. {name}")
                    
                    try:
                        idx = int(input("Enter number to remove: ")) - 1
                        if 0 <= idx < len(users):
                            removed_user = users[idx]
                            del door_lock.authorized_faces[removed_user]
                            door_lock.save_authorized_faces()
                            print(f"Removed {removed_user} from authorized users")
                        else:
                            print("Invalid selection")
                    except ValueError:
                        print("Invalid input")
                else:
                    print("No authorized users to remove")
            
            elif choice == '7':
                if door_lock.authorized_faces:
                    door_lock.run_door_lock_system()
                else:
                    print("No authorized faces found. Please add authorized users first.")
            
            elif choice == '8':
                break
            
            else:
                print("Invalid choice")