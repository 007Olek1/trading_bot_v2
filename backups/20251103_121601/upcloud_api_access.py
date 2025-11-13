#!/usr/bin/env python3
"""
Скрипт для восстановления доступа к серверу через UpCloud API
"""

import requests
import json
import base64

# API credentials
API_TOKEN = "ucat_01K835A57Y941XN3SM3PSV32B2"
BASE_URL = "https://api.upcloud.com/1.3"
SERVER_IP = "5.22.220.105"
SSH_PUBLIC_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAINvt5Nqq602AgsaNjArzniwkEY2uPRXBnqMKLSABZn+I trading_bot_upcloud"

# Базовая аутентификация (API token как username, пустой password)
auth = (API_TOKEN, "")

def get_servers():
    """Получить список всех серверов"""
    url = f"{BASE_URL}/server"
    response = requests.get(url, auth=auth)
    return response.json()

def find_server_by_ip(ip):
    """Найти сервер по IP адресу"""
    servers = get_servers()
    for server in servers.get("servers", {}).get("server", []):
        for ip_addr in server.get("ip_addresses", {}).get("ip_address", []):
            if ip_addr.get("address") == ip:
                return server.get("uuid")
    return None

def get_server_info(server_uuid):
    """Получить информацию о сервере"""
    url = f"{BASE_URL}/server/{server_uuid}"
    response = requests.get(url, auth=auth)
    return response.json()

def get_server_details(server_uuid):
    """Получить детальную информацию о сервере"""
    url = f"{BASE_URL}/server/{server_uuid}"
    response = requests.get(url, auth=auth)
    return response.json()

def create_file_on_server(server_uuid, file_path, file_content):
    """Создать файл на сервере через API (если поддерживается)"""
    # UpCloud API не поддерживает прямой доступ к файловой системе
    # Но можно использовать cloud-init или SSH через консоль
    print(f"⚠️ UpCloud API не поддерживает прямую запись файлов")
    print(f"📋 Создайте файл через Console:")
    print(f"   echo '{file_content}' > {file_path}")

def main():
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║     🔧 ВОССТАНОВЛЕНИЕ ДОСТУПА ЧЕРЕЗ UPLOUD API               ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print("")
    
    print(f"🔍 Поиск сервера {SERVER_IP}...")
    try:
        server_uuid = find_server_by_ip(SERVER_IP)
        
        if not server_uuid:
            print(f"❌ Сервер с IP {SERVER_IP} не найден!")
            print("📋 Получаю список всех серверов...")
            servers = get_servers()
            print(json.dumps(servers, indent=2))
            return
        
        print(f"✅ Сервер найден! UUID: {server_uuid}")
        print("")
        
        print("📊 Информация о сервере:")
        server_info = get_server_details(server_uuid)
        print(json.dumps(server_info, indent=2))
        print("")
        
        print("🔑 Настройка SSH доступа:")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("⚠️ UpCloud API не поддерживает прямую настройку SSH ключей")
        print("📋 Воспользуйтесь одним из способов:")
        print("")
        print("СПОСОБ 1: Через UpCloud Console (VNC)")
        print("   1. Зайдите в https://hub.upcloud.com/")
        print("   2. Откройте сервер и нажмите 'Console'")
        print("   3. Выполните команды:")
        print("")
        print("   mkdir -p ~/.ssh && chmod 700 ~/.ssh")
        print(f"   echo '{SSH_PUBLIC_KEY}' >> ~/.ssh/authorized_keys")
        print("   chmod 600 ~/.ssh/authorized_keys")
        print("")
        print("СПОСОБ 2: Через SSH с паролем (если включен)")
        print("   ssh root@5.22.220.105")
        print("   # Введите пароль из UpCloud панели")
        print("")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


