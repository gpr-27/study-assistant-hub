import subprocess
import webbrowser
import time
import os

import requests
from urllib.parse import urljoin

def kill_process_on_port(port):
    try:
        # Find and kill any process using the port
        subprocess.run(['lsof', '-ti', f':{port}'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(['lsof', '-ti', f':{port}', '|', 'xargs', 'kill', '-9'], shell=True)
        time.sleep(1)  # Give the system time to free up the port
        return True
    except Exception as e:
        print(f"Error killing process on port {port}: {str(e)}")
        return False

def run_streamlit_app(module_path, port):
    print(f"Starting {module_path} on port {port}...")
    
    try:
        # First attempt to start the app
        process = subprocess.Popen(
            ['streamlit', 'run', module_path, '--server.port', str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Give some time for the process to start and capture initial output
        time.sleep(2)
        
        # Check if process failed to start due to port being in use
        if process.poll() is not None:
            _, stderr = process.communicate()
            if "Port is already in use" in stderr:
                print(f"Port {port} is busy, attempting to free it...")
                if kill_process_on_port(port):
                    # Retry starting the app
                    process = subprocess.Popen(
                        ['streamlit', 'run', module_path, '--server.port', str(port)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
        
        # Give some time for the process to start and capture initial output
        time.sleep(2)
        
        # Check for any error output
        if process.poll() is not None:
            _, stderr = process.communicate()
            print(f"Error starting {module_path}:")
            print(stderr)
            return None
            
        # Try to read some output to verify it's working
        stdout_data = process.stdout.readline()
        if stdout_data:
            print(f"✅ {module_path} started successfully")
            print(f"Output: {stdout_data.strip()}")
            return process
        else:
            print(f"⚠️ No output from {module_path}, but process is running")
            return process
            
    except Exception as e:
        print(f"Exception while starting {module_path}: {str(e)}")
        return None

def cleanup_all_ports(ports):
    print("🧹 Cleaning up ports before starting applications...")
    for port in ports:
        kill_process_on_port(port)
    time.sleep(2)  # Give system time to fully release ports

def main():
    # Define the apps and their ports
    apps = [
        ('chat/app.py', 8501),
        ('quiz/quiz_app.py', 8502),
        ('flashcards/flashcards_app.py', 8503),
        ('studyplanner/app.py', 8504)
    ]
    
    # Clean up ports before starting
    ports = [port for _, port in apps]
    cleanup_all_ports(ports)
    
    print("\n🚀 Starting Streamlit applications...\n")
    
    # Start all Streamlit apps
    processes = []
    for app_path, port in apps:
        if os.path.exists(app_path):
            process = run_streamlit_app(app_path, port)
            if process:
                processes.append(process)
        else:
            print(f"❌ {app_path} not found!")
    
    if processes:
        print("\n✨ Applications started successfully!")
        print("💻 Opening landing page...")
        
        # Open the landing page in the default browser
        webbrowser.open('file://' + os.path.abspath('index.html'))
    else:
        print("\n❌ No applications were started successfully!")
        return
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Clean up processes on Ctrl+C
        for process in processes:
            process.terminate()
        print("\nShutting down all apps...")

if __name__ == "__main__":
    main()
