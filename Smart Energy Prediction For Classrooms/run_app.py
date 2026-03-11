#!/usr/bin/env python3
"""
Quick launcher for the Energy Usage Prediction Streamlit App
"""

import subprocess
import sys
import os

def main():
    print("🚀 Starting Energy Usage Prediction App...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("streamlit_app.py"):
        print("❌ Error: streamlit_app.py not found in current directory")
        print("Please run this script from the project directory")
        return
    
    # Check if model exists
    if not os.path.exists("saved_lstm_model_improved"):
        print("❌ Error: Model directory 'saved_lstm_model_improved' not found")
        print("Please ensure you have trained the model first")
        return
    
    try:
        # Run streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {e}")

if __name__ == "__main__":
    main()