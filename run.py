import sys
import os
import streamlit.web.cli as stcli

def resolve_path(path):
    # When frozen, look in MEIPASS, otherwise use current directory
    base_path = getattr(sys, '_MEIPASS', os.getcwd())
    resolved_path = os.path.join(base_path, path)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Cannot find file: {resolved_path}")
    return resolved_path

def main():
    # PyInstaller runtime configuration
    if getattr(sys, 'frozen', False):
        os.environ['STREAMLIT_RUNNING_IN_PYINSTALLER'] = '1'
        sys.path.append(sys._MEIPASS)
    
    sys.argv = [
        "streamlit",
        "run",
        resolve_path("streamlit_app.py"),
        "--global.developmentMode=false",
        "--server.port=8501",
        "--server.headless=true"
    ]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()