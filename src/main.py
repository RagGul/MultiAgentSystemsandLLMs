import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from src.ui.streamlit_app import app

if __name__ == "__main__":
    app()
