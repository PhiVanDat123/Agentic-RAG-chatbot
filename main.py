import sys
import asyncio
from run_backend import start_backend   # import backend
from frontend import main           # import frontend

# 7. MAIN EXECUTION
# ================================
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "backend":
        # Run backend
        print("Starting FastAPI backend...")
        asyncio.run(start_backend())
    else:
        # Run frontend
        print("Starting Streamlit frontend...")
        main()
