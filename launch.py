import os
import subprocess
import sys

port = int(os.environ.get("PORT", 8080))

# Remove env vars that would override our --server.port flag
for key in ("STREAMLIT_SERVER_PORT",):
    os.environ.pop(key, None)

os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_SERVER_ENABLE_CORS"] = "false"
os.environ["STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION"] = "false"

subprocess.run(
    [
        sys.executable, "-m", "streamlit", "run",
        "pages/Stock_Predictor.py",
        "--server.port", str(port),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
    ],
    check=True,
)
