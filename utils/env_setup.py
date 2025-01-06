# pyright: reportMissingImports=false, reportUnusedVariable=warning, reportUntypedBaseClass=error, reportUndefinedVariable=false
import os
import ssl
import torch


def is_colab():
    try:
        from google.colab import drive

        return True
    except:
        return False


def setup_environment(device_type: str, ngrok_token: str = None) -> str:
    """
    Sets up MLflow environment for either GPU (Google Colab) or CPU/MPS (local) environments.

    Args:
        device_type: Type of device ('cuda' or 'cpu')
        ngrok_token: Optional ngrok authentication token for Colab setup

    Returns:
        str: Working directory path

    Raises:
        RuntimeError: If required dependencies or configurations fail
    """
    if is_colab():
        import sys
        from google.colab import drive
        from google.colab import userdata

        drive.mount("/content/drive")
        folder_dir = "/content/drive/MyDrive/data"  # change to your folder structure!
        tracking_uri = folder_dir + "/mlruns"

        try:
            import subprocess

            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "mlflow", "pyngrok"]
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to install required packages: {e}")

        # Start MLflow UI in background
        try:
            os.environ["MLFLOW_TRACKING_URI"] = f"file://{tracking_uri}"
            get_ipython().system_raw(
                f"mlflow ui --port 5000 --backend-store-uri {tracking_uri} &"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to start MLflow UI: {e}")

        # Setup ngrok tunnel
        try:
            from pyngrok import ngrok

            # Use provided token or try to get from userdata
            auth_token = ngrok_token
            if not auth_token:
                try:
                    auth_token = userdata.get("NGROK_TOKEN")
                except (userdata.SecretNotFoundError, userdata.NotebookAccessError):
                    print("Warning: Could not access NGROK_TOKEN from userdata")
                    return folder_dir, tracking_uri

            if not auth_token:
                print("No ngrok authentication token available")
                return folder_dir, tracking_uri

            ngrok.set_auth_token(auth_token)
            public_url = ngrok.connect(5000)
            print(f"MLflow Tracking UI is available at: {public_url}")

        except ImportError:
            print("Warning: pyngrok package not installed")
            return folder_dir, tracking_uri
        except Exception as e:
            raise RuntimeError(f"Failed to setup ngrok tunnel: {e}")

    else:
        # Local environment setup
        folder_dir = os.getcwd()
        tracking_uri = folder_dir + "/mlruns"
        # for downloading pre-trained model locally
        ssl._create_default_https_context = ssl._create_unverified_context

    return folder_dir, tracking_uri


def check_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    return device
