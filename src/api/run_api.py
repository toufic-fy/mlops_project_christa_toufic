import uvicorn
import sys

def main():
 # Default values
    app_path = "api.app:app"
    host = "127.0.0.1"
    port = 8000

    # Allow optional host/port arguments
    for arg in sys.argv:
        if arg.startswith("--host="):
            host = arg.split("=")[1]
        elif arg.startswith("--port="):
            port = int(arg.split("=")[1])

    uvicorn.run(app_path, host=host, port=port)

if __name__ == "__main__":
    main()
