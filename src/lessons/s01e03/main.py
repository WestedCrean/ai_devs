"""
S01E03: Projektowanie API dla efektywnej pracy z modelem

This lesson implements a public HTTP endpoint that acts as an intelligent 
proxy-assistant for a logistics system with conversation memory.
"""

from .server import run_server


def main():
    """Main function to run the S01E03 server"""
    print("Starting S01E03 logistics proxy server...")
    print("Server will be available at http://localhost:3000")
    print("Press Ctrl+C to stop the server")
    
    # Start the FastAPI server
    run_server()


if __name__ == "__main__":
    main()