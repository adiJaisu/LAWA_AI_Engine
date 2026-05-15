"""
Main entry point for the FastAPI application.

Includes:
- Application initialization.
- Route inclusion from the router.
- Logging setup.
- Server execution using Uvicorn.

Author: HCLTech
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ai_vms.config.config_manager import config
from ai_vms.routes.router import api_router  # Import the main router
import ai_vms.models
from ai_vms.config.logging_config import LoggingConfig

logger = LoggingConfig().setup_logging()

# Initialize FastAPI application
app = FastAPI(title="AI-VMS Backend", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin (change this for security)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Include all API routes from the router
app.include_router(api_router)

# Health Check API
@app.get("/health", tags=["Health Check"])
def health_check():
    """
    Health check endpoint to verify service availability.
    """
    return {"status": "healthy", "environment": config.ENV}

def start():
    """
    Starts the FastAPI application using Uvicorn.

    This function retrieves configuration values for the host and port and 
    launches the application server.
    """
    logger.info("Initializing the FastAPI AI-VMS Backend Server...")
    uvicorn.run(app, host=config.HOST, port=config.PORT)

# Run the server if executed as a script
if __name__ == "__main__":
    start()
