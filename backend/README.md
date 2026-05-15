# AI-VMS Backend

## Overview
AI-VMS Backend is a FastAPI-based backend service designed for scalable and high-performance applications. It controls all the functionalities and triggers all the functions.It also includes structured configurations, database integration, and versioned API routes.

## Features
- FastAPI framework for building APIs
- Structured configuration management
- Database integration using SQLAlchemy
- Versioned API routing
- Docker support for containerized deployment

## Folder Structure
```
BACKEND/
│-- config/                # Configuration files (development, production, testing)
│   │-- config_manager.py  # Manages configuration settings
│   │-- development.py     # Development environment settings
│   │-- production.py      # Production environment settings
│   │-- testing.py         # Testing environment settings
|   |-- logging_config.py  # Logger Configuration and setup
│-- src/                   # Application source code
│   │-- constant/          # Application Constants
│   │-- database/          # Database setup and connection
│   │-- models/            # Database models
│   │-- routes/            # API routes (organized by version)
│   │-- schemas/           # API Request/Response Schemas
│   │-- services/          # Business logic and service layer
│   │-- utils/             # Utility functions
│-- tests/                 # Unit and integration tests
│-- Dockerfile             # Docker setup for deployment
│-- main.py                # Application entry point
│-- README.md              # Documentation
│-- requirements.txt       # Dependencies and packages
```

## Getting Started
### Prerequisites
- Python 3.8+
- MySQL (or any preferred database)
- Docker (optional for containerized deployment)

### Running the Application
#### Using Uvicorn
```bash
uvicorn main:app --host 0.0.0.0 --port 8010 --reload
```

## Environment Configuration
Set the `APP_ENV` variable to specify the environment:
```bash
export APP_ENV=development  # Options: development, production, testing
```

## API Routes
The application follows versioned API routing:
```
/src/routes/
   ├── v1/
   │   ├── users.py  # User-related endpoints
   │   ├── router.py  # v1 Router
   ├── router.py  # Main API router
```

## Testing
Run tests using:
```bash
pytest tests/
```

## Contributing
1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Open a Pull Request

