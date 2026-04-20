# Tailgate Detection Microservice

This microservice provides automated tracking and detection of tailgating violations using video analytics. It is built on top of the `Ai_engine` and communicates via RabbitMQ.

## Folder Structure
- `app.py`: Main entry point for the service.
- `Dockerfile`: Container configuration for Deployment.
- `requirements.txt`: Python package dependencies.
- `ARCHITECTURE.md`: Technical documentation.

## Key Features
- **LED-Integrated Access**: Monitors badge reader LEDs to distinguish authorized entries from tailgaters.
- **Punch/Token Logic**: Handles multiple entries during a single authorized window.
- **Database Backend**: Uses SQLite for employee records and access logging.
- **Microservice Architecture**: Decoupled from the main stream handler for horizontal scaling.

## Data Flow
1. **Input**: Receiving frame batches via `queue_tailgate_detection`.
2. **Processing**: `Ai_engine` processes the batch using the `TailgateDetection` business case.
3. **Alerting**: Violation events are triggered if unauthorized entries are detected.
4. **Output**: Results sent back to the Event Manager.
