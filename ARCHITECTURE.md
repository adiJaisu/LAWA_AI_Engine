# LAWA AI Engine: Pipeline Architecture Documentation

## 1. Executive Summary
The LAWA AI Engine is a high-performance vision processing pipeline designed for real-time video analytics. It utilizes a decoupled, microservices-oriented architecture to ensure scalability, low latency (via Shared Memory), and modularity (via RabbitMQ RPC).

---

## 2. Pipeline Flow Diagram

## 2. Comprehensive Pipeline Flowchart

```mermaid
flowchart TD
    %% Main Pipeline Nodes
    SH["Stream Handler / Video Decoder<br/><i>(RTSP / GStreamer / FFmpeg)</i>"]
    UQ{"Usecase Request Queue<br/><i>(e.g., loitering_requests)</i>"}
    
    subgraph "AI Engine (Consumer Worker)"
        direction TB
        AE_V["Validation & Preprocessing"]
        SHM[("(Shared Memory<br/>/dev/shm)")]
        RPC["RabbitMQ RPC Client<br/><i>(BlockingConnection + Exclusive Callback)</i>"]
    end

    IQ["Inference Request Queue<br/><i>(e.g., ai_inference_detection_queue)</i>"]
    
    subgraph "Inference Service"
        IS["Inference Engine<br/><i>(YOLOv8 Predictor)</i>"]
        weights[("(Model Weights<br/>.pt / .onnx)")]
    end

    BL["Business Case Executer<br/><i>(ByteTrack + ROI pointPolygonTest)</i>"]
    EQ{"Usecase Event Queue<br/><i>(e.g., loitering_events)</i>"}
    
    subgraph "Event Manager (Evidence Node)"
        EM["Event Consumer"]
        VP["Video Evidence Processor<br/><b>10 Frames @ 1 FPS -> 10s Video</b>"]
    end

    DB[("PostgreSQL Database<br/><i>(Path + Meta)</i>")]
    WS["WebSocket Service<br/><i>(Real-time Broadcast)</i>"]
    FE["Frontend Dashboard<br/><i>(User Alert)</i>"]

    %% Connections
    SH -->|Publishes Frame Data| UQ
    UQ -->|Consumes Batch| AE_V
    AE_V -->|Saves .raw to| SHM
    AE_V -->|Triggers| RPC
    RPC -->|RPC Call: Path + Shape| IQ
    IQ -->|Loads weights| IS
    IS --- weights
    IS -->|Reads Frame Data| SHM
    IS -->|Returns JSON Results| RPC
    RPC -->|Detection Results| BL
    BL -->|If Alert Triggered:<br/>Publishes 10 Frames Batch| EQ
    EQ --> EM
    EM --> VP
    VP -->|Stores video_path| DB
    VP -->|Broadcast Notify| WS
    WS --> FE

    %% New Usecase Requirements (Side Boxes)
    subgraph "NEW USECASE CHECKLIST"
        direction TB
        N1["<b>1. Decoder Setup</b><br/>Register new target queue name<br/>in Decoder Publisher config."]
        N2["<b>2. RMQ Infrastructure</b><br/>Create Request & Event<br/>queues in RabbitMQ."]
        N3["<b>3. Logic Development</b><br/>- Create business_cases/NewUsecase.py<br/>- Create executers/NewUsecase_executer.py<br/>- Update main.py dispatch switch."]
        N4["<b>4. Inference Integration</b><br/>- Use existing Detection/Seg client<br/>- OR create new Client for new model."]
        N5["<b>5. Event Manager</b><br/>Add Event Queue name to<br/>self.event_queues list."]
    end

    %% Linking Requirements to Flow
    N1 -.-> SH
    N2 -.-> UQ
    N2 -.-> EQ
    N3 -.-> BL
    N5 -.-> EM

    %% Styling
    style N1 fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#000
    style N2 fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#000
    style N3 fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#000
    style N4 fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#000
    style N5 fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#000
    style SHM fill:#eee,stroke:#333,stroke-dasharray: 5 5
    style VP fill:#fff4dd,stroke:#d4a017,stroke-width:2px,color:#000
```

---

## 3. Detailed Component Interaction

### A. Stream Management
The **Stream Handler** (Video Decoder) serves as the source. It decodes RTSP streams and distributes frames into specific usecase queues. This allows multiple AI models to process the same stream independently without redundant decoding.

### B. The RPC & Shared Memory Technique
To avoid the overhead of sending large binary image data over RabbitMQ, the **AI Engine** employs a hybrid approach:
1. **Shared Memory**: The frame is written to `/dev/shm` (a RAM-based filesystem).
2. **RPC Payload**: Only the path to the frame and necessary metadata (shape, classes) are sent to the inference service.
3. **Synchronization**: The engine waits for the model's response using a unique `correlation_id` on a temporary callback queue.

### C. Business Case Logic
Once results (bounding boxes, track IDs, masks) are received, the **Executers** apply spatial and temporal rules. If an event is triggered, the engine gathers the required frames for evidence and publishes them to the Event Manager.

### D. Evidence Generation & Notification
When an event is confirmed, a batch of 10 frames is sent to the **Event Manager**. 
- **Evidence Video**: It generates a 10-second video at 1 FPS.
- **Database Persistence**: Camera details, event details, and the video path are stored in PostgreSQL.
- **WebSocket Broadcast**: A real-time notification is sent to the frontend for immediate alerting.

---

## 4. Extension Guidelines ("Box Beside the Box")

### Adding a New Usecase
> [!NOTE]
> **Checklist for adding a new usecase:**
> 1. **Queues**: Define `usecase_requests` and `usecase_events` in RabbitMQ.
> 2. **Logic**: Implement a new class in `Ai_engine/src/business_cases/`.
> 3. **Executer**: Create a wrapper in `Ai_engine/src/executers/`.
> 4. **Routing**: Add the usecase name to the `dispatch_vision_task` in `main.py`.
> 5. **EventManager**: Add the new event queue to the listener list in `EventManager/main.py`.

### Changing the Model Type
| Scenario | Action Required |
| :--- | :--- |
| **Same Model (e.g., YOLOv8)** | Change the `classes_id` or `confidence` parameters in the RPC request. |
| **Same Model Task** | Swap between `AiInferenceDetectionClient`, `AiInferenceSegmentationClient`, or `AiInferenceClassificationClient`. |
| **Custom Model (e.g., ONNX/TensorRT)** | 1. Deploy a new Inference Service container.<br>2. Define a new RabbitMQ queue for this service.<br>3. Add a matching Client class in `Ai_engine` to handle the new RPC routing. |
| **Trained Weights Update** | Update the model weights file path in the Inference Service's environment variables. |
