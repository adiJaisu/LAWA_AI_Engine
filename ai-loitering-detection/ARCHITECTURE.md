# AI Engine Reference Architecture

This document provides a comprehensive overview of the `ai-engine` processing pipeline. By reading this guide, you will understand the data flow, global management systems, thread integration, and how you can seamlessly extend the application to support new computer vision usecases.

## 1. System Overview
The `ai-engine` is designed to be a highly concurrent, modular AI pipeline that bridges Camera RTSP metadata (via RabbitMQ) to a GPU Processing fleet, returning analytical events. 
Currently configured to execute the **Loitering Detection** pipeline standalone.

### Directory Structure
```
ai-loitering-detection/
├── Dockerfile                  # Application build configuration
├── requirements.txt            # Python dependencies
├── app.py                      # Application Entrypoint & Orchestrator
└── ai-engine/src/              # Core Source Code Layer
    ├── business_cases/         # Usecase logic (e.g. Loitering checks)
    ├── concurrent/             # (Not Used in Loitering)
    ├── config/                 # config.ini configurations
    ├── constant/               # Core configuration strings & defaults
    ├── detectors/              # Model Wrappers (YOLO ObjectDetector)
    ├── Exception/              # Custom Error Management
    ├── executers/              # Iterators to handle Batch operations
    ├── models/                 # Cached Model files (*.pt)
    └── utils/                  # Core Systems (RabbitMQ, GPU manager, etc)
```

---

## 2. Core Components & Workflow

### Phase 1: Entry & Ingestion (`app.py` & `RabbitmqConsumer.py`)
1. **Application Start**: `app.py` is invoked. It maps the environmental variables and reads `config.ini` to establish worker constraints.
2. **Global Detector Loading**: `GPUManager` is initialized early. It allocates primary execution models into RAM/VRAM so the system is primed before data arrives.
3. **Message Consumption**: `RabbitMQConsumerManager` creates dedicated threads listening strictly to the configured usecase queue (e.g., `queue_loitering_detection`). As messages arrive, they are placed directly into `VisionPipeline.global_rabbitmq_message_queue` for main-loop retrieval.

### Phase 2: Pipeline Orchestration (`main.py`)
1. **Async Batch Engine**: The function `process_vision_stream()` continuously polls the global thread-safe queue.
2. **Metadata Validation**: The `Validator` processes incoming frames, checking parameters such as missing IDs or corrupt crops.
3. **Sub-Batching & Base64 Decoding**: `Utilies.py` decodes the input Base64 buffers dynamically utilizing `TurboJPEG` multithreading to rapidly restore the visual matrices safely. 
4. **Task Assignment**: The entire validated batch is handed to the `GPUManager.assign_task(...)` method for dispatch to the first available GPU worker thread.

### Phase 3: The Global Worker & Executor (`GPUdevicemanager.py` -> `Executer.py`)
1. **Freeing The Thread**: `GPUWorker` accepts the batch to run isolated from the main receiver.
2. **Executing The Dispatch**: The worker natively runs `dispatch_vision_task()` pointing to `execute_loitering_detection()`.
3. **Camera Context Integration**: Because streams generate continuous state (like Dwell Time and Tracking persistence), the executor relies on `VisionPipeline.camera_tracker[camera_id]`. This maps individual camera IDs uniquely to a stateful `LoiteringDetection` class instantiation.
4. **Looping**: For each frame inside the batch, it passes the data natively into the instantiated custom state-class `detect()` function.

### Phase 4: Business Logic (`LoiteringDetection.py` & `ObjectDetector.py`)
1. **Tracking Generation**: `LoiteringDetection` makes a call to `detector.make_prediction_with_tracking()`. 
2. **Tracker Utilization**: The custom overriding tracker applies Bytrack sorting on the frames internally and assigns spatial IDs that persist across inferences securely. 
3. **Algorithmic Logic**: Using Shapely polygon intersections, the tracker bounds map to Zone interactions ("NRUA", "NRPA", "RPA") resolving proximity and loitering limits based on dynamic `config.ini` threshold properties heavily logging the occurrences.

---

## 3. How to Integrate a New Usecase in the Future

If your project requires adding a completely new use-case to this codebase, follow this standard pattern to enforce decoupling and architecture stability:

### Step 1: Define Constants and Queue Logic
Head to `src/constant/constants.py` and map your new logic. 
- Create a Queue string: `NEW_DETECTION_QUEUE = queue_new_detection`.
- Create a Usecase string: `NEW_USECASE_NAME = "New_Detection"`.
- Map them inside the `USECASE_QUEUE_MAPPING` dictionary.

### Step 2: Implement the Algorithm (`business_cases`)
Create `src/business_cases/NewDetection.py`:
- Establish an `__init__()` configuring class dependencies relying heavily on configurations sourced natively from `cfg`.
- Write a core `def detect(self, frame, rois, detector, ...):` method returning the output `Frame, boolean_alert, data_metrics`.

### Step 3: Bundle By Executor (`executers`)
Create `src/executers/NewDetector_executer.py`:
- Instantiate your business logic class using the identical `VisionPipeline.camera_tracker[camera_id]` design element to guarantee memory allocation cleanly per-camera source streaming.
- Iterate over the incoming `.batch` to compile the final payload lists correctly.

### Step 4: Plug it into the Main Dispatch (`main.py`)
Edit `dispatch_vision_task()` in `src/main.py`:
```python
if usecase_name == Constants.LOITERING_DETECTION_USECASE:
    results = execute_loitering_detection(...)
elif usecase_name == Constants.NEW_USECASE_NAME:
    from src.executers.NewDetector_executer import execute_new_detection
    results = execute_new_detection(validated_msg_with_frames_and_metadatas, detector=detector)
```

### Step 5: Adjust `app.py` Initializer Contexts
If the new usecase utilizes entirely distinct primary inference (meaning YOLO isn't applicable), update your `app.py` initialization step configures so that `GPUManager` is supplied with the proper secondary configurations via:
`GPUManager(num_workers_per_gpu=..., model_path=..., secondary_model_path=...)`

Following this strictly guarantees the RabbitMQ -> Python Backend remains fully decoupled, testable, and completely self-scaling via the GPU Manager queue tracking logic.
