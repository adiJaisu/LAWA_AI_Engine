from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import httpx
from pydantic import BaseModel

from ai_vms.config.logging_config import LoggingConfig
from ai_vms.database.database_connection import get_db
from ai_vms.crud.cameras import get_camera_by_id

logger = LoggingConfig().setup_logging()

router = APIRouter(prefix="/stream", tags=["Streaming"])

# ── Port Mapping Logic (Must match orchestrator/container_manager.py) ──────────
WEBRTC_BASE_PORT = {
    "ffmpeg":     8088,
    "opencv":     8092,
    "gstreamer":  8096,
    "deepstream": 8100,
}
WEBRTC_RESOURCE_OFFSET = {"cpu": 0, "gpu": 1, "vaapi": 2}

class WebRTCOffer(BaseModel):
    sdp: str
    type: str
    camera_id: str

@router.post("/webrtc/offer")
async def webrtc_offer(offer: WebRTCOffer, db: Session = Depends(get_db)):
    """
    Proxies the WebRTC SDP offer from the UI to the internal Video Decoder.
    Routes to the correct worker container based on the camera's pipeline/resource.
    """
    try:
        # 1. Fetch camera to find its assigned port
        try:
            cam_id_int = int(offer.camera_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid camera_id format")

        camera = get_camera_by_id(db, cam_id_int)
        if not camera:
            logger.error(f"[STREAM] Camera {offer.camera_id} not found in database during WebRTC offer")
            raise HTTPException(status_code=404, detail=f"Camera {offer.camera_id} not found")

        # 2. Calculate Port
        # Enum objects in SQLAlchemy need .name to get the string value
        try:
            pipeline = camera.decoding_pipeline.name.lower()
            resource = camera.decoding_resource.name.lower()
        except AttributeError:
            # Fallback for raw strings if they aren't Enums
            pipeline = str(camera.decoding_pipeline).lower()
            resource = str(camera.decoding_resource).lower()
        
        base_port = WEBRTC_BASE_PORT.get(pipeline, 8088)
        offset    = WEBRTC_RESOURCE_OFFSET.get(resource, 0)
        target_port = base_port + offset
        target_host = "host.docker.internal"
        target_url = f"http://{target_host}:{target_port}/offer"
        
        logger.info(f"[STREAM] Proxying WebRTC offer for cam={offer.camera_id} ({camera.name}) "
                    f"to {target_url} [Pipeline: {pipeline}, Resource: {resource}]")

        # 3. Proxy request
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://{target_host}:{target_port}/offer",
                json={
                    "sdp": offer.sdp,
                    "type": offer.type,
                    "camera_id": offer.camera_id
                },
                timeout=10.0
            )
            
            if response.status_code != 200:
                logger.error(f"Video Decoder (port {target_port}) returned {response.status_code}: {response.text}")
                raise HTTPException(status_code=500, detail=f"Decoder on port {target_port} failed negotiation")
                
            return response.json()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to connect to Video Decoder WebRTC server: {e}")
        raise HTTPException(status_code=503, detail="Video Decoder WebRTC service unavailable")
