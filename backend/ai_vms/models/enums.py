import enum

class CameraType(str, enum.Enum):
    ip = "rtsp"
    usb = "usb"


class DecodingResource(str, enum.Enum):
    cpu = "cpu"
    gpu = "gpu"
    vaapi = "vaapi"


class DecodingPipeline(str, enum.Enum):
    ffmpeg = "ffmpeg"
    opencv = "opencv"
    gstreamer = "gstreamer"


class DockerMode(str, enum.Enum):
    build = "build"
    load = "load"


class AIResource(str, enum.Enum):
    cpu = "cpu"
    gpu = "gpu"
    vaapi = "vaapi"
