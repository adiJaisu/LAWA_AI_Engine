import { useRef, useEffect, useState } from "react";
import { formatDateTime } from "../utils/dateTimeUtils";

interface Props {
  videoUrl: string | null;
  startTime?: string;
}

const VideoPlayer = ({ videoUrl, startTime }: Props) => {
  const videoRef = useRef<HTMLVideoElement>(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (videoUrl && videoRef.current) {
      videoRef.current.play().catch(() => { });
      setIsPlaying(true);
    }
  }, [videoUrl]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleLoadedMetadata = () => {
      setDuration(video.duration);
    };

    const handleTimeUpdate = () => {
      setCurrentTime(video.currentTime);
      setProgress((video.currentTime / video.duration) * 100);
    };

    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);

    video.addEventListener("loadedmetadata", handleLoadedMetadata);
    video.addEventListener("timeupdate", handleTimeUpdate);
    video.addEventListener("play", handlePlay);
    video.addEventListener("pause", handlePause);

    return () => {
      video.removeEventListener("loadedmetadata", handleLoadedMetadata);
      video.removeEventListener("timeupdate", handleTimeUpdate);
      video.removeEventListener("play", handlePlay);
      video.removeEventListener("pause", handlePause);
    };
  }, [videoUrl]);

  const handlePlayPause = () => {
    if (!videoRef.current) return;

    if (videoRef.current.paused) {
      videoRef.current.play();
    } else {
      videoRef.current.pause();
    }
  };

  const handleVolumeToggle = () => {
    if (!videoRef.current) return;
    videoRef.current.muted = !videoRef.current.muted;
    setIsMuted(videoRef.current.muted);
  };

  const handleFullscreen = () => {
    const container = videoRef.current?.parentElement;
    if (!container) return;

    if (document.fullscreenElement) {
      document.exitFullscreen();
    } else {
      container.requestFullscreen();
    }
  };


  const handleDownload = () => {
    if (!videoUrl) return;

    const link = document.createElement("a");
    link.href = videoUrl;
    link.download = "event_video.mp4";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleProgressClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!videoRef.current) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const pos = (e.clientX - rect.left) / rect.width;
    videoRef.current.currentTime = pos * videoRef.current.duration;
  };

  const formatTime = (seconds: number) => {
    if (isNaN(seconds)) return "0:00";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <div className="event-video-container">
      <video
        ref={videoRef}
        className="event-video-player"
        autoPlay
        src={videoUrl ?? undefined}
      >
        Your browser does not support the video tag.
      </video>

      {startTime && (
        <div className="video-timestamp">
          {formatDateTime(startTime)}
        </div>
      )}

      <div className="video-controls">
        <div className="video-time-display">
          {formatTime(currentTime)} / {formatTime(duration)}
        </div>

        <div
          className="video-progress-bar"
          onClick={handleProgressClick}
        >
          <div
            className="video-progress-fill"
            style={{ width: `${progress}%` }}
          ></div>
        </div>

        <div className="control-buttons">
          <button className="control-btn" onClick={handlePlayPause} title={isPlaying ? "Pause" : "Play"}>
            <span className="material-icons">
              {isPlaying ? "pause" : "play_arrow"}
            </span>
          </button>

          <button className="control-btn" onClick={handleVolumeToggle} title={isMuted ? "Unmute" : "Mute"}>
            <span className="material-icons">
              {isMuted ? "volume_off" : "volume_up"}
            </span>
          </button>

          <button className="control-btn" onClick={handleFullscreen} title="Fullscreen">
            <span className="material-icons">fullscreen</span>
          </button>

          <button className="control-btn" onClick={handleDownload} title="Download">
            <span className="material-icons">download</span>
          </button>
        </div>
      </div>
    </div>
  );
};

export default VideoPlayer;
