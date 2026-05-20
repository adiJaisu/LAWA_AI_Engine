// ─── VideoCell.tsx ────────────────────────────────────────────────────────────

import React, { useState, useRef, useCallback, useEffect } from 'react';
import InfoMenu from './InfoMenu';
import { formatTime } from '../../types/Video.types';
import type { VideoState } from '../../types/Video.types';
import WebRTCPlayer from '../../components/WebRTCPlayer';

interface VideoCellProps {
  video: VideoState;
  onUpdate: (id: number, patch: Partial<VideoState>) => void;
}

const VideoCell: React.FC<VideoCellProps> = ({ video, onUpdate }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [showMenu, setShowMenu] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);

  useEffect(() => {
    const handleFsChange = () => setIsFullscreen(!!document.fullscreenElement);
    document.addEventListener('fullscreenchange', handleFsChange);
    return () => document.removeEventListener('fullscreenchange', handleFsChange);
  }, []);

  const togglePlay = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    if (video.isPlaying) { v.pause(); onUpdate(video.id, { isPlaying: false }); }
    else { v.play(); onUpdate(video.id, { isPlaying: true }); }
  }, [video.isPlaying, video.id, onUpdate]);

  const handleTimeUpdate = () => {
    const v = videoRef.current;
    if (v) onUpdate(video.id, { currentTime: v.currentTime, duration: v.duration });
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const v = videoRef.current;
    if (v) { v.currentTime = Number(e.target.value); onUpdate(video.id, { currentTime: v.currentTime }); }
  };

  const handleVolume = (e: React.ChangeEvent<HTMLInputElement>) => {
    const v = videoRef.current;
    const vol = Number(e.target.value);
    if (v) { v.volume = vol; onUpdate(video.id, { volume: vol, isMuted: vol === 0 }); }
  };

  const handleMuteToggle = () => {
    const v = videoRef.current;
    if (!v) return;
    v.muted = !v.muted;
    onUpdate(video.id, { isMuted: v.muted });
  };

  const toggleFullscreen = () => {
    if (!containerRef.current) return;
    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen().catch(err =>
        console.error(`Fullscreen error: ${err.message}`)
      );
    } else {
      document.exitFullscreen();
    }
  };

  const progress = video.duration ? (video.currentTime / video.duration) * 100 : 0;

  return (
    <div ref={containerRef} className="video-cell">
      <div className="cell-label">
        <span className="cell-dot" />
        {video.label}
      </div>

      <div className="cell-menu-wrap">
        <button
          className="cell-dots-btn"
          onMouseDown={(e) => { e.stopPropagation(); setShowMenu(prev => !prev); }}
          title="Camera Info"
        >
          <span className="material-icons">more_vert</span>
        </button>
        {showMenu && <InfoMenu video={video} onClose={() => setShowMenu(false)} />}
      </div>

      {video.isLiveStream ? (
        <WebRTCPlayer 
          cameraId={video.id} 
          className="cell-video" 
          onResolutionUpdate={(res) => onUpdate(video.id, { resolution: res })}
        />
      ) : (
        <>
          <video
            ref={videoRef}
            className="cell-video"
            src={video.src}
            onTimeUpdate={handleTimeUpdate}
            onLoadedMetadata={handleTimeUpdate}
            muted={video.isMuted}
            autoPlay
            loop
            onClick={togglePlay}
            playsInline
          />
          {!video.isPlaying && (
            <button className="big-play-btn" onClick={togglePlay}>
              <span className="material-icons">play_arrow</span>
            </button>
          )}
        </>
      )}

      {!video.isLiveStream && (
        <div className="cell-controls visible">
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${progress}%` }} />
            <input
              type="range" className="progress-range"
              min={0} max={video.duration || 100} step={0.1}
              value={video.currentTime} onChange={handleSeek}
            />
          </div>

          <div className="controls-row">
            <div className="controls-left">
              <button className="ctrl-btn" onClick={togglePlay}>
                <span className="material-icons">{video.isPlaying ? 'pause' : 'play_arrow'}</span>
              </button>
              <button className="ctrl-btn" onClick={toggleMute}>
                <span className="material-icons">
                  {video.isMuted || video.volume === 0
                    ? 'volume_off'
                    : video.volume < 0.5 ? 'volume_down' : 'volume_up'}
                </span>
              </button>
              <div className="volume-wrap">
                <input
                  type="range" className="volume-range"
                  min={0} max={1} step={0.01}
                  value={video.isMuted ? 0 : video.volume} onChange={handleVolume}
                />
              </div>
              <span className="ctrl-time">
                {formatTime(video.currentTime)} / {formatTime(video.duration)}
              </span>
            </div>

            <div className="controls-right">
              <button className="ctrl-btn" onClick={toggleFullscreen} title="Fullscreen">
                <span className="material-icons">
                  {isFullscreen ? 'fullscreen_exit' : 'fullscreen'}
                </span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default VideoCell;