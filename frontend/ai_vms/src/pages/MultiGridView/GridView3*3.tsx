// ─── GridView3x3.tsx ──────────────────────────────────────────────────────────

import React from 'react';
import VideoCell from './VideoCell';
import type { VideoState } from '../../types/Video.types';

interface GridView3x3Props {
  videos: VideoState[];
  onUpdate: (id: number, patch: Partial<VideoState>) => void;
}

const GridView3x3: React.FC<GridView3x3Props> = ({ videos, onUpdate }) => {
  const visible = videos.slice(0, 9);

  return (
    <div className="mgv-grid mgv-grid--3">
      {visible.map(video => (
        <VideoCell key={video.id} video={video} onUpdate={onUpdate} />
      ))}
    </div>
  );
};

export default GridView3x3;