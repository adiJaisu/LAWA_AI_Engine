// ─── GridView2x2.tsx ──────────────────────────────────────────────────────────

import React from 'react';
import VideoCell from './VideoCell';
import type { VideoState } from '../../types/Video.types';

interface GridView2x2Props {
  videos: VideoState[];
  onUpdate: (id: number, patch: Partial<VideoState>) => void;
}

const GridView2x2: React.FC<GridView2x2Props> = ({ videos, onUpdate }) => {
  const visible = videos.slice(0, 4);

  return (
    <div className="mgv-grid mgv-grid--2">
      {visible.map(video => (
        <VideoCell key={video.id} video={video} onUpdate={onUpdate} />
      ))}
    </div>
  );
};

export default GridView2x2;