// ─── GridView1x1.tsx ──────────────────────────────────────────────────────────

import React from 'react';
import VideoCell from './VideoCell';
import type { VideoState } from '../../types/Video.types';

interface GridView1x1Props {
  videos: VideoState[];
  onUpdate: (id: number, patch: Partial<VideoState>) => void;
}

const GridView1x1: React.FC<GridView1x1Props> = ({ videos, onUpdate }) => {
  const video = videos[0];

  if (!video) return null;

  return (
    <div className="mgv-grid mgv-grid--1">
      <VideoCell video={video} onUpdate={onUpdate} />
    </div>
  );
};

export default GridView1x1;