// ─── MultiGridView.tsx ────────────────────────────────────────────────────────

import React, { useState, useCallback } from 'react';
import '../../styles/MultiGridView.css';
import Layout from '../../components/Layout';
import { INITIAL_VIDEOS } from '../../types/Video.types';
import type { VideoState, GridSize } from '../../types/Video.types';
import GridView1x1 from './GridView1*1';
import GridView2x2 from './GridView2*2';
import GridView3x3 from './GridView3*3';

// ─── Grid Switcher Icons ──────────────────────────────────────────────────────

const GridIcon1x1 = () => (
  <svg width="15" height="15" viewBox="0 0 15 15" fill="currentColor">
    <rect x="1" y="1" width="13" height="13" rx="2" />
  </svg>
);

const GridIcon2x2 = () => (
  <svg width="15" height="15" viewBox="0 0 15 15" fill="currentColor">
    <rect x="1"   y="1"   width="5.5" height="5.5" rx="1" />
    <rect x="8.5" y="1"   width="5.5" height="5.5" rx="1" />
    <rect x="1"   y="8.5" width="5.5" height="5.5" rx="1" />
    <rect x="8.5" y="8.5" width="5.5" height="5.5" rx="1" />
  </svg>
);

const GridIcon3x3 = () => (
  <svg width="15" height="15" viewBox="0 0 15 15" fill="currentColor">
    <rect x="1"    y="1"    width="3.5" height="3.5" rx="0.6" />
    <rect x="5.7"  y="1"    width="3.5" height="3.5" rx="0.6" />
    <rect x="10.4" y="1"    width="3.5" height="3.5" rx="0.6" />
    <rect x="1"    y="5.7"  width="3.5" height="3.5" rx="0.6" />
    <rect x="5.7"  y="5.7"  width="3.5" height="3.5" rx="0.6" />
    <rect x="10.4" y="5.7"  width="3.5" height="3.5" rx="0.6" />
    <rect x="1"    y="10.4" width="3.5" height="3.5" rx="0.6" />
    <rect x="5.7"  y="10.4" width="3.5" height="3.5" rx="0.6" />
    <rect x="10.4" y="10.4" width="3.5" height="3.5" rx="0.6" />
  </svg>
);

const GRID_OPTIONS: { size: GridSize; label: string; Icon: React.FC }[] = [
  { size: 1, label: '1×1', Icon: GridIcon1x1 },
  { size: 2, label: '2×2', Icon: GridIcon2x2 },
  { size: 3, label: '3×3', Icon: GridIcon3x3 },
];

// ─── Main Component ───────────────────────────────────────────────────────────

const MultiGridView: React.FC = () => {
  const [videos, setVideos] = useState<VideoState[]>(INITIAL_VIDEOS);
  const [gridSize, setGridSize] = useState<GridSize>(3);

  const updateVideo = useCallback((id: number, patch: Partial<VideoState>) => {
    setVideos(prev => prev.map(v => v.id === id ? { ...v, ...patch } : v));
  }, []);

  const renderGrid = () => {
    switch (gridSize) {
      case 1: return <GridView1x1 videos={videos} onUpdate={updateVideo} />;
      case 2: return <GridView2x2 videos={videos} onUpdate={updateVideo} />;
      case 3: return <GridView3x3 videos={videos} onUpdate={updateVideo} />;
    }
  };

  return (
    <Layout>
      <div className="page-container">
        <div className="page-header">
          <div className="header-left">
            <span className="header-title">Multi-Grid View</span>
            <span className="header-subtitle">Live camera feeds — independent controls per stream</span>
          </div>

          <div className="right-header">
            <div className="grid-switcher">
              {GRID_OPTIONS.map(({ size, label, Icon }) => (
                <button
                  key={size}
                  className={`grid-switch-btn ${gridSize === size ? 'active' : ''}`}
                  onClick={() => setGridSize(size)}
                  title={`${label} view`}
                >
                  <Icon />
                  <span>{label}</span>
                </button>
              ))}
            </div>
          </div>
        </div>

        {renderGrid()}
      </div>
    </Layout>
  );
};

export default MultiGridView;