// // ─── MultiGridView.tsx ────────────────────────────────────────────────────────

// import React, { useState, useCallback } from 'react';
// import '../../styles/MultiGridView.css';
// import Layout from '../../components/Layout';
// import { INITIAL_VIDEOS } from '../../types/Video.types';
// import type { VideoState, GridSize } from '../../types/Video.types';
// import GridView1x1 from './GridView1*1';
// import GridView2x2 from './GridView2*2';
// import GridView3x3 from './GridView3*3';

// // ─── Grid Switcher Icons ──────────────────────────────────────────────────────

// const GridIcon1x1 = () => (
//   <svg width="15" height="15" viewBox="0 0 15 15" fill="currentColor">
//     <rect x="1" y="1" width="13" height="13" rx="2" />
//   </svg>
// );

// const GridIcon2x2 = () => (
//   <svg width="15" height="15" viewBox="0 0 15 15" fill="currentColor">
//     <rect x="1"   y="1"   width="5.5" height="5.5" rx="1" />
//     <rect x="8.5" y="1"   width="5.5" height="5.5" rx="1" />
//     <rect x="1"   y="8.5" width="5.5" height="5.5" rx="1" />
//     <rect x="8.5" y="8.5" width="5.5" height="5.5" rx="1" />
//   </svg>
// );

// const GridIcon3x3 = () => (
//   <svg width="15" height="15" viewBox="0 0 15 15" fill="currentColor">
//     <rect x="1"    y="1"    width="3.5" height="3.5" rx="0.6" />
//     <rect x="5.7"  y="1"    width="3.5" height="3.5" rx="0.6" />
//     <rect x="10.4" y="1"    width="3.5" height="3.5" rx="0.6" />
//     <rect x="1"    y="5.7"  width="3.5" height="3.5" rx="0.6" />
//     <rect x="5.7"  y="5.7"  width="3.5" height="3.5" rx="0.6" />
//     <rect x="10.4" y="5.7"  width="3.5" height="3.5" rx="0.6" />
//     <rect x="1"    y="10.4" width="3.5" height="3.5" rx="0.6" />
//     <rect x="5.7"  y="10.4" width="3.5" height="3.5" rx="0.6" />
//     <rect x="10.4" y="10.4" width="3.5" height="3.5" rx="0.6" />
//   </svg>
// );

// const GRID_OPTIONS: { size: GridSize; label: string; Icon: React.FC }[] = [
//   { size: 1, label: '1×1', Icon: GridIcon1x1 },
//   { size: 2, label: '2×2', Icon: GridIcon2x2 },
//   { size: 3, label: '3×3', Icon: GridIcon3x3 },
// ];

// // ─── Main Component ───────────────────────────────────────────────────────────

// const MultiGridView: React.FC = () => {
//   const [videos, setVideos] = useState<VideoState[]>(INITIAL_VIDEOS);
//   const [gridSize, setGridSize] = useState<GridSize>(3);

//   React.useEffect(() => {
//     const backendHost = window.location.hostname;
//     const token = sessionStorage.getItem("access-token");
    
//     fetch(`http://${backendHost}:8010/api/v1/camera/getallcameras`, {
//       headers: {
//         "Authorization": `Bearer ${token}`
//       }
//     })
//       .then(res => res.json())
//       .then(data => {
//         const camerasList = data.cameras || [];
//         setVideos(prev => {
//           const newVideos = [...prev];
//           camerasList.forEach((c: any, index: number) => {
//             if (index < newVideos.length) {
//               newVideos[index] = {
//                 ...newVideos[index],
//                 id: c.cameraId,
//                 label: c.name || `Camera ${c.cameraId}`,
//                 rtspUrl: c.rtspUrl,
//                 resolution: c.resolution || '1920x1080',
//                 isLiveStream: true,
//                 usecase: '' // Diagram: no default usecase
//               };
//             }
//           });
//           return newVideos;
//         });
//       })
//       .catch(err => console.error("Failed to fetch cameras:", err));
//   }, []);

//   const updateVideo = useCallback((id: number, patch: Partial<VideoState>) => {
//     setVideos(prev => prev.map(v => v.id === id ? { ...v, ...patch } : v));
//   }, []);


//   const renderGrid = () => {
//     switch (gridSize) {
//       case 1: return <GridView1x1 videos={videos} onUpdate={updateVideo} />;
//       case 2: return <GridView2x2 videos={videos} onUpdate={updateVideo} />;
//       case 3: return <GridView3x3 videos={videos} onUpdate={updateVideo} />;
//     }
//   };

//   return (
//     <Layout>
//       <div className="page-container">
//         <div className="page-header">
//           <div className="header-left">
//             <span className="header-title">Multi-Grid View</span>
//             <span className="header-subtitle">Live camera feeds — independent controls per stream</span>
//           </div>

//           <div className="right-header">
//             <div className="grid-switcher">
//               {GRID_OPTIONS.map(({ size, label, Icon }) => (
//                 <button
//                   key={size}
//                   className={`grid-switch-btn ${gridSize === size ? 'active' : ''}`}
//                   onClick={() => setGridSize(size)}
//                   title={`${label} view`}
//                 >
//                   <Icon />
//                   <span>{label}</span>
//                 </button>
//               ))}
//             </div>
//           </div>
//         </div>

//         {renderGrid()}
//       </div>
//     </Layout>
//   );
// };

// export default MultiGridView;


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
 
// cameras per page for each grid size
const CAMERAS_PER_PAGE: Record<GridSize, number> = {
  1: 1,
  2: 4,
  3: 9,
};
 
// ─── Main Component ───────────────────────────────────────────────────────────
 
const MultiGridView: React.FC = () => {
  const [videos, setVideos] = useState<VideoState[]>(INITIAL_VIDEOS);
  const [gridSize, setGridSize] = useState<GridSize>(3);
  const [currentPage, setCurrentPage] = useState(1);
 
  React.useEffect(() => {
    const backendHost = window.location.hostname;
    const token = sessionStorage.getItem("access-token");
 
    fetch(`http://${backendHost}:8010/api/v1/camera/getallcameras`, {
      headers: {
        "Authorization": `Bearer ${token}`
      }
    })
      .then(res => res.json())
      .then(data => {
        const camerasList = data.cameras || [];
        setVideos(prev => {
          const newVideos = [...prev];
          camerasList.forEach((c: any, index: number) => {
            if (index < newVideos.length) {
              newVideos[index] = {
                ...newVideos[index],
                id: c.cameraId,
                label: c.name || `Camera ${c.cameraId}`,
                rtspUrl: c.rtspUrl,
                resolution: c.resolution || '1920x1080',
                isLiveStream: true,
                usecase: c.usecases?.map((u: any) => u.usecaseName).join(', ') || ''
              };
            }
          });
          return newVideos;
        });
      })
      .catch(err => console.error("Failed to fetch cameras:", err));
  }, []);
 
  const updateVideo = useCallback((id: number, patch: Partial<VideoState>) => {
    setVideos(prev => prev.map(v => v.id === id ? { ...v, ...patch } : v));
  }, []);
 
  // reset to page 1 when grid size changes
  const handleGridChange = (size: GridSize) => {
    setGridSize(size);
    setCurrentPage(1);
  };
 
  const perPage = CAMERAS_PER_PAGE[gridSize];
  const totalPages = Math.ceil(videos.length / perPage);
  const startIndex = (currentPage - 1) * perPage;
  const pagedVideos = videos.slice(startIndex, startIndex + perPage);
 
  const handlePageInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = Number(e.target.value);
    if (val >= 1 && val <= totalPages) setCurrentPage(val);
  };
 
  const renderGrid = () => {
    switch (gridSize) {
      case 1: return <GridView1x1 videos={pagedVideos} onUpdate={updateVideo} />;
      case 2: return <GridView2x2 videos={pagedVideos} onUpdate={updateVideo} />;
      case 3: return <GridView3x3 videos={pagedVideos} onUpdate={updateVideo} />;
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
                  onClick={() => handleGridChange(size)}
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
 
        {/* ── Pagination ── */}
        {totalPages > 1 && (
          <div className="pagination-bar">
            <button
              className="pagination-arrow-btn"
              onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
              disabled={currentPage === 1}
              title="Previous page"
            >
              <span className="material-icons">chevron_left</span>
            </button>
 
            <input
              type="number"
              className="pagination-input"
              value={currentPage}
              min={1}
              max={totalPages}
              onChange={handlePageInput}
            />
 
<span className="pagination-of">of</span>
<span className="pagination-total">{totalPages}</span>
 
            <button
              className="pagination-arrow-btn"
              onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
              disabled={currentPage === totalPages}
              title="Next page"
            >
              <span className="material-icons">chevron_right</span>
            </button>
          </div>
        )}
 
      </div>
    </Layout>
  );
};
 
export default MultiGridView;