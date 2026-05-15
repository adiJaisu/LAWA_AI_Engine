// ─── InfoMenu.tsx ─────────────────────────────────────────────────────────────

import React, { useRef, useEffect } from 'react';
import type { VideoState } from '../../types/Video.types';

interface InfoMenuProps {
  video: VideoState;
  onClose: () => void;
}

const InfoMenu: React.FC<InfoMenuProps> = ({ video, onClose }) => {
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const timer = setTimeout(() => {
      const handleOutside = (e: MouseEvent) => {
        if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
          onClose();
        }
      };
      document.addEventListener('mousedown', handleOutside);
      return () => document.removeEventListener('mousedown', handleOutside);
    }, 0);
    return () => clearTimeout(timer);
  }, [onClose]);

  return (
    <div className="info-menu" ref={menuRef}>
      <div className="info-menu-header">Camera Info</div>
      <table className="info-table">
        <tbody>
          <tr>
            <td className="info-key">Name</td>
            <td className="info-val">{video.label}</td>
          </tr>
          <tr>
            <td className="info-key">RTSP URL</td>
            <td className="info-val info-val--mono">{video.rtspUrl}</td>
          </tr>
          <tr>
            <td className="info-key">Resolution</td>
            <td className="info-val">{video.resolution}</td>
          </tr>
          <tr>
            <td className="info-key">Use Case</td>
            <td className="info-val">{video.usecase}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
};

export default InfoMenu;