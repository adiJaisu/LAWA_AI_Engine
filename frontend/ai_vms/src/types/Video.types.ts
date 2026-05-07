

export interface VideoState {
  id: number;
  isPlaying: boolean;
  isMuted: boolean;
  volume: number;
  currentTime: number;
  duration: number;
  label: string;
  src: string;
  rtspUrl: string;
  resolution: string;
  usecase: string;
}

export type GridSize = 1 | 2 | 3;

export const formatTime = (seconds: number): string => {
  if (isNaN(seconds)) return '00:00';
  const m = Math.floor(seconds / 60).toString().padStart(2, '0');
  const s = Math.floor(seconds % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
};

export const INITIAL_VIDEOS: VideoState[] = Array.from({ length: 9 }, (_, i) => ({
  id: i + 1,
  isPlaying: true,
  isMuted: true,
  volume: 0.7,
  currentTime: 0,
  duration: 0,
  label: `Camera ${i + 1}`,
  src: `https://www.w3schools.com/html/mov_bbb.mp4`,
  rtspUrl: i % 2 === 0 ? `rtsp://192.168.1.${10 + i}:554/stream1` : 'NA',
  resolution: ['1920×1080', '1280×720', '3840×2160'][i % 3],
  usecase: ['Storage Area Exit/Entry', 'Customer Interactions', 'Cash Handling',
            'Vault Access', 'Teller Operations', 'None'][i % 6],
}));