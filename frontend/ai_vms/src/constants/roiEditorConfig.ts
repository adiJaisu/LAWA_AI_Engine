export const ROI_EDITOR_CONFIG = {
  MAX_SCALE: 4.0,
  MIN_SCALE: 0.2,
  SCALE_STEP: 0.1,
  CLOSE_POLYGON_THRESHOLD: 15,
  MIN_RECTANGLE_SIZE: 5,

  MESSAGES: {
    ROI_LOCKED: 'Cannot edit a locked ROI. Please unlock it first.',
    ROI_HIDDEN: 'Cannot edit a hidden ROI. Please make it visible first.',
  },

  Z_INDEX: {
    RECTANGLE: '20',
    LABEL: '21',
    HANDLE: '30',
    MODAL: '2000',
    MODAL_OVERLAY: '2000',
    MODAL_CONTENT: '2001',
  },

  TOOL_NAMES: {
    RECTANGLE: 'Rectangle ROI',
    POLYGON: 'Polygon ROI',
    EDIT: 'Edit ROI',
  } as const,
} as const;
