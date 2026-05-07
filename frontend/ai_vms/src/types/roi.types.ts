export interface CameraFrameResponse {
  code: number;
  message: string;
  frameFile: string | null;
}

export interface RoiAnnotationsResponse {
  code: number;
  message: string;
  roiJson?: string | RoiJsonObject | null;
  annotations: Annotation[];
}

export interface RoiSaveResponse {
  code: number;
  message: string;
  success: boolean;
}

export interface UseCase {
  usecaseId: number;
  usecaseName: string;
  is_active: boolean;
}

export interface CameraDetails {
  cameraModel: string;
  cameraHeight: number;
}

export interface Point {
  x: number;
  y: number;
}

export interface RectCoords {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface BaseAnnotation extends RectCoords {
  label: string;
  type: "rectangle" | "polygon";
  useCases?: string[];
  isLocked?: boolean;
  isVisible?: boolean;
}

export interface RectAnnotation extends BaseAnnotation {
  type: "rectangle";
}

export interface PolygonAnnotation extends BaseAnnotation {
  type: "polygon";
  points: Point[];
}

export type Annotation = RectAnnotation | PolygonAnnotation;

export interface RoiJsonObject {
  annotations: Annotation[];
  imageWidth: number;
  imageHeight: number;
  timestamp?: string;
}

export interface Tool {
  label: 'Rectangle ROI' | 'Polygon ROI' | 'Edit ROI';
  icon: string;
}

export interface ImageDimensions {
  width: number;
  height: number;
  naturalWidth: number;
  naturalHeight: number;
}

export type DrawingMode = 'rectangle' | 'polygon' | 'edit' | null;
