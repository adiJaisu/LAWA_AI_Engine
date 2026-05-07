export type CameraFormData = {
  name: string;
  type: string;
  resolution: string;
  fps: string;
  codec: string;
  rtspUrl?: string;
  height: string;
  resolutionWidth: string;
  resolutionHeight: string;
  decodingResource: string;
  decodingPipeline: string;
  dockerMode: string;
  status: number;
};

export type Camera = {
  name: string;
  resolution: string;
  rtspUrl?: string;
  fps: string;
  type: string;
  codec: string;
  height: string;
  resolutionWidth: string;
  resolutionHeight: string;
  decodingResource: string;
  decodingPipeline: string;
  dockerMode: string;
  status: number;
};

export interface CameraApiItem {
  cameraId: number;
  name: string;
  type: string;
  codec?: string;
  fps?: string;
  rtspUrl?: string;
  resolution: string;
  height?: number;
  resolutionWidth?: number;
  resolutionHeight?: number;
  decodingResource?: string;
  decodingPipeline?: string;
  dockerMode?: string;
  usecases?: { usecaseId: number; usecaseName: string }[];
  roi?: any;
  status: number;
  createdAt: string;
  updatedAt: string;
}

export type GetAllCamerasResponse = {
  code: number;
  message: string;
  cameras: CameraApiItem[];
};

export type GetCameraDetailsResponse = {
  code: number;
  message: string;
  cameraDetails: CameraApiItem;
};

export type UpdateRoiResponse = {
  code: number;
  message: string;
  roi: any;
};

export type CameraRow = {
  id: number;
  cameraId?: number;
  name: string;
  type: string;
  codec?: string;
  fps?: string;
  status: "Active" | "Inactive";
  rtspUrl?: string;
  resolution: string;
  height?: number;
  resolutionWidth?: number;
  resolutionHeight?: number;
  decodingResource?: string;
  decodingPipeline?: string;
  dockerMode?: string;
  createdAt?: string;
  updatedAt?: string;
  roi?: any;
  usecases?: Usecase[];
};

export type FilterState = {
  name: string;
  workstation: string;
  branch: string;
};

export interface Usecase {
  usecaseId: number;
  usecaseName: string;
  classes?: string[];
}

export interface GetAllUsecasesResponse {
  code: number;
  message: string;
  usecaseDetails: Usecase[];
}

export interface GetCameraFrameResponse {
  code: number;
  message: string;
  frameFile: Blob | string | null;
}
