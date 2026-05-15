import type {
  GetAllCamerasResponse,
  GetAllUsecasesResponse,
  GetCameraDetailsResponse,
  GetCameraFrameResponse,
  UpdateRoiResponse
} from "../types/camera.types";
import { http } from "./axiosInstance";

export const getAllCameras = async (): Promise<GetAllCamerasResponse> => {
  try {
    const response = await http.get<GetAllCamerasResponse>(`/camera/getallcameras`);
    return response.data;
  } catch (error: any) {
    console.error("getAllCameras failed:", error);
    throw error;
  }
};

export const getCameraDetails = async (cameraId: number): Promise<GetCameraDetailsResponse> => {
  try {
    const response = await http.get<GetCameraDetailsResponse>(`/camera/getcameradetail/${cameraId}`);
    return response.data;
  } catch (error: any) {
    console.error("getCameraDetails failed:", error);
    throw error;
  }
};

export const addCamera = async (payload: any) => {
  try {
    const response = await http.post(`/camera/addcamera`, payload);
    return response.data;
  } catch (error: any) {
    console.error("addCamera failed:", error);
    throw error;
  }
};

export const updateCamera = async (payload: any) => {
  try {
    const response = await http.put(`/camera/updatecamera`, payload);
    return response.data;
  } catch (error: any) {
    console.error("updateCamera failed:", error);
    throw error;
  }
};

export const deleteCamera = async (cameraId: number): Promise<{ code: number; message: string }> => {
  try {
    const response = await http.delete(`/camera/deletecamera/${cameraId}`);
    return response.data;
  } catch (error: any) {
    console.error("deleteCamera failed:", error);
    throw error;
  }
};

export const getAllUsecases = async (): Promise<GetAllUsecasesResponse> => {
  try {
    const response = await http.get<GetAllUsecasesResponse>(`/usecases/getallusecases`);
    return response.data;
  } catch (error: any) {
    console.error("getAllUsecases failed:", error);
    throw error;
  }
};

export const updateRoi = async (payload: any): Promise<UpdateRoiResponse> => {
  try {
    const response = await http.put<UpdateRoiResponse>(`/roi/updateroi`, payload);
    return response.data;
  } catch (error: any) {
    console.error("updateRoi failed:", error);
    throw error;
  }
};

export const getCameraFrame = async (cameraId: number): Promise<GetCameraFrameResponse> => {
  try {
    const response = await http.get<GetCameraFrameResponse>(`/roi/getcameraframe?cameraId=${cameraId}`);
    return response.data;
  } catch (error: any) {
    console.error("getCameraFrame failed:", error);
    throw error;
  }
};

export const refreshCameraFrame = async (cameraId: number): Promise<GetCameraFrameResponse> => {
  try {
    const response = await http.post<GetCameraFrameResponse>(`/roi/refreshcameraframe?cameraId=${cameraId}`);
    return response.data;
  } catch (error: any) {
    console.error("refreshCameraFrame failed:", error);
    throw error;
  }
};
