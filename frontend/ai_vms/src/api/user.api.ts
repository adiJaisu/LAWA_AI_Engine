import type { GetAllUsersResponse, AddUserPayload, UpdateUserPayload } from "../types/user.types";
import { http } from "./axiosInstance";

export const getAllUsers = async (): Promise<GetAllUsersResponse> => {
  try {
    const response = await http.get<GetAllUsersResponse>(`users/getallusers`);
    return response.data;
  } catch (error: any) {
    console.error("getAllUsers failed:", error);
    throw error;
  }
};

export const getUserDetail = async (userId: number) => {
  const response = await http.get(`/users/getuserdetail/${userId}`);
  return response.data.user;
};

export const addUser = async (payload: AddUserPayload) => {
  try {
    const response = await http.post(`/users/adduser`, payload);
    return response.data;
  } catch (error: any) {
    console.error("addUser failed:", error);
    throw error;
  }
};

export const updateUser = async (payload: UpdateUserPayload) => {
  try {
    const response = await http.put(`/users/updateuser`, payload);
    return response.data;
  } catch (error: any) {
    console.error("updateUser failed:", error);
    throw error;
  }
};

export const deleteUser = async (userId: number): Promise<{ code: number; message: string }> => {
  try {
    const response = await http.delete(`/users/deleteuser/${userId}`);
    return response.data;
  } catch (error: any) {
    console.error("deleteUser failed:", error);
    throw error;
  }
};
