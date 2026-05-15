import type { GetAllRolesResponse } from "../types/role.types";
import { http } from "./axiosInstance";

export const getAllRoles = async (
): Promise<GetAllRolesResponse> => {

  try {
    const response = await http.get<GetAllRolesResponse>(
      `/roles/getallroles`,
    );

    return response.data;
  } catch (error: any) {
    console.error("getAllRoles failed:", error);
    throw error;
  }
};
