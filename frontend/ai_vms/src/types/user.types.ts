export type UserTableRow = {
  id: string | number;
  firstName: string;
  lastName: string;
  username: string;
  status: number;
  role: string;
  roleId?: number;
  createdAt: string;
};

export type UserFilterState = {
  username: string;
  firstName: string;
  lastName: string;
  role: string;
  fromDate: Date | null;
  toDate: Date | null;
};

export type UserFormData = {
  firstName: string;
  lastName: string;
  emailId: string;
  status: "Active" | "Inactive";
  mobileNumber: string;
  role: string;
};

export type User = {
  userId: number;
  username: string;
  firstName: string;
  lastName: string;
  roleId: number;
  roleName: string;
  status: number;
  createdAt: string;
  updatedAt: string;
};

export type GetAllUsersResponse = {
  code: number;
  message: string;
  users: User[];
};

export type AddUserPayload = {
  email: string;
  firstName: string;
  lastName: string;
  isActive: boolean;
  roleId: number;
  createdBy?: number;
};

export type UpdateUserPayload = {
  userId: string | number;
  email: string;
  firstName: string;
  lastName: string;
  isActive: boolean;
  roleId?: number;
  updatedBy?: number;
};