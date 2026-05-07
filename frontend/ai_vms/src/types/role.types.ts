export type Role = {
  roleId: number;
  roleName: string;
};

export type GetAllRolesResponse = {
  code: number;
  message: string;
  roleDetails: Role[];
};