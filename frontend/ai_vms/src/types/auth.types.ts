export type User = {
  id: number;
  email: string;
  first_name: string;
  last_name: string;
  role_id: number;
};

export type AuthContextType = {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  authError: string | null;
  login: (email: string, password: string) => Promise<void>;
  signup: (email: string, password: string, first_name: string, last_name: string) => Promise<void>;
  logout: () => Promise<void>;
};
