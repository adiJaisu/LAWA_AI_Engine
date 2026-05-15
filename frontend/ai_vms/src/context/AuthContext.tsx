import {
  createContext,
  useContext,
  useEffect,
  useState
} from "react";
import type { ReactNode } from "react";
import { loginUser, signupUser, verifySession, getUser } from "../api/auth.api";
import type { AuthContextType, User } from "../types/auth.types";

const AuthContext = createContext<AuthContextType | null>(null);

export const useAuth = (): AuthContextType => {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
};

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUser] = useState<User | null>(getUser());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const init = async () => {
      const data = await verifySession();
      if (data) {
        setUser(data.user);
      } else {
        setUser(null);
      }
      setLoading(false);
    };
    void init();
  }, []);

  const login = async (email: string, password: string) => {
    setLoading(true);
    setError(null);
    try {
      const data = await loginUser(email, password);
      // After login, we might want full details
      const fullData = await verifySession();
      if (fullData) {
        setUser(fullData.user);
      } else {
        setUser(data.user);
      }
    } catch (err: any) {
      setError(err.message || "Login failed");
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const signup = async (email: string, password: string, first_name: string, last_name: string) => {
    setLoading(true);
    setError(null);
    try {
      await signupUser(email, password, first_name, last_name);
    } catch (err: any) {
      setError(err.message || "Signup failed");
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const logout = async () => {
    sessionStorage.clear();
    localStorage.clear();
    setUser(null);
    window.location.replace("/login");
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated: !!user,
        isLoading: loading,
        authError: error,
        login,
        signup,
        logout
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};
