export const loginUser = async (email: string, password: string) => {
  const backendUrl = import.meta.env.VITE_BACKEND_URL;
  const response = await fetch(`${backendUrl}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Login failed");
  }

  const data = await response.json();
  sessionStorage.setItem("access-token", data.access_token);
  sessionStorage.setItem("user", JSON.stringify(data.user));

  // Optionally fetch full details if needed, but for now data.user is enough
  return data;
};

export const signupUser = async (email: string, password: string, firstName: string, lastName: string) => {
  const backendUrl = import.meta.env.VITE_BACKEND_URL;
  const response = await fetch(`${backendUrl}/auth/signup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password, firstName, lastName }),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.detail || "Signup failed");
  }

  return await response.json();
};

export const verifySession = async () => {
  const token = sessionStorage.getItem("access-token");
  if (!token) return null;

  const backendUrl = import.meta.env.VITE_BACKEND_URL;
  try {
    const response = await fetch(`${backendUrl}/auth/secure`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${token}`
      },
    });

    if (response.ok) {
      const data = await response.json();
      sessionStorage.setItem("user", JSON.stringify(data.user));
      sessionStorage.setItem("permissions", JSON.stringify(data.permissions));
      sessionStorage.setItem("camera", JSON.stringify(data.camera));
      return data;
    } else {
      sessionStorage.clear();
      return null;
    }
  } catch (error) {
    console.error("Session verification failed", error);
    return null;
  }
};

export const getUser = () => {
  const user = sessionStorage.getItem("user");
  return user ? JSON.parse(user) : null;
};

export const getPermissions = () => {
  const permissions = sessionStorage.getItem("permissions");
  return permissions ? JSON.parse(permissions) : [];
};

export const getAuthToken = (): string | null => sessionStorage.getItem("access-token");