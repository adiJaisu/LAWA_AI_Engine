import axios, { AxiosError, AxiosHeaders } from "axios";
import { getAuthToken } from "./auth.api";

const baseURL = import.meta.env.VITE_BACKEND_URL || "/";

const PUBLIC_PATH_PREFIXES = ["/login", "/auth/secure"];
const isPublicPath = (url?: string) => {
    if (!url) return false;
    try {
        const path = url.startsWith("http") ? new URL(url).pathname : url;
        return PUBLIC_PATH_PREFIXES.some((p) => path.startsWith(p));
    } catch {
        return false;
    }
};

export const http = axios.create({
    baseURL,
    timeout: 30000,
});

http.interceptors.request.use(
    (config) => {
        if (!isPublicPath(config.url)) {
            const token = getAuthToken();
            if (!token) {
                try {
                    window.location.replace("/login");
                    sessionStorage.clear();
                    localStorage.clear();
                } catch { }
                if (!window.location.pathname.startsWith("/login")) {
                    window.location.replace("/login");
                }
                return Promise.reject(new axios.Cancel("No auth token; user logged out"));
            }
            const headers = config.headers as AxiosHeaders;
            headers.set("Authorization", `Bearer ${token}`);
        }

        const isFormData =
            typeof FormData !== "undefined" && config.data instanceof FormData;

        if (isFormData) {
            if (config.headers) {
                delete (config.headers as any)["Content-Type"];
                delete (config.headers as any)["content-type"];
            }
        } else {
            const looksLikeJsonBody =
                config.data &&
                typeof config.data === "object" &&
                !(config.data instanceof ArrayBuffer) &&
                !(config.data instanceof Blob) &&
                !(config.data instanceof URLSearchParams);

            if (looksLikeJsonBody) {
                const headers = config.headers as AxiosHeaders;
                if (!headers.get("Content-Type")) {
                    headers.set("Content-Type", "application/json");
                }
            }
        }

        return config;
    },
    (error: AxiosError) => Promise.reject(error)
);

http.interceptors.response.use(
    (response) => response,
    (error: AxiosError<any>) => {
        const status = error.response?.status;

        if (status === 401 || status === 403) {
            try {
                window.location.replace("/login");
                sessionStorage.clear();
                localStorage.clear();
            } catch { }
            if (!window.location.pathname.startsWith("/login")) {
                window.location.replace("/login");
            }
        }

        return Promise.reject(error);
    }
);