import { useEffect, useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import "../styles/login.css";
import ai_vmsLogo from "../../src/assets/images/AIVMS-logo.svg";

const Login = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { login, isAuthenticated, isLoading, authError } = useAuth();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [signingIn, setSigningIn] = useState<boolean>(false);

  useEffect(() => {
    if (!isLoading && isAuthenticated) {
      const origin = (location.state as any)?.from?.pathname || "/camera-details";
      navigate(origin, { replace: true });
    }
  }, [isAuthenticated, isLoading, navigate, location.state]);

  useEffect(() => {
    if (authError) setError(authError);
  }, [authError]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSigningIn(true);
    setError(null);

    try {
      await login(email, password);
    } catch (err: any) {
      setError(err.message || "Login failed. Try again.");
    } finally {
      setSigningIn(false);
    }
  };

  if (isLoading) {
    return (
      <div className="loading-container">
        <div className="spinner" />
        <p>Verifying authentication...</p>
      </div>
    );
  }

  return (
    <div className="login-container">
      <div className="hero-section">
        <div className="logo">
          <img src={ai_vmsLogo} alt="AI-VMS Logo" />
        </div>
      </div>

      <div className="auth-section">
        <div className="auth-form active">
          <div className="login-form-header">
            <p className="welcome-text">Welcome to</p>
            <h2>AIVMS</h2>
            <p>Please sign in to access your workspace</p>
          </div>

          {error && <div className="error-message">{error}</div>}

          <form onSubmit={handleSubmit} className="login-form">
            <div className="form-group">
              <label htmlFor="email">Email</label>
              <input
                type="email"
                id="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                placeholder="Enter your email"
              />
            </div>
            <div className="form-group">
              <label htmlFor="password">Password</label>
              <input
                type="password"
                id="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                placeholder="Enter your password"
              />
            </div>
            <button type="submit" className="btn-login" disabled={signingIn}>
              {signingIn ? "Signing in..." : "Sign in"}
            </button>
          </form>

          <div className="divider">
            <span>or</span>
          </div>

          <div className="login-form-footer">
            <p>Don't have an account? <span className="link" onClick={() => navigate("/signup")}>Sign up</span></p>
          </div>
        </div>

        <p className="copyright">Copyright © 2026 <b>AI-VMS</b> and its related entities. All Rights Reserved.</p>
      </div>
    </div>
  );
};

export default Login;