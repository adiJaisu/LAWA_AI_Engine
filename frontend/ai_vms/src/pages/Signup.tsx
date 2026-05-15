import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import "../styles/login.css";
import ai_vmsLogo from "../../src/assets/images/AIVMS-logo.svg";

const Signup = () => {
    const navigate = useNavigate();
    const { signup } = useAuth();

    const [email, setEmail] = useState("");
    const [password, setPassword] = useState("");
    const [firstName, setFirstName] = useState("");
    const [lastName, setLastName] = useState("");
    const [error, setError] = useState<string | null>(null);
    const [signingUp, setSigningUp] = useState<boolean>(false);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setSigningUp(true);
        setError(null);

        try {
            await signup(email, password, firstName, lastName);
            navigate("/login");
        } catch (err: any) {
            setError(err.message || "Signup failed. Try again.");
        } finally {
            setSigningUp(false);
        }
    };

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
                        <p className="welcome-text">Start your journey</p>
                        <h2>Create Account</h2>
                        <p>Fill in the details to get started with AIVMS</p>
                    </div>

                    {error && <div className="error-message">{error}</div>}

                    <form onSubmit={handleSubmit} className="login-form">
                        <div className="form-group">
                            <label htmlFor="firstName">First Name</label>
                            <input
                                type="text"
                                id="firstName"
                                value={firstName}
                                onChange={(e) => setFirstName(e.target.value)}
                                required
                            />
                        </div>
                        <div className="form-group">
                            <label htmlFor="lastName">Last Name</label>
                            <input
                                type="text"
                                id="lastName"
                                value={lastName}
                                onChange={(e) => setLastName(e.target.value)}
                                required
                            />
                        </div>
                        <div className="form-group">
                            <label htmlFor="email">Email</label>
                            <input
                                type="email"
                                id="email"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                required
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
                            />
                        </div>
                        <button type="submit" className="btn-login" disabled={signingUp}>
                            {signingUp ? "Signing up..." : "Sign up"}
                        </button>
                    </form>

                    <div className="login-form-footer">
                        <p>Already have an account? <span className="link" onClick={() => navigate("/login")}>Login</span></p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Signup;
