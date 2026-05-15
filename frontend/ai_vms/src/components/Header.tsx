import { useEffect, useState } from "react";
import { useAuth } from "../context/AuthContext";
import type { User } from "../types/auth.types";
import "../styles/header.css";
import { createBackgroundFrameService } from "../streaming/BackgroundFrameService";

const Header = () => {
  const frameService = createBackgroundFrameService({
    intervalMs: 333,
    width: 1920,
    height: 1080,
    mimeType: "image/jpeg",
    quality: 0.7,
    fieldName: "frame",
  });
  const [isDropdownOpen, setIsDropdownOpen] = useState<boolean>(false);
  const { user, logout } = useAuth();
  const [roleId, setRoleId] = useState<number | null>(null);
  useEffect(() => {
    const storedUser = sessionStorage.getItem("user");
    if (storedUser) {
      try {
        const parsedUser = JSON.parse(storedUser);
        setRoleId(parsedUser.role_id);
      } catch (error) {
        console.error("Invalid user data in sessionStorage");
      }
    }
  }, [user]);
  const toggleDropdown = () => {
    setIsDropdownOpen((prev) => !prev);
  };

  const handleLogout = async () => {
    frameService.stop();
    await logout();
  };

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as HTMLElement | null;
      if (isDropdownOpen && target && !target.closest(".user-menu-container")) {
        setIsDropdownOpen(false);
      }
    };
    document.addEventListener("click", handleClickOutside);
    return () => {
      document.removeEventListener("click", handleClickOutside);
    };
  }, [isDropdownOpen]);

  const getUserInitials = (user?: User | null): string => {
    if (user?.first_name) {
      const initials = (user.first_name[0] || "") + (user.last_name?.[0] || "");
      return initials.toUpperCase() || "AD";
    }
    return "AD";
  };

  return (
    <header className="header">
      <div className="header-left">
        <div className="logo-container">
          <div className="hcl-logo">
            <span className="hcl-text">HCLTech</span>
            <span className="hcl-tagline">Supercharging Progress™</span>
          </div>
          <div className="divider-line"></div>
          <div className="aivms-logo">
            <span className="aivms-text">AIVMS</span>
          </div>
        </div>
      </div>

      <div className="header-right">
        <div className="notifications">
          <button className="icon-btn notification-btn" disabled>
            <span className="material-icons bell-icon">notifications</span>
          </button>
        </div>

        <div className="user-menu-container">
          <button className="user-profile-btn" onClick={toggleDropdown}>
            <div className="user-avatar">
              <span style={{ fontSize: "14px", fontWeight: "600" }}>
                {getUserInitials(user)}
              </span>
            </div>
            <div className="user-info">
              <span className="user-name">
                {user ? `${user.first_name} ${user.last_name}` : "User"}
              </span>
            </div>
            <span
              className="material-icons dropdown-arrow"
              style={{ transform: isDropdownOpen ? "rotate(180deg)" : "rotate(0deg)" }}
            >
              expand_more
            </span>
          </button>

          <div className={`user-dropdown ${isDropdownOpen ? "active" : ""}`}>
            <div className="dropdown-header">
              <div className="dropdown-avatar">
                <span style={{ fontSize: "20px", fontWeight: "600" }}>
                  {getUserInitials(user)}
                </span>
              </div>
              <div>
                <div className="dropdown-name">
                  {user ? `${user.first_name} ${user.last_name}` : "User"}
                </div>
                <div className="dropdown-email">
                  {user?.email || "user@ai_vms.com"}
                </div>

                {roleId !== null && (
                  <div className="dropdown-role">
                    {roleId === 1 ? "Superuser" :
                      roleId === 2 ? "Supervisor" :
                        roleId === 3 ? "Branch Manager" :
                          roleId === 4 ? "Claim Associate" : null}
                  </div>
                )}
              </div>
            </div>

            <div className="dropdown-divider" />

            <button className="dropdown-item logout" onClick={handleLogout}>
              <span className="material-icons">logout</span>
              <span>Logout</span>
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;