import { useLocation, useNavigate } from "react-router-dom";
import "../styles/sidebar.css";
import type { NavItem } from "../types/sidebar.types";
import { checkPermission } from "../utils/permissionUtils";

const Sidebar = () => {
  const location = useLocation();
  const navigate = useNavigate();

  const navItems: NavItem[] = [
    ...(checkPermission("camera", "read")
      ? [
        {
          icon: "nest_cam_outdoor",
          label: "Camera",
          path: "/camera-details",
        },
      ]
      : []),

    ...(checkPermission("user", "read")
      ? [
        {
          icon: "people",
          label: "Users",
          path: "/users",
        },
      ]
      : []),

      {
    icon: "grid_view",
    label: "Multi Grid View",
    path: "/MultiGridView",
  },
  ];

  return (
    <nav className="nav-container">
      <div className="nav-items">
        {navItems.map((item) => (
          <div
            key={item.path}
            className={`nav-item ${location.pathname === item.path ? "active" : ""
              }`}
            onClick={(e) => {
              if (e.ctrlKey || e.metaKey || e.shiftKey) {
                window.open(item.path, "_blank");
                return;
              }
              navigate(item.path);
            }}
          >
            <span className="material-icons item-icon">{item.icon}</span>
            <span className="tooltip">{item.label}</span>
          </div>
        ))}
      </div>
    </nav>
  );
};

export default Sidebar;
