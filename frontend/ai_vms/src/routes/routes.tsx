import { Routes, Route, Navigate } from "react-router-dom";
import ProtectedRoute from "../components/ProtectedRoute";
import PermissionRoute from "../components/PermissionRoute";
import Login from "../pages/Login";
import Signup from "../pages/Signup";
import CameraDetails from "../pages/Camera/CameraDetails";
import AddCamera from "../pages/Camera/AddCamera";
import AddUser from "../pages/User/AddUser";
import UserDetail from "../pages/User/UserDetail";
import ROIEditor from "../pages/ROI/ROIEditor";
import MultiGridView from "../pages/MultiGridView/MultiGridView";

const AppRoutes = () => {
  return (
    <Routes>
      {/* Public Routes */}
      <Route path="/login" element={<Login />} />
      <Route path="/signup" element={<Signup />} />

      {/* Camera Routes */}
      <Route
        path="/camera-details"
        element={
          <ProtectedRoute>
            <PermissionRoute resource="camera" scope="read">
              <CameraDetails />
            </PermissionRoute>
          </ProtectedRoute>
        }
      />

      <Route
        path="/add-camera"
        element={
          <ProtectedRoute>
            <PermissionRoute resource="camera" scope="create">
              <AddCamera />
            </PermissionRoute>
          </ProtectedRoute>
        }
      />

      <Route
        path="/roi-editor"
        element={
          <ProtectedRoute>
            <PermissionRoute resource="camera" scope="read">
              <ROIEditor />
            </PermissionRoute>
          </ProtectedRoute>
        }
      />

      {/* User Routes */}
      <Route
        path="/users"
        element={
          <ProtectedRoute>
            <PermissionRoute resource="user" scope="read">
              <UserDetail />
            </PermissionRoute>
          </ProtectedRoute>
        }
      />

      <Route
        path="/add-user"
        element={
          <ProtectedRoute>
            <PermissionRoute resource="user" scope="create">
              <AddUser />
            </PermissionRoute>
          </ProtectedRoute>
        }
      />
      <Route
  path="/MultiGridView"
  element={
    <ProtectedRoute>
      <MultiGridView />
    </ProtectedRoute>
  }
/>

      {/* Default Redirect */}
      <Route path="*" element={<Navigate to="/camera-details" replace />} />
    </Routes>
  );
};

export default AppRoutes;