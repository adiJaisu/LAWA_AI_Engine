import { useEffect, useRef, useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { Toast } from "primereact/toast";

import Layout from "../../components/Layout";
import type { UserTableRow } from "../../types/user.types";

import { getAllRoles } from "../../api/role.api";
import { addUser, updateUser } from "../../api/user.api";
import type { Role } from "../../types/role.types";

import "../../styles/global.css";
import "../../styles/components.css";
import "../../styles/camera.css";
import "../../styles/layout.css";
import "../../styles/prime_table.css";

const EMPTY_USER: UserTableRow = {
  id: "",
  firstName: "",
  lastName: "",
  username: "",
  status: 1,
  role: "",
  createdAt: ""
};

const AddUser = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const toastRef = useRef<Toast>(null);

  const editUser = location.state?.userData as UserTableRow | undefined;
  const isEditMode = Boolean(editUser);

  const [formData, setFormData] = useState<UserTableRow>(EMPTY_USER);
  const [loadingMeta, setLoadingMeta] = useState(true);

  const [roles, setRoles] = useState<Role[]>([]);
  const [selectedRoleId, setSelectedRoleId] = useState<number | undefined>();

  /* -------------------- helpers -------------------- */

  const updateForm = (patch: Partial<UserTableRow>) => {
    setFormData(prev => ({ ...prev, ...patch }));
  };

  const isValidEmail = (email: string) =>
    /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);

  const showError = (msg: string) =>
    toastRef.current?.show({
      severity: "error",
      summary: "Validation Error",
      detail: msg,
      life: 3000
    });

  const showSuccess = (msg: string) =>
    toastRef.current?.show({
      severity: "success",
      summary: "Success",
      detail: msg,
      life: 2500
    });

  const validateForm = () => {
    if (!formData.firstName.trim()) {
      showError("First Name is required");
      return false;
    }
    if (!/^[a-zA-Z\s]+$/.test(formData.firstName)) {
      showError("First Name must contain only alphabets");
      return false;
    }
    if (!formData.lastName.trim()) {
      showError("Last Name is required");
      return false;
    }
    if (!/^[a-zA-Z\s]+$/.test(formData.lastName)) {
      showError("Last Name must contain only alphabets");
      return false;
    }
    if (!formData.username.trim()) {
      showError("Email is required");
      return false;
    }
    if (!isValidEmail(formData.username)) {
      showError("Please enter a valid email address");
      return false;
    }
    if (!formData.role) {
      showError("Role is required");
      return false;
    }
    return true;
  };

  const getRoleId = (roleName: string) =>
    roles.find(r => r.roleName === roleName)?.roleId;

  /* -------------------- role change handler -------------------- */

  const handleRoleChange = (newRole: string) => {
    const roleObj = roles.find(r => r.roleName === newRole);
    const newRoleId = roleObj?.roleId;

    updateForm({ role: newRole });
    setSelectedRoleId(newRoleId);
  };

  /* -------------------- effects -------------------- */

  useEffect(() => {
    getAllRoles().then((rolesRes) => {
      setRoles(rolesRes.roleDetails);
      setLoadingMeta(false);
    });
  }, []);

  useEffect(() => {
    if (editUser) {
      setFormData(editUser);
      const roleObj = roles.find(r => r.roleName === editUser.role);
      if (roleObj) setSelectedRoleId(roleObj.roleId);
    }
  }, [editUser, roles]);

  /* -------------------- submit -------------------- */

  const handleSubmit = async () => {
    if (!validateForm()) return;

    const roleId = getRoleId(formData.role);
    if (!roleId) { showError("Invalid role selected"); return; }

    try {
      if (isEditMode && editUser) {
        const response = await updateUser({
          userId: editUser.id,
          email: formData.username,
          firstName: formData.firstName,
          lastName: formData.lastName,
          isActive: formData.status === 1,
          roleId,
        });
        if (response.code === 200) {
          showSuccess("User updated successfully");
          setTimeout(() => navigate("/users"), 1200);
        } else {
          showError(response.message);
        }
      } else {
        const response = await addUser({
          email: formData.username,
          firstName: formData.firstName,
          lastName: formData.lastName,
          isActive: true,
          roleId,
        });
        if (response.code === 200) {
          showSuccess("User created successfully");
          setTimeout(() => navigate("/users"), 1200);
        } else {
          showError(response.message);
        }
      }
    } catch (err: any) {
      showError(err.message || "Something went wrong");
    }
  };

  /* -------------------- render -------------------- */

  return (
    <Layout>
      <Toast ref={toastRef} position="top-right" />
      <div className="page-container">
        <div className="page-header">
          <div className="header-left">
            <div className="header-title">
              {isEditMode ? "EDIT USER" : "ADD USER"}
            </div>
            <div className="header-subtitle">
              {isEditMode ? "Update user details" : "Complete all required fields"}
            </div>
          </div>
          <div className="action-buttons">
            <button className="icon-button" onClick={() => navigate(-1)}>
              <span className="material-icons">arrow_back</span>
              Back
            </button>
          </div>
        </div>

        <div className="form-wrapper">
          <div className="form-content">
            <div className="form-section">

              {/* Row 1: First Name / Last Name */}
              <div className="form-row">
                <div className="form-group">
                  <label className="segment-label">
                    First Name <span className="required-star">*</span>
                  </label>
                  <div className="segment-style input-segment">
                    <input
                      className="segment-input"
                      value={formData.firstName}
                      onChange={e => {
                        if (/^[a-zA-Z\s]*$/.test(e.target.value))
                          updateForm({ firstName: e.target.value });
                      }}
                    />
                  </div>
                </div>

                <div className="form-group">
                  <label className="segment-label">
                    Last Name <span className="required-star">*</span>
                  </label>
                  <div className="segment-style input-segment">
                    <input
                      className="segment-input"
                      value={formData.lastName}
                      onChange={e => {
                        if (/^[a-zA-Z\s]*$/.test(e.target.value))
                          updateForm({ lastName: e.target.value });
                      }}
                    />
                  </div>
                </div>
              </div>

              {/* Row 2: Email / Role */}
              <div className="form-row">
                <div className="form-group">
                  <label className="segment-label">
                    Email <span className="required-star">*</span>
                  </label>
                  <div className="segment-style input-segment">
                    <input
                      type="email"
                      className="segment-input"
                      value={formData.username}
                      onChange={e => updateForm({ username: e.target.value })}
                    />
                  </div>
                </div>

                <div className="form-group">
                  <label className="segment-label">
                    Role <span className="required-star">*</span>
                  </label>
                  <div className="segment-style dropdown-segment">
                    <select
                      className="segment-input"
                      value={formData.role}
                      disabled={loadingMeta || roles.length === 0}
                      onChange={e => handleRoleChange(e.target.value)}
                    >
                      <option value="">Select role</option>
                      {roles.map(role => (
                        <option key={role.roleId} value={role.roleName}>
                          {role.roleName}
                        </option>
                      ))}
                    </select>
                    <span className="material-icons dropdown-arrow">
                      arrow_drop_down
                    </span>
                  </div>
                </div>
              </div>

              {/* Status Row */}
              {isEditMode && (
                <div className="form-row">
                  <div className="form-group">
                    <label className="segment-label">Status</label>
                    <div className="segment-style dropdown-segment">
                      <select
                        className="segment-input"
                        value={formData.status}
                        onChange={e => updateForm({ status: Number(e.target.value) })}
                      >
                        <option value={1}>Active</option>
                        <option value={0}>Inactive</option>
                      </select>
                      <span className="material-icons dropdown-arrow">
                        arrow_drop_down
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="form-footer">
              <button className="btn btn-secondary" onClick={() => navigate(-1)}>
                Cancel
              </button>
              <button className="btn btn-primary" onClick={handleSubmit}>
                {isEditMode ? "Update User" : "Add User"}
              </button>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default AddUser;