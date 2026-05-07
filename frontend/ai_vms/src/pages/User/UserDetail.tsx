import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { DataTable } from "primereact/datatable";
import { Column } from "primereact/column";
import { InputSwitch } from 'primereact/inputswitch';
import Layout from "../../components/Layout";
import type { UserTableRow, UserFilterState, User } from "../../types/user.types";
import UserDetailsFilters from "./UserDetailsFilters";
import Pagination from "../../components/Pagination";
import ConfirmDeleteModal from "../../components/ConfirmDeleteModal";
import { getAllUsers, deleteUser, updateUser } from "../../api/user.api";
import { getUser } from "../../api/auth.api";
import { formatDateTime, toTargetTimeZoneInstant } from "../../utils/dateTimeUtils";
import { useToast } from "../../providers/toastProvider";
import { checkPermission } from "../../utils/permissionUtils";

const UserDetail = () => {
  const navigate = useNavigate();
  const { showToast } = useToast();
  const canCreate = checkPermission("user", "create");
  const canEdit = checkPermission("user", "update");
  const canDelete = checkPermission("user", "delete");
  const showActions = canEdit || canDelete;
  const [isFilterOpen, setIsFilterOpen] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [selectedUser, setSelectedUser] = useState<UserTableRow | null>(null);
  const [filteredUsers, setFilteredUsers] = useState<UserTableRow[]>([]);
  const [filters, setFilters] = useState<UserFilterState>({
    username: "",
    firstName: "",
    lastName: "",
    role: "",
    fromDate: null,
    toDate: null
  });

  const hasActiveFilters =
    !!filters.username ||
    !!filters.firstName ||
    !!filters.lastName ||
    !!filters.role ||
    filters.fromDate !== null ||
    filters.toDate !== null;
  const user = getUser();
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [roles, setRoles] = useState<string[]>([]);

  useEffect(() => {
    const fetchUsers = async () => {
      try {
        const response = await getAllUsers();
        setUsers(response.users);
        const uniqueRoles = Array.from(
          new Set(response.users.map(u => u.roleName))
        ).sort();
        setRoles(uniqueRoles);

      } catch (err) {
        setError("Failed to load users");
      } finally {
        setLoading(false);
      }
    };

    fetchUsers();
  }, []);

  const getDefaultFilters = (): UserFilterState => ({
    username: "",
    firstName: "",
    lastName: "",
    role: "",
    fromDate: null,
    toDate: null
  });

  const USER_DATA: UserTableRow[] = users.map((u) => ({
    id: u.userId,
    firstName: u.firstName,
    lastName: u.lastName,
    username: u.username,
    status: u.status,
    role: u.roleName,
    roleId: u.roleId,
    createdAt: u.createdAt
  }));

  useEffect(() => {
    setFilteredUsers(USER_DATA);
  }, [users]);

  const rowsPerPage = 10;
  const [currentPage, setCurrentPage] = useState(1);
  const totalPages = Math.ceil(filteredUsers.length / rowsPerPage);
  const paginatedUsers = filteredUsers.slice((currentPage - 1) * rowsPerPage, currentPage * rowsPerPage);

  // Apply filters
  const applyFilters = () => {
    const filtered = USER_DATA.filter(user => {
      if (
        filters.username &&
        !user.username.toLowerCase().includes(filters.username.toLowerCase())
      ) {
        return false;
      }
      if (
        filters.firstName &&
        !user.firstName.toLowerCase().includes(filters.firstName.toLowerCase())
      ) {
        return false;
      }
      if (
        filters.lastName &&
        !user.lastName.toLowerCase().includes(filters.lastName.toLowerCase())
      ) {
        return false;
      }

      if (filters.role && user.role !== filters.role) {
        return false;
      }

      if (filters.fromDate) {
        const userDate = new Date(user.createdAt).getTime();
        const from = toTargetTimeZoneInstant(filters.fromDate)?.getTime();
        if (typeof from === "number" && userDate < from) return false;
      }

      if (filters.toDate) {
        const userDate = new Date(user.createdAt).getTime();
        const to = toTargetTimeZoneInstant(filters.toDate)?.getTime();
        if (typeof to === "number" && userDate > to) return false;
      }

      return true;
    });

    setFilteredUsers(filtered);
    setIsFilterOpen(false);
  };


  const clearFilters = () => {
    setFilters({
      username: "",
      firstName: "",
      lastName: "",
      role: "",
      fromDate: null,
      toDate: null
    });

    setFilteredUsers(USER_DATA);
  };

  const handleResetFilters = () => {
    const emptyFilters = getDefaultFilters();

    setFilters(emptyFilters);
    setFilteredUsers(USER_DATA);

    console.log("Filters reset");
  };

  const handleStatusToggle = async (row: UserTableRow, value: boolean) => {
    if (!user) return;

    setUsers(prev =>
      prev.map(u =>
        u.userId === row.id
          ? { ...u, status: value ? 1 : 0 }
          : u
      )
    );

    try {
      await updateUser({
        userId: row.id,
        email: row.username,
        firstName: row.firstName,
        lastName: row.lastName,
        isActive: value,
        roleId: row.roleId
      });

      showToast({
        severity: "success",
        summary: "Status Updated",
        detail: `User has been ${value ? "activated" : "deactivated"}`,
        life: 3000
      });

    } catch (err) {
      setUsers(prev =>
        prev.map(u =>
          u.userId === row.id
            ? { ...u, status: row.status }
            : u
        )
      );

      showToast({
        severity: "error",
        summary: "Update Failed",
        detail: "Could not update user status",
        life: 3000
      });
    }
  };

  const handleEdit = (row: UserTableRow) => {
    navigate("/add-user", { state: { userData: row } });
  };

  const handleDeleteClick = (row: UserTableRow) => {
    setSelectedUser(row);
    setShowDeleteModal(true);
  };

  const confirmDelete = async () => {
    if (!selectedUser) return;

    try {

      await deleteUser(Number(selectedUser.id));

      showToast({
        severity: "success",
        summary: "User Deleted",
        detail: `${selectedUser.firstName} has been deleted successfully`,
        life: 2500
      });

      setUsers(prev =>
        prev.filter(u => u.userId !== selectedUser.id)
      );

    } catch (err: any) {
      showToast({
        severity: "error",
        summary: "Delete Failed",
        detail: err.message || "Unable to delete user",
        life: 3000
      });
    } finally {
      setShowDeleteModal(false);
      setSelectedUser(null);
    }
  };


  const cancelDelete = () => {
    setShowDeleteModal(false);
    setSelectedUser(null);
  };

  return (
    <Layout>
      <div className="page-container">
        <div className="form-wrapper">
          {/* ================= HEADER ================= */}
          <div className="page-header">
            <div className="header-info">
              <div className="header-title">USER DETAILS</div>
              <div className="header-subtitle">
                Detailed user information is displayed below
              </div>
            </div>

            <div className="action-buttons">
              {canCreate && (
                <button
                  className="icon-button"
                  onClick={() => navigate("/add-user")}
                  title="Add User"
                >
                  <span className="material-icons">add_circle</span>
                </button>
              )}

              <button
                className="icon-button"
                onClick={handleResetFilters}
                title="Reset Filters"
                // multi
              >
                <span className="material-icons">refresh</span>
              </button>

              <button
                className="icon-button"
                onClick={() => setIsFilterOpen(true)}
                title="Open Filters"
              >
                <span className="material-icons">manage_search</span>
              </button>
            </div>
          </div>

          {/* ================= FILTER OVERLAY ================= */}
          {isFilterOpen && (
            <div
              className="filter-overlay active"
              onClick={() => setIsFilterOpen(false)}
            />
          )}

          {/* ================= FILTER PANEL ================= */}
          <UserDetailsFilters
            isOpen={isFilterOpen}
            filters={filters}
            setFilters={setFilters}
            onClose={() => setIsFilterOpen(false)}
            onApply={applyFilters}
            onClear={clearFilters}
            roles={roles}
          />

          {/* ================= TABLE ================= */}
          <div className="form-content">
            {loading && <p>Loading users...</p>}
            {error && <p className="error-message">{error}</p>}
            <div className="custom-prime-table">
              <DataTable
                value={paginatedUsers}
                className=""
                scrollable
                scrollHeight="calc(100vh - 350px)"
              >
                <Column field="firstName" header="First Name" />
                <Column field="lastName" header="Last Name" />
                <Column field="username" header="Email" />
                {canEdit && (
                  <Column
                    header="Status"
                    body={(row: UserTableRow) => (
                      <InputSwitch
                        checked={row.status === 1}
                        disabled={Number(row.id) === Number(user?.id)}
                        onChange={(e) => handleStatusToggle(row, e.value)}
                      />
                    )}
                  />
                )}
                <Column field="role" header="Role" />
                <Column
                  header="Created At"
                  body={(row) => formatDateTime(row.createdAt)}
                />

                {showActions && (
                  <Column
                    header="Actions"
                    body={(row: UserTableRow) => (
                      <div className="action-buttons">
                        {canEdit && (
                          <button
                            className="action-btn edit"
                            title="Edit"
                            onClick={() => handleEdit(row)}
                          >
                            <span className="material-icons">edit</span>
                          </button>
                        )}

                        {canDelete && Number(row.id) !== Number(user?.id) && (
                          <button
                            className="action-btn delete"
                            title="Delete"
                            onClick={() => handleDeleteClick(row)}
                          >
                            <span className="material-icons">delete</span>
                          </button>
                        )}
                      </div>
                    )}
                  />
                )}

              </DataTable>
            </div>
          </div>
          <Pagination
            currentPage={currentPage}
            totalPages={totalPages}
            onPageChange={(page) => { setCurrentPage(page); }}
          />
        </div>
      </div>
      <ConfirmDeleteModal
        isOpen={showDeleteModal}
        itemName={selectedUser?.firstName}
        onConfirm={confirmDelete}
        onCancel={cancelDelete}
      />
    </Layout>
  );
};

export default UserDetail;
