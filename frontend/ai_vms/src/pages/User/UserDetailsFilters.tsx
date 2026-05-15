import type { UserFilterState } from "../../types/user.types";

interface UserFiltersProps {
  isOpen: boolean;
  filters: UserFilterState;
  setFilters: React.Dispatch<React.SetStateAction<UserFilterState>>;
  onClose: () => void;
  onApply: () => void;
  onClear: () => void;
  roles: string[];
}

const UserDetailsFilters = ({
  isOpen,
  filters,
  setFilters,
  onClose,
  onApply,
  onClear,
  roles
}: UserFiltersProps) => {

  return (
    <>
      {/* Overlay */}
      {isOpen && (
        <div className="filter-overlay active" onClick={onClose} />
      )}

      {/* Panel */}
      <div className={`filter-panel ${isOpen ? "active" : ""}`}>
        <div className="filter-header">
          <h2>Filter by</h2>
          <button className="icon-button-filter" onClick={onClose}>
            <span className="material-icons">close</span>
          </button>
        </div>


        <div className="form-section">
          <label className="segment-label">First Name</label>
          <div className="segment-style input-segment">
            <input
              className="segment-input"
              placeholder="Enter First Name"
              value={filters.firstName}
              onChange={(e) =>
                setFilters(f => ({
                  ...f,
                  firstName: e.target.value
                }))
              }
            />
          </div>
        </div>
        <div className="form-section">
          <label className="segment-label">Last Name</label>
          <div className="segment-style input-segment">
            <input
              className="segment-input"
              placeholder="Enter Last Name"
              value={filters.lastName}
              onChange={(e) =>
                setFilters(f => ({
                  ...f,
                  lastName: e.target.value
                }))
              }
            />
          </div>
        </div>

        <div className="form-section">
          <label className="segment-label">Email</label>
          <div className="segment-style input-segment">
            <input
              className="segment-input"
              placeholder="Enter Email"
              value={filters.username}
              onChange={(e) =>
                setFilters(f => ({
                  ...f,
                  username: e.target.value
                }))
              }
            />
          </div>
        </div>

        {/* Role */}
        <div className="form-section">
          <label className="segment-label">Role</label>
          <div className="segment-style dropdown-segment">
            <select
              className="segment-input"
              value={filters.role}
              onChange={(e) =>
                setFilters((f) => ({
                  ...f,
                  role: e.target.value
                }))
              }
            >
              <option value="">Select role</option>
              {roles.map((role) => (
                <option key={role} value={role}>
                  {role}
                </option>
              ))}
            </select>
            <span className="material-icons dropdown-arrow">
              arrow_drop_down
            </span>
          </div>
        </div>

        {/* Actions */}
        <div className="btn-container">
          <button className="btn btn-primary" onClick={onApply}>
            Apply Filters
          </button>
          <button className="btn btn-secondary" onClick={onClear}>
            Clear Filters
          </button>
        </div>
      </div>
    </>
  );
};

export default UserDetailsFilters;
