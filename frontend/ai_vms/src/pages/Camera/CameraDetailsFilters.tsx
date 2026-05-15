import type { FilterState } from "../../types/camera.types";

interface CameraDetailsFiltersProps {
  isOpen: boolean;
  filters: FilterState;
  setFilters: React.Dispatch<React.SetStateAction<FilterState>>;
  onClose: () => void;
  onApply: () => void;
  onClear: () => void;
}

const CameraDetailsFilters = ({
  isOpen,
  filters,
  setFilters,
  onClose,
  onApply,
  onClear
}: CameraDetailsFiltersProps) => {

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

        {/* Camera Name */}
        <div className="form-section">
          <label className="segment-label">Name</label>
          <div className="segment-style input-segment">
            <input
              className="segment-input"
              placeholder="Enter camera name"
              value={filters.name}
              onChange={(e) =>
                setFilters((f: FilterState) => ({
                  ...f,
                  name: e.target.value
                }))
              }
            />
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

export default CameraDetailsFilters;