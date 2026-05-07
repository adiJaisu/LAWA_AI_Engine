interface PaginationProps {
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
}

const Pagination = ({
  currentPage,
  totalPages,
  onPageChange
}: PaginationProps) => {
  const getVisiblePages = () => {
    const pages: (number | "...")[] = [];

    if (totalPages <= 5) {
      return Array.from({ length: totalPages }, (_, i) => i + 1);
    }

    pages.push(1);

    if (currentPage > 3) {
      pages.push("...");
    }

    const start = Math.max(2, currentPage - 1);
    const end = Math.min(totalPages - 1, currentPage + 1);

    for (let i = start; i <= end; i++) {
      pages.push(i);
    }

    if (currentPage < totalPages - 2) {
      pages.push("...");
    }

    pages.push(totalPages);

    return pages;
  };

  // Removed: if (totalPages <= 1) return null;
  // Pagination should always be visible even with 1 page

  return (
    <div className="form-footer">
      <div className="pagination-container">
        <ul className="grid-pagination primary">
          {/* Previous */}
          <li className="page-item">
            <button
              className="page-link"
              disabled={currentPage === 1}
              onClick={() => onPageChange(currentPage - 1)}
              title="Previous Page"
            >
              <span className="material-icons">chevron_left</span>
            </button>
          </li>

          {/* Page numbers */}
          {getVisiblePages().map((p, index) => (
            <li key={index} className="page-item">
              {p === "..." ? (
                <span className="page-link disabled">…</span>
              ) : (
                <button
                  className={`page-link ${currentPage === p ? "active" : ""}`}
                  onClick={() => onPageChange(p as number)}
                >
                  {p}
                </button>
              )}
            </li>
          ))}

          {/* Next */}
          <li className="page-item">
            <button
              className="page-link"
              disabled={currentPage === totalPages}
              onClick={() => onPageChange(currentPage + 1)}
              title="Next Page"
            >
              <span className="material-icons">chevron_right</span>
            </button>
          </li>
        </ul>

        <div className="pagination-info">
          Showing page {currentPage} of {totalPages}
        </div>
      </div>
    </div>
  );
};

export default Pagination;