import "../styles/modal.css"
interface ConfirmDeleteModalProps {
  isOpen: boolean;
  title?: string;
  description?: string;
  itemName?: string;
  onConfirm: () => void;
  onCancel: () => void;
  confirmText?: string;
  cancelText?: string;
}

const ConfirmDeleteModal = ({
  isOpen,
  title = "Confirm Delete",
  description,
  itemName,
  onConfirm,
  onCancel,
  confirmText = "Delete",
  cancelText = "Cancel"
}: ConfirmDeleteModalProps) => {
  if (!isOpen) return null;

  return (
    <div className="modal-overlay">
      <div className="modal">
        <h3>{title}</h3>

        <p>
          {description ??
            <>
              Are you sure you want to delete{" "}
              {itemName && <strong>{itemName}</strong>}?
            </>
          }
        </p>

        <div className="modal-actions">
          <button className="btn btn-secondary" onClick={onCancel}>
            {cancelText}
          </button>

          <button className="btn btn-primary" onClick={onConfirm}>
            {confirmText}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ConfirmDeleteModal;
