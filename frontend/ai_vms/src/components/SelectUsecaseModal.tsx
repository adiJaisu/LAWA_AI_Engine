import { useEffect, useState } from "react";
import { getAllUsecases, updateCamera } from "../api/camera.api";
import { getAuthToken } from "../api/auth.api";
import type { Usecase, CameraRow } from "../types/camera.types";

interface SelectUsecaseModalProps {
    isOpen: boolean;
    camera: CameraRow | null;
    onClose: () => void;
    onSuccess: (usecases: Usecase[]) => void;
}

const SelectUsecaseModal = ({
    isOpen,
    camera,
    onClose,
    onSuccess
}: SelectUsecaseModalProps) => {
    const token = getAuthToken();
    const [usecases, setUsecases] = useState<Usecase[]>([]);
    const [selectedIds, setSelectedIds] = useState<number[]>([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        if (!isOpen || !token) return;

        getAllUsecases()
            .then(res => {
                if (res.code === 200) {
                    setUsecases(res.usecaseDetails);
                }
            })
            .catch(console.error);
    }, [isOpen, token]);

    useEffect(() => {
        if (!isOpen || !camera) return;

        if (camera.usecases && camera.usecases.length > 0) {
            const existingIds = camera.usecases.map(
                (u) => u.usecaseId
            );
            setSelectedIds(existingIds);
        } else {
            setSelectedIds([]);
        }
    }, [isOpen, camera]);

    if (!isOpen || !camera) return null;

    const toggleUsecase = (id: number) => {
        setSelectedIds(prev =>
            prev.includes(id)
                ? prev.filter(x => x !== id)
                : [...prev, id]
        );
    };

    const handleSave = async () => {

        setLoading(true);

        try {
            await updateCamera({
                cameraId: camera.cameraId,
                usecaseIds: selectedIds
            });

            const selectedUsecases = usecases.filter(u =>
                selectedIds.includes(u.usecaseId)
            );

            onSuccess(selectedUsecases);
            onClose();
        } catch (err) {
            console.error("Update failed", err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="modal-overlay">
            <div className="modal usecase-modal">
                <div className="usecase-title">
                    Select Usecases for {camera.name}
                </div>

                <div className="usecase-list">
                    {usecases.length === 0 && (
                        <div className="usecase-empty">
                            No usecases available
                        </div>
                    )}

                    {usecases.map((u) => (
                        <label
                            key={u.usecaseId}
                            className="usecase-item"
                        >
                            <input
                                type="checkbox"
                                className="usecase-checkbox"
                                checked={selectedIds.includes(u.usecaseId)}
                                onChange={() => toggleUsecase(u.usecaseId)}
                            />
                            {u.usecaseName}
                        </label>
                    ))}
                </div>

                <div className="modal-actions">
                    <button
                        className="btn btn-secondary"
                        onClick={onClose}
                    >
                        Cancel
                    </button>

                    <button
                        className="btn btn-primary"
                        onClick={handleSave}
                        disabled={loading}
                    >
                        {loading ? "Saving..." : "Save"}
                    </button>
                </div>
            </div>
        </div>
    );
};

export default SelectUsecaseModal;
