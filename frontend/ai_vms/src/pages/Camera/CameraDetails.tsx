import { useState, useEffect, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { DataTable } from "primereact/datatable";
import { Tooltip } from "primereact/tooltip";
import { getAllCameras, deleteCamera, updateCamera } from "../../api/camera.api";
import { Column } from "primereact/column";
import Layout from "../../components/Layout";
import type { FilterState, CameraRow, Usecase } from "../../types/camera.types";
import CameraDetailsFilters from "./CameraDetailsFilters";
import SelectUsecaseModal from "../../components/SelectUsecaseModal";
import Pagination from "../../components/Pagination";
import ConfirmDeleteModal from "../../components/ConfirmDeleteModal";
import { useToast } from "../../providers/toastProvider";
import { checkPermission } from "../../utils/permissionUtils";

const formatDecodingResource = (value?: string) => {
  if (!value) return "CPU";
  if (value.toLowerCase() === "vaapi") return "VAAPI";
  return value.toUpperCase();
};

const formatDecodingPipeline = (value?: string) => {
  if (!value) return "FFmpeg";
  const normalized = value.toLowerCase();
  if (normalized === "opencv") return "OpenCV";
  if (normalized === "gstreamer") return "GStreamer";
  return "FFmpeg";
};

const formatDockerMode = (value?: string) => {
  return value?.toLowerCase() === "build" ? "Build" : "Load";
};

const CameraDetails = () => {
  const navigate = useNavigate();
  const { showToast } = useToast();
  const canUpdate = checkPermission("camera", "update");
  const canDelete = checkPermission("camera", "delete");
  const showActions = canUpdate || canDelete;

  const [cameras, setCameras] = useState<CameraRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [isFilterOpen, setIsFilterOpen] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [selectedCamera, setSelectedCamera] = useState<CameraRow | null>(null);

  const [showUsecaseModal, setShowUsecaseModal] = useState(false);
  const [selectedCameraForUsecase, setSelectedCameraForUsecase] = useState<CameraRow | null>(null);

  const [draftFilters, setDraftFilters] = useState<FilterState>({
    name: "",
    workstation: "",
    branch: ""
  });

  const [appliedFilters, setAppliedFilters] = useState<FilterState>({
    name: "",
    workstation: "",
    branch: ""
  });


  const filteredCameras = useMemo(() => {
    return cameras.filter(camera => {
      if (appliedFilters.name && !camera.name.toLowerCase().includes(appliedFilters.name.toLowerCase())) {
        return false;
      }
      return true;
    });
  }, [cameras, appliedFilters]);

  const rowsPerPage = 10;
  const [currentPage, setCurrentPage] = useState(1);

  const totalPages = Math.ceil(filteredCameras.length / rowsPerPage);

  const paginatedData = filteredCameras.slice(
    (currentPage - 1) * rowsPerPage,
    currentPage * rowsPerPage
  );

  useEffect(() => {
    setLoading(true);
    getAllCameras()
      .then(res => {
        const mapped = res.cameras.map(c => ({
          id: c.cameraId,
          cameraId: c.cameraId,
          name: c.name,
          type: c.type,
          status: (c.status === 1 ? "Active" : "Inactive") as "Active" | "Inactive",
          rtspUrl: c.rtspUrl ?? "",
          resolution: c.resolution,
          codec: c.codec || "",
          fps: c.fps || "",
          height: c.height,
          resolutionWidth: c.resolutionWidth,
          resolutionHeight: c.resolutionHeight,
          decodingResource: c.decodingResource || "cpu",
          decodingPipeline: c.decodingPipeline || "ffmpeg",
          dockerMode: c.dockerMode || "load",
          createdAt: c.createdAt,
          updatedAt: c.updatedAt,
          roi: c.roi,
          usecases: c.usecases || []
        }));
        setCameras(mapped);
      })
      .catch(err => {
        console.error("getAllCameras failed", err);
        setError(err.message || "Failed to fetch cameras");
      })
      .finally(() => setLoading(false));
  }, []);

  const applyFilters = () => {
    setAppliedFilters(draftFilters);
    setIsFilterOpen(false);
  };

  const clearFilters = () => {
    const emptyFilters = { name: "", workstation: "", branch: "" };
    setDraftFilters(emptyFilters);
    setAppliedFilters(emptyFilters);
  };

  const handleResetFilters = () => {
    const emptyFilters: FilterState = { name: "", workstation: "", branch: "" };
    setDraftFilters(emptyFilters);
    setAppliedFilters(emptyFilters);
  };

  const handleOpenFilters = () => {
    setDraftFilters(appliedFilters);
    setIsFilterOpen(true);
  };

  const handleCloseFilters = () => {
    setDraftFilters(appliedFilters);
    setIsFilterOpen(false);
  };

  const handleSelectUsecase = (row: CameraRow) => {
    setSelectedCameraForUsecase(row);
    setShowUsecaseModal(true);
  };

  const handleUsecaseSuccess = (updatedUsecases: Usecase[]) => {
    setCameras(prev =>
      prev.map(cam =>
        cam.cameraId === selectedCameraForUsecase?.cameraId
          ? { ...cam, usecases: updatedUsecases }
          : cam
      )
    );
    showToast({
      severity: "success",
      summary: "Usecase updated",
      detail: `Camera: ${selectedCameraForUsecase?.name} has been updated successfully`,
      life: 2500
    });
  };

  const handleEdit = (row: CameraRow) => {
    navigate("/add-camera", { state: { camera: row } });
  };

  const handleROIClick = (row: CameraRow) => {
    navigate(`/roi-editor?cameraId=${row.cameraId}`, { state: { camera: row } });
  };

  const handleDeleteClick = (row: CameraRow) => {
    setSelectedCamera(row);
    setShowDeleteModal(true);
  };

  const toggleStatus = async (id: number) => {
    const camera = cameras.find(c => c.id === id);
    if (!camera) return;

    const nextStatus = camera.status === "Active" ? "Inactive" : "Active";

    try {
      const res = await updateCamera({
        cameraId: camera.cameraId || camera.id,
        status: nextStatus === "Active" ? 1 : 0
      });

      if (res && res.code === 200) {
        setCameras(prev =>
          prev.map(c => (c.id === id ? { ...c, status: nextStatus } : c))
        );
        showToast({
          severity: "success",
          summary: "Status Updated",
          detail: `${camera.name} is now ${nextStatus}`,
          life: 2500
        });
      } else {
        showToast({
          severity: "error",
          summary: "Update Failed",
          detail: res?.message || "Unable to update camera status",
          life: 3000
        });
      }
    } catch (err: any) {
      showToast({
        severity: "error",
        summary: "Update Failed",
        detail: err.message || "Unable to update camera status",
        life: 3000
      });
    }
  };

  const confirmDelete = async () => {
    if (!selectedCamera) return;

    try {
      await deleteCamera(selectedCamera.cameraId || selectedCamera.id);

      showToast({
        severity: "success",
        summary: "Camera Deleted",
        detail: `${selectedCamera.name} has been deleted successfully`,
        life: 2500
      });

      setCameras(prev =>
        prev.filter(c => c.id !== selectedCamera.id)
      );

    } catch (err: any) {
      showToast({
        severity: "error",
        summary: "Delete Failed",
        detail: err.message || "Unable to delete camera",
        life: 3000
      });
    } finally {
      setShowDeleteModal(false);
      setSelectedCamera(null);
    }
  };

  const cancelDelete = () => {
    setShowDeleteModal(false);
    setSelectedCamera(null);
  };

  return (
    <Layout>
      <Tooltip target=".usecase-tooltip" position="bottom" />
      <div className="page-container">
        <div className="form-wrapper">
          <div className="page-header">
            <div className="header-info">
              <div className="header-title">CAMERA DETAILS</div>
              <div className="header-subtitle">
                Detailed camera information is displayed below
              </div>
            </div>

            <div className="action-buttons">
              <button
                className="icon-button"
                onClick={() => navigate("/add-camera")}
                title="Add camera"
              >
                <span className="material-icons">add_circle</span>
              </button>

              <button
                className="icon-button"
                onClick={handleResetFilters}
                title="Reset Filters"
                // disabled={!hasActiveFilters}
              >
                <span className="material-icons">refresh</span>
              </button>

              <button
                className="icon-button"
                onClick={handleOpenFilters}
                title="Open Filters"
              >
                <span className="material-icons">manage_search</span>
              </button>
            </div>
          </div>

          <CameraDetailsFilters
            isOpen={isFilterOpen}
            filters={draftFilters}
            setFilters={setDraftFilters}
            onClose={handleCloseFilters}
            onApply={applyFilters}
            onClear={clearFilters}
          />

          <div className="form-content">
            {loading && <div className="loading">Loading cameras...</div>}
            {error && <div className="error-message">{error}</div>}
            <div className="custom-prime-table">
              <DataTable
                value={paginatedData}
                scrollable
                scrollHeight="calc(100vh - 350px)"
                emptyMessage={appliedFilters.name ? "No cameras match the selected filter" : "No cameras available"}
              >
                <Column field="name" header="Name" />
                <Column field="type" header="Type" />

                {canUpdate && (
                  <Column
                    header="Status"
                    headerStyle={{ width: '80px', textAlign: 'center' }}
                    bodyStyle={{ width: '80px', textAlign: 'center' }}
                    body={(row: CameraRow) => (
                      <div style={{ display: "flex", alignItems: "center", justifyContent: 'center', gap: "6px" }}>
                        <div
                          className={`toggle-segment ${row.status === "Active" ? "active" : ""}`}
                          onClick={() => toggleStatus(row.id)}
                          role="switch"
                          aria-checked={row.status === "Active"}
                          style={{ cursor: "pointer", width: '44px' }}
                        >
                          <div className="toggle-switch" />
                        </div>
                      </div>
                    )}
                  />
                )}

                <Column
                  header="RTSP URL"
                  body={(row: CameraRow) => (
                    <span>{row.type === 'rtsp' ? row.rtspUrl : 'NA'}</span>
                  )}
                />
                <Column
                  header="Resolution"
                  body={(row: CameraRow) => (
                    <span>
                      {row.resolutionWidth && row.resolutionHeight
                        ? `${row.resolutionWidth} x ${row.resolutionHeight}`
                        : "NA"}
                    </span>
                  )}
                />
                <Column
                  header="Height (m)"
                  body={(row: CameraRow) => (
                    <span>{row.height ?? "NA"}</span>
                  )}
                />
                <Column
                  header="Use Cases"
                  body={(row: CameraRow) => (
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                      {row.usecases && row.usecases.length > 0 ? (
                        row.usecases.map((u, i) => (
                          <span key={i} style={{
                          }}>
                            {u.usecaseName}
                          </span>
                        ))
                      ) : (
                        <span style={{ color: '#999', fontSize: '11px' }}>None</span>
                      )}
                    </div>
                  )}
                />
                <Column field="codec" header="CODEC" />
                <Column field="fps" header="FPS" />
                <Column
                  header="Decoding Resource"
                  body={(row: CameraRow) => formatDecodingResource(row.decodingResource)}
                />
                <Column
                  header="Decoding Pipeline"
                  body={(row: CameraRow) => formatDecodingPipeline(row.decodingPipeline)}
                />
                <Column
                  header="Docker Mode"
                  body={(row: CameraRow) => formatDockerMode(row.dockerMode)}
                />

                {showActions && (
                  <Column
                    header="Actions"
                    body={(row: CameraRow) => (
                      <div className="action-buttons">
                        {canUpdate && (
                          <button
                            className="action-btn usecase-tooltip edit"
                            title="Manage Usecases"
                            data-pr-tooltip={
                              row.usecases && row.usecases.length > 0
                                ? row.usecases.map(u => u.usecaseName).join("\n")
                                : "No usecases assigned"
                            }
                            onClick={() => handleSelectUsecase(row)}
                          >
                            <span className="material-icons">checklist</span>
                          </button>
                        )}

                        {canUpdate && (
                          <button
                            className="action-btn edit"
                            title="ROI"
                            onClick={() => handleROIClick(row)}
                          >
                            <span className="material-icons">format_shapes</span>
                          </button>
                        )}

                        {canUpdate && (
                          <button
                            className="action-btn edit"
                            title="Edit"
                            onClick={() => handleEdit(row)}
                          >
                            <span className="material-icons">edit</span>
                          </button>
                        )}

                        {canDelete && (
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

      <SelectUsecaseModal
        isOpen={showUsecaseModal}
        camera={selectedCameraForUsecase}
        onClose={() => setShowUsecaseModal(false)}
        onSuccess={handleUsecaseSuccess}
      />

      <ConfirmDeleteModal
        isOpen={showDeleteModal}
        itemName={selectedCamera?.name}
        onConfirm={confirmDelete}
        onCancel={cancelDelete}
      />
    </Layout>
  );
};

export default CameraDetails;
