// import { useState, useEffect, useRef } from 'react';
// import { useNavigate } from 'react-router-dom';
// import { useLocation } from 'react-router-dom';
// import { Toast } from "primereact/toast";
// import Layout from '../../components/Layout';
// import "../../styles/camera.css";
// import type { CameraFormData } from '../../types/camera.types';
// import { addCamera, updateCamera } from "../../api/camera.api";
// import { useToast } from '../../providers/toastProvider';

// const normalizeDecodingResource = (value?: string) => {
//   const normalized = (value || "").toLowerCase();
//   return normalized === "gpu" || normalized === "vaapi" ? normalized : "cpu";
// };

// const normalizeDecodingPipeline = (value?: string) => {
//   const normalized = (value || "").toLowerCase();
//   return normalized === "opencv" || normalized === "gstreamer" ? normalized : "ffmpeg";
// };

// const normalizeDockerMode = (value?: string) => {
//   const normalized = (value || "").toLowerCase();
//   return normalized === "build" ? "build" : "load";
// };

// const AddCamera = () => {
//   const navigate = useNavigate();
//   const location = useLocation();
//   const { showToast } = useToast();
//   const toastRef = useRef<Toast>(null);
//   const editCamera = location.state?.camera;
//   const isEditMode = Boolean(editCamera && (editCamera.cameraId || editCamera.id));

//   const [formData, setFormData] = useState<CameraFormData>({
//     name: '',
//     type: '',
//     resolution: '',
//     fps: '',
//     codec: '',
//     rtspUrl: '',
//     height: '',
//     resolutionWidth: '',
//     resolutionHeight: '',
//     decodingResource: 'cpu',
//     decodingPipeline: 'ffmpeg',
//     dockerMode: 'load',
//     status: 1
//   });

//   /* -------------------- effects -------------------- */

//   useEffect(() => {
//     if (!editCamera) return;

//     setFormData(prev => ({
//       ...prev,
//       name: editCamera.name || "",
//       type: editCamera.type || "",
//       rtspUrl: editCamera.rtspUrl || "",
//       resolution: editCamera.resolution || "",
//       codec: editCamera.codec || "",
//       fps: editCamera.fps || "",
//       height: editCamera.height != null ? String(editCamera.height) : "",
//       resolutionWidth: editCamera.resolutionWidth != null ? String(editCamera.resolutionWidth) : "",
//       resolutionHeight: editCamera.resolutionHeight != null ? String(editCamera.resolutionHeight) : "",
//       decodingResource: normalizeDecodingResource(editCamera.decodingResource),
//       decodingPipeline: normalizeDecodingPipeline(editCamera.decodingPipeline),
//       dockerMode: normalizeDockerMode(editCamera.dockerMode),
//       status: editCamera.status === "Active" || editCamera.status === 1 ? 1 : 0
//     }));
//   }, [editCamera]);

//   /* -------------------- helpers -------------------- */

//   const showError = (msg: string) =>
//     toastRef.current?.show({
//       severity: "error",
//       summary: "Validation Error",
//       detail: msg,
//       life: 3000
//     });

//   const parseOptionalNumber = (value: string) => {
//     const trimmed = value.trim();
//     return trimmed === "" ? undefined : Number(trimmed);
//   };

//   /* -------------------- validation -------------------- */

//   const validateForm = (): boolean => {
//     if (!formData.name.trim()) {
//       showError("Camera name is required");
//       return false;
//     }

//     if (!formData.codec) {
//       showError("CODEC is required");
//       return false;
//     }

//     // if (!formData.resolution) {
//     //   showError("Resolution is required");
//     //   return false;
//     // }

//     if (!formData.fps || String(formData.fps).trim() === "") {
//       showError("Frame rate is required");
//       return false;
//     }
//     if (isNaN(Number(formData.fps)) || Number(formData.fps) <= 0) {
//       showError("Frame rate must be a valid positive number");
//       return false;
//     }

//     if (!formData.type) {
//       showError("Camera type is required");
//       return false;
//     }

//     for (const [label, value] of [
//       ["Height", formData.height],
//       ["Resolution width", formData.resolutionWidth],
//       ["Resolution height", formData.resolutionHeight]
//     ] as const) {
//       if (value.trim() !== "" && (isNaN(Number(value)) || Number(value) <= 0)) {
//         showError(`${label} must be a valid positive number`);
//         return false;
//       }
//     }

//     if (formData.type === 'rtsp') {
//       if (!formData.rtspUrl?.trim()) {
//         showError("RTSP URL is required");
//         return false;
//       }
//       if (!formData.rtspUrl.startsWith("rtsp://")) {
//         showError("RTSP URL must start with rtsp://");
//         return false;
//       }
//     }

//     return true;
//   };

//   /* -------------------- submit -------------------- */

//   const handleSubmit = () => {
//     if (!validateForm()) return;

//     (async () => {
//       try {
//         if (isEditMode && editCamera) {
//           const payload = {
//             cameraId: editCamera.cameraId || editCamera.id,
//             name: formData.name,
//             type: formData.type,
//             rtspUrl: formData.type === 'rtsp' ? formData.rtspUrl : undefined,
//             resolution: formData.resolution,
//             height: parseOptionalNumber(formData.height),
//             resolutionWidth: parseOptionalNumber(formData.resolutionWidth),
//             resolutionHeight: parseOptionalNumber(formData.resolutionHeight),
//             decodingResource: formData.decodingResource,
//             decodingPipeline: formData.decodingPipeline,
//             dockerMode: formData.dockerMode,
//             codec: formData.codec,
//             fps: Math.floor(Number(formData.fps)),
//             status: formData.status,
//             usecaseIds: editCamera.usecases ? editCamera.usecases.map((u: any) => u.usecaseId) : [],
//             roi: editCamera.roi || null
//           };
//           const res = await updateCamera(payload);
//           if (res && res.code === 200) {
//             showToast({
//               severity: "success",
//               summary: "Success",
//               detail: "Camera updated successfully",
//               life: 2000
//             });
//             setTimeout(() => navigate("/camera-details"), 1000);
//           } else {
//             showToast({
//               severity: "error",
//               summary: "Error",
//               detail: res?.message || "Failed to update camera",
//               life: 3000
//             });
//           }
//         } else {
//           const payload: any = {
//             name: formData.name,
//             type: formData.type,
//             rtspUrl: formData.type === 'rtsp' ? formData.rtspUrl : undefined,
//             resolution: formData.resolution,
//             height: parseOptionalNumber(formData.height),
//             resolutionWidth: parseOptionalNumber(formData.resolutionWidth),
//             resolutionHeight: parseOptionalNumber(formData.resolutionHeight),
//             decodingResource: formData.decodingResource,
//             decodingPipeline: formData.decodingPipeline,
//             dockerMode: formData.dockerMode,
//             codec: formData.codec,
//             fps: Math.floor(Number(formData.fps)),
//             status: 1,
//             usecaseIds: [],
//             roi: null
//           };
//           const res = await addCamera(payload);
//           if (res && res.code === 200) {
//             showToast({
//               severity: "success",
//               summary: "Success",
//               detail: "Camera created successfully",
//               life: 2000
//             });
//             setTimeout(() => navigate("/camera-details"), 1000);
//           } else {
//             showToast({
//               severity: "error",
//               summary: "Error",
//               detail: res?.message || "Failed to add camera",
//               life: 3000
//             });
//           }
//         }

//       } catch (err: any) {
//         showToast({
//           severity: "error",
//           summary: "Error",
//           detail: err?.message || "Something went wrong",
//           life: 3000
//         });
//       }
//     })();
//   };

//   /* -------------------- render -------------------- */

//   return (
//     <Layout>
//       <Toast ref={toastRef} position="top-right" />
//       <div className="page-container">
//         <div className="page-header">
//           <div className="header-left">
//             <div className="header-title">
//               {isEditMode ? "EDIT CAMERA" : "ADD CAMERA"}
//             </div>
//             <div className="header-subtitle">
//               {isEditMode ? "Update camera details" : "Complete all required fields and submit"}
//             </div>
//           </div>

//           <div className="action-buttons">
//             <button className="icon-button" onClick={() => navigate(-1)}>
//               <span className="material-icons">arrow_back</span>
//               Back
//             </button>
//           </div>
//         </div>

//         <div className="form-wrapper">
//           <div className="form-content">
//             <div className="form-section">

//               {/* Row 1: Name / Type */}
//               <div className="form-row">
//                 <div className="form-group">
//                   <label className="segment-label">
//                     Name <span className="required-star">*</span>
//                   </label>
//                   <div className="segment-style input-segment">
//                     <input
//                       type="text"
//                       className="segment-input"
//                       placeholder="Enter camera name"
//                       value={formData.name}
//                       onChange={(e) => setFormData({ ...formData, name: e.target.value })}
//                     />
//                   </div>
//                 </div>

//                 <div className="form-group">
//                   <label className="segment-label">
//                     Type <span className="required-star">*</span>
//                   </label>
//                   <div className="segment-style dropdown-segment">
//                     <select
//                       className="segment-input"
//                       value={formData.type}
//                       onChange={(e) => setFormData({ ...formData, type: e.target.value })}
//                     >
//                       <option value="">Select type</option>
//                       <option value="rtsp">RTSP</option>
//                       <option value="usb">USB</option>
//                     </select>
//                     <span className="material-icons dropdown-arrow">arrow_drop_down</span>
//                   </div>
//                 </div>
//               </div>

//               {/* Row 2: CODEC / Resolution */}
//               <div className="form-row">
//                 <div className="form-group">
//                   <label className="segment-label">
//                     CODEC <span className="required-star">*</span>
//                   </label>
//                   <div className="segment-style dropdown-segment">
//                     <select
//                       className="segment-input"
//                       value={formData.codec}
//                       onChange={(e) => setFormData({ ...formData, codec: e.target.value })}
//                     >
//                       <option value="">Select CODEC</option>
//                       <option value="H.264">H.264</option>
//                       <option value="H.265">H.265</option>
//                       <option value="MJPEG">MJPEG</option>
//                       <option value="MPEG-4">MPEG-4</option>
//                       <option value="VP8">VP8</option>
//                     </select>
//                     <span className="material-icons dropdown-arrow">arrow_drop_down</span>
//                   </div>
//                 </div>

//                 <div className="form-group">
//                   <label className="segment-label">
//                     Frame Rate <span className="required-star">*</span>
//                   </label>
//                   <div className="segment-style input-segment">
//                     <input
//                       type="text"
//                       className="segment-input"
//                       placeholder="Enter frame rate"
//                       value={formData.fps}
//                       onChange={(e) => setFormData({ ...formData, fps: e.target.value })}
//                     />
//                   </div>
//                 </div>

//               </div>

//               {/* Row 3: Frame Rate / Status (if edit) */}
//               <div className="form-row">
                

//                 {isEditMode && (
//                   <div className="form-group">
//                     <label className="segment-label">Status</label>
//                     <div className="segment-style dropdown-segment">
//                       <select
//                         className="segment-input"
//                         value={formData.status}
//                         onChange={(e) => setFormData({ ...formData, status: Number(e.target.value) })}
//                       >
//                         <option value={1}>Active</option>
//                         <option value={0}>Inactive</option>
//                       </select>
//                       <span className="material-icons dropdown-arrow">arrow_drop_down</span>
//                     </div>
//                   </div>
//                 )}
//               </div>

//               {/* Row 4: Dimensions */}
//               <div className="form-row">
//                 <div className="form-group">
//                   <label className="segment-label">Height (m)</label>
//                   <div className="segment-style input-segment">
//                     <input
//                       type="number"
//                       min="0"
//                       step="0.01"
//                       className="segment-input"
//                       placeholder="Enter camera height"
//                       value={formData.height}
//                       onChange={(e) => setFormData({ ...formData, height: e.target.value })}
//                     />
//                   </div>
//                 </div>

//                 <div className="form-group">
//                   <label className="segment-label">Resolution Width</label>
//                   <div className="segment-style input-segment">
//                     <input
//                       type="number"
//                       min="0"
//                       step="1"
//                       className="segment-input"
//                       placeholder="Enter width"
//                       value={formData.resolutionWidth}
//                       onChange={(e) => setFormData({ ...formData, resolutionWidth: e.target.value })}
//                     />
//                   </div>
//                 </div>
//               </div>

//               {/* Row 5: Resolution Height / Decoding Resource */}
//               <div className="form-row">
//                 <div className="form-group">
//                   <label className="segment-label">Resolution Height</label>
//                   <div className="segment-style input-segment">
//                     <input
//                       type="number"
//                       min="0"
//                       step="1"
//                       className="segment-input"
//                       placeholder="Enter height"
//                       value={formData.resolutionHeight}
//                       onChange={(e) => setFormData({ ...formData, resolutionHeight: e.target.value })}
//                     />
//                   </div>
//                 </div>

//                 <div className="form-group">
//                   <label className="segment-label">
//                     Decoding Resource <span className="required-star">*</span>
//                     </label>
//                   <div className="segment-style dropdown-segment">
//                     <select
//                       className="segment-input"
//                       value={formData.decodingResource}
//                       onChange={(e) => setFormData({ ...formData, decodingResource: e.target.value })}
//                     >
//                       <option value="cpu">CPU</option>
//                       <option value="gpu">GPU</option>
//                       <option value="vaapi">VAAPI</option>
//                     </select>
//                     <span className="material-icons dropdown-arrow">arrow_drop_down</span>
//                   </div>
//                 </div>
//               </div>

//               {/* Row 6: Decoding Pipeline / Docker Mode */}
//               <div className="form-row">
//                 <div className="form-group">
//                   <label className="segment-label">
//                     Decoding Pipeline <span className="required-star">*</span>
//                     </label>
//                   <div className="segment-style dropdown-segment">
//                     <select
//                       className="segment-input"
//                       value={formData.decodingPipeline}
//                       onChange={(e) => setFormData({ ...formData, decodingPipeline: e.target.value })}
//                     >
//                       <option value="ffmpeg">FFmpeg</option>
//                       <option value="opencv">OpenCV</option>
//                       <option value="gstreamer">GStreamer</option>
//                       <option value="deepstream">Deepstream</option>
//                     </select>
//                     <span className="material-icons dropdown-arrow">arrow_drop_down</span>
//                   </div>
//                 </div>

//                 <div className="form-group">
//                   <label className="segment-label">Docker Mode</label>
//                   <div className="segment-style dropdown-segment">
//                     <select
//                       className="segment-input"
//                       value={formData.dockerMode}
//                       onChange={(e) => setFormData({ ...formData, dockerMode: e.target.value })}
//                     >
//                       <option value="load">Load</option>
//                       <option value="build">Build</option>
//                     </select>
//                     <span className="material-icons dropdown-arrow">arrow_drop_down</span>
//                   </div>
//                 </div>
//               </div>

//               {/* Row 7: RTSP URL */}
//               {formData.type === 'rtsp' && (
//                 <div className="form-row">
//                   <div className="form-group">
//                     <label className="segment-label">
//                       RTSP URL <span className="required-star">*</span>
//                     </label>
//                     <div className="segment-style input-segment">
//                       <input
//                         type="text"
//                         className="segment-input"
//                         placeholder="rtsp://username:password@ip:port/stream"
//                         value={formData.rtspUrl}
//                         onChange={(e) => setFormData({ ...formData, rtspUrl: e.target.value })}
//                       />
//                     </div>
//                   </div>
//                 </div>
//               )}

//             </div>
//           </div>
//         </div>

//         <div className="camera-footer">
//           <div className="form-footer">
//             <button className="btn btn-secondary" onClick={() => navigate(-1)}>
//               Cancel
//             </button>
//             <button className="btn btn-primary" onClick={handleSubmit}>
//               {isEditMode ? "Update Camera" : "Add Camera"}
//             </button>
//           </div>
//         </div>
//       </div>
//     </Layout>
//   );
// };

// export default AddCamera;


import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useLocation } from 'react-router-dom';
import { Toast } from "primereact/toast";
import Layout from '../../components/Layout';
import "../../styles/camera.css";
import type { CameraFormData } from '../../types/camera.types';
import { addCamera, updateCamera } from "../../api/camera.api";
import { useToast } from '../../providers/toastProvider';
 
const normalizeDecodingResource = (value?: string) => {
  const normalized = (value || "").toLowerCase();
  return normalized === "gpu" || normalized === "vaapi" ? normalized : "cpu";
};
const normalizeDecodingPipeline = (value?: string) => {
  const normalized = (value || "").toLowerCase();
  return normalized === "opencv" || normalized === "gstreamer" || normalized === "deepstream" ? normalized : "ffmpeg";
};
 
 
const normalizeDockerMode = (value?: string) => {
  const normalized = (value || "").toLowerCase();
  return normalized === "build" ? "build" : "load";
};
 
const AddCamera = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { showToast } = useToast();
  const toastRef = useRef<Toast>(null);
  const editCamera = location.state?.camera;
  const isEditMode = Boolean(editCamera && (editCamera.cameraId || editCamera.id));
 
  const [formData, setFormData] = useState<CameraFormData>({
    name: '',
    type: '',
    resolution: '',
    fps: '',
    codec: 'H.264',
    rtspUrl: '',
    height: '',
    resolutionWidth: '',
    resolutionHeight: '',
    decodingResource: 'cpu',
    decodingPipeline: 'ffmpeg',
    dockerMode: 'load',
    status: 1
  });
 
  /* -------------------- effects -------------------- */
 
  useEffect(() => {
    if (!editCamera) return;
 
    setFormData(prev => ({
      ...prev,
      name: editCamera.name || "",
      type: editCamera.type || "",
      rtspUrl: editCamera.rtspUrl || "",
      resolution: editCamera.resolution || "",
      codec: editCamera.codec || "H.264",
      fps: editCamera.fps || "",
      height: editCamera.height != null ? String(editCamera.height) : "",
      resolutionWidth: editCamera.resolutionWidth != null ? String(editCamera.resolutionWidth) : "",
      resolutionHeight: editCamera.resolutionHeight != null ? String(editCamera.resolutionHeight) : "",
      decodingResource: normalizeDecodingResource(editCamera.decodingResource),
      decodingPipeline: normalizeDecodingPipeline(editCamera.decodingPipeline),
      dockerMode: normalizeDockerMode(editCamera.dockerMode),
      status: editCamera.status === "Active" || editCamera.status === 1 ? 1 : 0
    }));
  }, [editCamera]);
 
  /* -------------------- helpers -------------------- */
 
  const showError = (msg: string) =>
    toastRef.current?.show({
      severity: "error",
      summary: "Validation Error",
      detail: msg,
      life: 3000
    });
 
  const parseOptionalNumber = (value: string) => {
    const trimmed = value.trim();
    return trimmed === "" ? undefined : Number(trimmed);
  };
 
  /* -------------------- validation -------------------- */
 
  const validateForm = (): boolean => {
    if (!formData.name.trim()) {
      showError("Camera name is required");
      return false;
    }
 
    // if (!formData.codec) {
    //   showError("CODEC is required");
    //   return false;
    // }
 
    if (!formData.type) {
      showError("Camera type is required");
      return false;
    }
 
    if (!formData.fps || String(formData.fps).trim() === "") {
      showError("Frame rate is required");
      return false;
    }
    if (isNaN(Number(formData.fps)) || Number(formData.fps) <= 0) {
      showError("Frame rate must be a valid positive number");
      return false;
    }
 
    for (const [label, value] of [
      ["Height", formData.height],
      ["Resolution width", formData.resolutionWidth],
      ["Resolution height", formData.resolutionHeight]
    ] as const) {
      if (value.trim() !== "" && (isNaN(Number(value)) || Number(value) <= 0)) {
        showError(`${label} must be a valid positive number`);
        return false;
      }
    }
 
    if (formData.type === 'rtsp') {
      if (!formData.rtspUrl?.trim()) {
        showError("RTSP URL is required");
        return false;
      }
      if (!formData.rtspUrl.startsWith("rtsp://")) {
        showError("RTSP URL must start with rtsp://");
        return false;
      }
    }
 
    if (formData.type === 'usb' || formData.type === 'video') {
      if (!formData.rtspUrl?.trim()) {
        showError(`${formData.type === 'usb' ? 'USB device path' : 'Video file path'} is required`);
        return false;
      }
    }
 
    return true;
  };
 
  /* -------------------- submit -------------------- */
 
  const handleSubmit = () => {
    if (!validateForm()) return;
 
    (async () => {
      try {
        if (isEditMode && editCamera) {
          const payload = {
            cameraId: editCamera.cameraId || editCamera.id,
            name: formData.name,
            type: formData.type,
            rtspUrl: formData.rtspUrl || undefined,
            resolution: formData.resolution,
            height: parseOptionalNumber(formData.height),
            resolutionWidth: parseOptionalNumber(formData.resolutionWidth),
            resolutionHeight: parseOptionalNumber(formData.resolutionHeight),
            decodingResource: formData.decodingResource,
            decodingPipeline: formData.decodingPipeline,
            dockerMode: formData.dockerMode,
            codec: "H.264",
            fps: Math.floor(Number(formData.fps)),
            status: formData.status,
            usecaseIds: editCamera.usecases ? editCamera.usecases.map((u: any) => u.usecaseId) : [],
            roi: editCamera.roi || null
          };
          const res = await updateCamera(payload);
          if (res && res.code === 200) {
            showToast({
              severity: "success",
              summary: "Success",
              detail: "Camera updated successfully",
              life: 2000
            });
            setTimeout(() => navigate("/camera-details"), 1000);
          } else {
            showToast({
              severity: "error",
              summary: "Error",
              detail: res?.message || "Failed to update camera",
              life: 3000
            });
          }
        } else {
          const payload: any = {
            name: formData.name,
            type: formData.type,
            rtspUrl: formData.rtspUrl || undefined,
            resolution: formData.resolution,
            height: parseOptionalNumber(formData.height),
            resolutionWidth: parseOptionalNumber(formData.resolutionWidth),
            resolutionHeight: parseOptionalNumber(formData.resolutionHeight),
            decodingResource: formData.decodingResource,
            decodingPipeline: formData.decodingPipeline,
            dockerMode: formData.dockerMode,
            codec: "H.264",
            fps: Math.floor(Number(formData.fps)),
            status: 1,
            usecaseIds: [],
            roi: null
          };
          const res = await addCamera(payload);
          if (res && res.code === 200) {
            showToast({
              severity: "success",
              summary: "Success",
              detail: "Camera created successfully",
              life: 2000
            });
            setTimeout(() => navigate("/camera-details"), 1000);
          } else {
            showToast({
              severity: "error",
              summary: "Error",
              detail: res?.message || "Failed to add camera",
              life: 3000
            });
          }
        }
 
      } catch (err: any) {
        showToast({
          severity: "error",
          summary: "Error",
          detail: err?.message || "Something went wrong",
          life: 3000
        });
      }
    })();
  };
 
  /* -------------------- render -------------------- */
 
  const urlLabel = formData.type === 'rtsp'
    ? 'RTSP URL'
    : formData.type === 'usb'
    ? 'USB Device Path'
    : 'Video File Path';
 
  const urlPlaceholder = formData.type === 'rtsp'
    ? 'rtsp://username:password@ip:port/stream'
    : formData.type === 'usb'
    ? '/dev/video0'
    : '/path/to/video/file.mp4';
 
  return (
    <Layout>
      <Toast ref={toastRef} position="top-right" />
      <div className="page-container">
        <div className="page-header">
          <div className="header-left">
            <div className="header-title">
              {isEditMode ? "EDIT CAMERA" : "ADD CAMERA"}
            </div>
            <div className="header-subtitle">
              {isEditMode ? "Update camera details" : "Complete all required fields and submit"}
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
            <div className="form-section" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
 
              {/* Row 1: Name / Type */}
              <div className="form-row">
                <div className="form-group">
                  <label className="segment-label">
                    Name <span className="required-star">*</span>
                  </label>
                  <div className="segment-style input-segment">
                    <input
                      type="text"
                      className="segment-input"
                      placeholder="Enter camera name"
                      value={formData.name}
                      onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    />
                  </div>
                </div>
 
                <div className="form-group">
                  <label className="segment-label">
                    Type <span className="required-star">*</span>
                  </label>
                  <div className="segment-style dropdown-segment">
                    <select
                      className="segment-input"
                      value={formData.type}
                      onChange={(e) => setFormData({ ...formData, type: e.target.value, rtspUrl: '' })}
                    >
                      <option value="">Select type</option>
                      <option value="rtsp">RTSP</option>
                      <option value="usb">USB</option>
                      <option value="video">Video</option>
                    </select>
                    <span className="material-icons dropdown-arrow">arrow_drop_down</span>
                  </div>
                </div>
              </div>
 
              {/* Row 2: CODEC - commented out */}
              {/* <div className="form-row">
                <div className="form-group">
                  <label className="segment-label">
                    CODEC <span className="required-star">*</span>
                  </label>
                  <div className="segment-style dropdown-segment">
                    <select
                      className="segment-input"
                      value={formData.codec}
                      onChange={(e) => setFormData({ ...formData, codec: e.target.value })}
                    >
                      <option value="">Select CODEC</option>
                      <option value="H.264">H.264</option>
                      <option value="H.265">H.265</option>
                      <option value="MJPEG">MJPEG</option>
                      <option value="MPEG-4">MPEG-4</option>
                      <option value="VP8">VP8</option>
                    </select>
                    <span className="material-icons dropdown-arrow">arrow_drop_down</span>
                  </div>
                </div>
              </div> */}
 
              {/* Row 3: Status (edit mode only) */}
              {isEditMode && (
                <div className="form-row">
                  <div className="form-group">
                    <label className="segment-label">Status</label>
                    <div className="segment-style dropdown-segment">
                      <select
                        className="segment-input"
                        value={formData.status}
                        onChange={(e) => setFormData({ ...formData, status: Number(e.target.value) })}
                      >
                        <option value={1}>Active</option>
                        <option value={0}>Inactive</option>
                      </select>
                      <span className="material-icons dropdown-arrow">arrow_drop_down</span>
                    </div>
                  </div>
                  <div className="form-group" />
                </div>
              )}
 
              {/* Row 4: Height / Resolution Width */}
              <div className="form-row">
                <div className="form-group">
                  <label className="segment-label">Height (m)</label>
                  <div className="segment-style input-segment">
                    <input
                      type="number"
                      min="0"
                      step="0.01"
                      className="segment-input"
                      placeholder="Enter camera height"
                      value={formData.height}
                      onChange={(e) => setFormData({ ...formData, height: e.target.value })}
                    />
                  </div>
                </div>
 
                <div className="form-group">
                  <label className="segment-label">Resolution Width</label>
                  <div className="segment-style input-segment">
                    <input
                      type="number"
                      min="0"
                      step="1"
                      className="segment-input"
                      placeholder="Enter width"
                      value={formData.resolutionWidth}
                      onChange={(e) => setFormData({ ...formData, resolutionWidth: e.target.value })}
                    />
                  </div>
                </div>
              </div>
 
              {/* Row 5: Resolution Height / Decoding Resource */}
              <div className="form-row">
                <div className="form-group">
                  <label className="segment-label">Resolution Height</label>
                  <div className="segment-style input-segment">
                    <input
                      type="number"
                      min="0"
                      step="1"
                      className="segment-input"
                      placeholder="Enter height"
                      value={formData.resolutionHeight}
                      onChange={(e) => setFormData({ ...formData, resolutionHeight: e.target.value })}
                    />
                  </div>
                </div>
 
                <div className="form-group">
                  <label className="segment-label">
                    Decoding Resource <span className="required-star">*</span>
                  </label>
                  <div className="segment-style dropdown-segment">
                    <select
                      className="segment-input"
                      value={formData.decodingResource}
                      onChange={(e) => setFormData({ ...formData, decodingResource: e.target.value })}
                    >
                      <option value="cpu">CPU</option>
                      <option value="gpu">GPU</option>
                      <option value="vaapi">VAAPI</option>
                    </select>
                    <span className="material-icons dropdown-arrow">arrow_drop_down</span>
                  </div>
                </div>
              </div>
 
              {/* Row 6: Decoding Pipeline / Docker Mode */}
              <div className="form-row">
                <div className="form-group">
                  <label className="segment-label">
                    Decoding Pipeline <span className="required-star">*</span>
                  </label>
                  <div className="segment-style dropdown-segment">
                    <select
                      className="segment-input"
                      value={formData.decodingPipeline}
                      onChange={(e) => setFormData({ ...formData, decodingPipeline: e.target.value })}
                    >
                      <option value="ffmpeg">FFmpeg</option>
                      <option value="opencv">OpenCV</option>
                      <option value="gstreamer">GStreamer</option>
                      <option value="deepstream">Deepstream</option>
                    </select>
                    <span className="material-icons dropdown-arrow">arrow_drop_down</span>
                  </div>
                </div>
 
                <div className="form-group">
                  <label className="segment-label">Docker Mode</label>
                  <div className="segment-style dropdown-segment">
                    <select
                      className="segment-input"
                      value={formData.dockerMode}
                      onChange={(e) => setFormData({ ...formData, dockerMode: e.target.value })}
                    >
                      <option value="load">Load</option>
                      <option value="build">Build</option>
                    </select>
                    <span className="material-icons dropdown-arrow">arrow_drop_down</span>
                  </div>
                </div>
              </div>
 
              {/* Row 7: Frame Rate */}
              <div className="form-row">
                <div className="form-group">
                  <label className="segment-label">
                    Frame Rate <span className="required-star">*</span>
                  </label>
                  <div className="segment-style input-segment">
                    <input
                      type="text"
                      className="segment-input"
                      placeholder="Enter frame rate"
                      value={formData.fps}
                      onChange={(e) => setFormData({ ...formData, fps: e.target.value })}
                    />
                  </div>
                </div>
                <div className="form-group" />
              </div>
 
              {/* Row 8: URL/Path - shown for rtsp, usb, video */}
              {(formData.type === 'rtsp' || formData.type === 'usb' || formData.type === 'video') && (
                <div className="form-row">
                  <div className="form-group">
                    <label className="segment-label">
                      {urlLabel} <span className="required-star">*</span>
                    </label>
                    <div className="segment-style input-segment">
                      <input
                        type="text"
                        className="segment-input"
                        placeholder={urlPlaceholder}
                        value={formData.rtspUrl}
                        onChange={(e) => setFormData({ ...formData, rtspUrl: e.target.value })}
                      />
                    </div>
                  </div>
                  
                </div>
              )}
 
            </div>
          </div>
        </div>
 
        <div className="camera-footer">
          <div className="form-footer">
            <button className="btn btn-secondary" onClick={() => navigate(-1)}>
              Cancel
            </button>
            <button className="btn btn-primary" onClick={handleSubmit}>
              {isEditMode ? "Update Camera" : "Add Camera"}
            </button>
          </div>
        </div>
      </div>
    </Layout>
  );
};
 
export default AddCamera;
 