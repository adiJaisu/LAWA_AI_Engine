import { useNavigate, useSearchParams } from 'react-router-dom'
import { Toast } from 'primereact/toast'
import { useRoiEditor } from '../../hooks/useRoiEditor'
import { RoiEditorContext } from '../../context/RoiEditorContext'
import { Canvas } from './Canvas'
import { LeftPanel } from './LeftPanel'
import { RightPanel } from './RightPanel'
import { checkPermission } from '../../utils/permissionUtils'
import '../../styles/ROI/RoiEditor.css'
import Layout from '../../components/Layout'

export const RoiEditor = () => {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const cameraId = searchParams.get('cameraId')
  const canUpdateCamera = checkPermission('camera', 'update')

  const roiState = useRoiEditor(cameraId ?? undefined)

  const handleSave = async () => {
    await roiState.saveAnnotations()
  }

  return (
    <Layout>
      <Toast ref={roiState.toastRef} position="top-right" />
      <RoiEditorContext.Provider value={roiState}>
        <div className="page-container">
          <div className="page-header">
            <div className="header-left">
              <div className="header-title">ROI EDITOR</div>
              <div className="header-subtitle">
                Edit ROI for camera: {roiState.cameraDetails?.name || `Camera ${cameraId ?? ''}`}
              </div>
            </div>

            <div className="action-buttons">
              {canUpdateCamera && (
                <button
                  className="icon-button"
                  onClick={() => {
                    void roiState.refreshCanvasFrame()
                  }}
                  disabled={roiState.isLoadingImage}
                  title="Reload canvas frame"
                >
                  <span className="material-icons">refresh</span>
                  <span className="buttonText">
                    {roiState.isLoadingImage ? 'Reloading...' : 'Reload Frame'}
                  </span>
                </button>
              )}

              <button
                className="icon-button"
                onClick={() => navigate(-1)}
              >
                <span className="material-icons">arrow_back</span>
                Back
              </button>

              {canUpdateCamera && (
                <button
                  onClick={handleSave}
                  disabled={roiState.isSavingAnnotations}
                  className="icon-button"
                  title="Save ROI"
                >
                  <span className="material-icons">save</span>
                  <span className="buttonText">
                    {roiState.isSavingAnnotations ? 'Saving...' : 'Save'}
                  </span>
                </button>
              )}
            </div>
          </div>
          <div className="roiWrapper">
            <div className="roiContainer">
              <LeftPanel />
              <Canvas />
              <RightPanel />
            </div>
          </div>
        </div>
      </RoiEditorContext.Provider>
    </Layout>
  )
}

export default RoiEditor
