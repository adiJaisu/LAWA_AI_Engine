import React, { useState } from 'react'
import { useRoiEditorContext } from '../../context/RoiEditorContext'
import { UseCaseModal } from './UseCaseModal'
import ConfirmDeleteModal from '../../components/ConfirmDeleteModal'
import { GEOMETRY_UTILS } from '../../utils/geometryUtils'
import type { Annotation } from '../../types/roi.types'
import '../../styles/ROI/RightPanel.css'

export const RightPanel: React.FC = () => {
  const {
    annotations,
    selectedAnnotationIndex,
    deleteAnnotation,
    toggleLockAnnotation,
    toggleVisibilityAnnotation,
    openUseCaseEditor,
    showUseCaseModal,
    editingUseCaseIndex,
    tempUseCases,
    setTempUseCases,
    saveUseCaseEdit,
    closeUseCaseModal,
    labelOptions,
  } = useRoiEditorContext()

  const [expandedIndices, setExpandedIndices] = useState<Set<number>>(new Set())
  const [deleteConfirmIndex, setDeleteConfirmIndex] = useState<number | null>(null)

  const toggleExpand = (index: number) => {
    setExpandedIndices((prev) => {
      const newSet = new Set(prev)
      if (newSet.has(index)) {
        newSet.delete(index)
      } else {
        newSet.add(index)
      }
      return newSet
    })
  }

  const isExpanded = (index: number) => expandedIndices.has(index)

  const getAnnotationCoords = (anno: Annotation) => {
    if (anno.type === 'rectangle') {
      return `(${Math.round(anno.x1)}, ${Math.round(anno.y1)}) to (${Math.round(anno.x2)}, ${Math.round(
        anno.y2
      )})`
    }

    const bounds = GEOMETRY_UTILS.getPolygonBounds(anno.points)
    return `${anno.points.length} points, Bounds: (${Math.round(bounds.x1)}, ${Math.round(
      bounds.y1
    )}) to (${Math.round(bounds.x2)}, ${Math.round(bounds.y2)})`
  }

  const confirmDeleteAnnotation = (index: number) => {
    setDeleteConfirmIndex(index)
  }

  const handleConfirmDelete = () => {
    if (deleteConfirmIndex !== null) {
      deleteAnnotation(deleteConfirmIndex)
      setDeleteConfirmIndex(null)
    }
  }

  return (
    <div className="rightSidebar">
      <h4 className="panelHeader">ROIs ({annotations.length})</h4>
      <div className="annotationsList">
        {(annotations as Annotation[]).map((anno: Annotation, index: number) => (
          <div
            key={index}
            className={`annotationItem ${selectedAnnotationIndex === index ? 'selected' : ''} ${anno.isLocked ? 'locked' : ''
              } ${anno.isVisible === false ? 'hidden' : ''}`}
          >
            <div className={`annotationHeader ${anno.isLocked ? 'disabled' : ''}`} onClick={() => toggleExpand(index)}>
              <div className="annotationMain">
                <i className={anno.type === 'rectangle' ? 'bi bi-bounding-box' : 'bi bi-pentagon'} />
                <span className={`material-icons expandIcon ${isExpanded(index) ? 'expanded' : ''}`}>
                  {isExpanded(index) ? 'expand_less' : 'expand_more'}
                </span>
                <span className="itemLabel">
                  ROI {index + 1} ({anno.type.charAt(0).toUpperCase()})
                </span>
                <div className="actionButtons">
                  <button
                    className="actionBtn lockBtn"
                    onClick={(e) => {
                      e.stopPropagation()
                      toggleLockAnnotation(index)
                    }}
                    title={anno.isLocked ? 'Unlock ROI' : 'Lock ROI'}
                  >
                    <span className="material-icons">{anno.isLocked ? 'lock' : 'lock_open'}</span>
                  </button>

                  <button
                    className="actionBtn visibilityBtn"
                    onClick={(e) => {
                      e.stopPropagation()
                      toggleVisibilityAnnotation(index)
                    }}
                    title={anno.isVisible === false ? 'Show ROI' : 'Hide ROI'}
                    disabled={anno.isLocked}
                  >
                    <span className="material-icons">{anno.isVisible === false ? 'visibility_off' : 'visibility'}</span>
                  </button>

                  <button
                    className="actionBtn addUsecaseBtn"
                    onClick={(e) => {
                      e.stopPropagation()
                      openUseCaseEditor(index)
                    }}
                    title="Edit Use Cases"
                    disabled={anno.isLocked}
                  >
                    <span className="material-icons">add_circle</span>
                  </button>

                  <button
                    className="actionBtn deleteBtn"
                    onClick={(e) => {
                      e.stopPropagation()
                      confirmDeleteAnnotation(index)
                    }}
                    title="Delete ROI"
                    disabled={anno.isLocked}
                  >
                    <span className="material-icons">delete</span>
                  </button>
                </div>
              </div>
            </div>

            {isExpanded(index) && !anno.isLocked && (
              <div className="annotationDetails">
                <div className="itemCoords" title={getAnnotationCoords(anno)}>
                  <span className="coordsLabel">Coordinates:</span>
                  {getAnnotationCoords(anno)}
                </div>

                {anno.useCases && anno.useCases.length > 0 && (
                  <div className="usecasesSection">
                    <span className="usecasesLabel">Use Cases ({anno.useCases.length}):</span>
                    <ul className="usecasesList">
                      {anno.useCases.map((usecase: string, ucIndex: number) => (
                        <li key={ucIndex} className="usecaseItem">
                          <div className="usecaseContent">
                            <span className="usecaseBullet">•</span>
                            <span className="usecaseText">{usecase}</span>
                          </div>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}

        {annotations.length === 0 && (
          <div className="emptyState">
            <span className="material-icons">info</span>
            <p>No ROI annotations yet. Select a tool from the left panel to start drawing.</p>
          </div>
        )}
      </div>

      {showUseCaseModal && editingUseCaseIndex !== null && (
        <UseCaseModal
          isOpen={showUseCaseModal}
          availableUsecases={labelOptions}
          tempUseCases={tempUseCases}
          setTempUseCases={setTempUseCases}
          onSave={saveUseCaseEdit}
          onClose={closeUseCaseModal}
        />
      )}

      <ConfirmDeleteModal
        isOpen={deleteConfirmIndex !== null}
        title="Delete ROI"
        description="Are you sure you want to delete this ROI annotation?"
        itemName={deleteConfirmIndex !== null ? `ROI ${deleteConfirmIndex + 1}` : undefined}
        onConfirm={handleConfirmDelete}
        onCancel={() => setDeleteConfirmIndex(null)}
        confirmText="Delete"
        cancelText="Cancel"
      />
    </div>
  )
}
