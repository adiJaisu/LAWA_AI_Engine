import React from 'react'
import { useRoiEditorContext } from '../../context/RoiEditorContext'
import { UseCaseModal } from './UseCaseModal'
import type { Annotation, Point, PolygonAnnotation } from '../../types/roi.types'
import '../../styles/ROI/Canvas.css'

export const Canvas: React.FC = () => {
  const roiState = useRoiEditorContext()

  const {
    imageSrc,
    isLoadingImage,
    imageLoadError,
    isLoadingAnnotations,
    annotations,
    selectedAnnotationIndex,
    isDrawing,
    isPolygonDrawing,
    currentRect,
    currentPolygonPoints,
    tempPoint,
    showLabelingModal,
    drawingMode,
    scale,
    imageDims,
    polygonAnnotations,
    canvasTransformStyle,
    labelOptions,
    selectedUseCases,
    setImageDimensions,
    handleImageMousedown,
    handleImageMousemove,
    handleImageMouseup,
    handleZoomWheel,
    zoomIn,
    zoomOut,
    resetZoom,
    getRectStyle,
    getPolygonLabelStyle,
    polygonPointsToString,
    saveAnnotation,
    cancelLabeling,
    finalizeCurrentPolygon,
    startVertexDrag,
    setSelectedUseCases,
  } = roiState

  const selectedAnnotation =
    selectedAnnotationIndex !== null
      ? (annotations[selectedAnnotationIndex] as Annotation | undefined)
      : undefined

  const handleDoubleClick = (e: React.MouseEvent) => {
    e.preventDefault()
    if (drawingMode === 'polygon' && isPolygonDrawing && currentPolygonPoints.length >= 3) {
      finalizeCurrentPolygon()
    }
  }

  return (
    <div
      className="roiImageArea"
      ref={roiState.imageAreaRef}
      onMouseDown={handleImageMousedown}
      onMouseMove={handleImageMousemove}
      onMouseUp={handleImageMouseup}
      onWheel={handleZoomWheel}
      onDoubleClick={handleDoubleClick}
    >
      {(isLoadingImage || isLoadingAnnotations) && (
        <div className="loadingOverlay">
          <div className="spinner"></div>
          <p>{isLoadingImage ? 'Loading camera frame...' : 'Loading annotations...'}</p>
        </div>
      )}

      {imageLoadError && !isLoadingImage && (
        <div className="errorMessage">
          <span className="material-icons">error_outline</span>
          <p>{imageLoadError}</p>
          <button onClick={() => window.location.reload()} className="retryBtn">
            Retry
          </button>
        </div>
      )}

      {imageSrc && !isLoadingImage && !isLoadingAnnotations && (
        <div className="zoomControls">
          <a
            href="#"
            onClick={(e) => {
              e.preventDefault()
              zoomOut()
            }}
            className="zoomBtn"
            title="Zoom Out"
          >
            <span className="material-icons">zoom_out</span>
          </a>
          <span className="zoomLevelTxt">{(scale * 100).toFixed(0)}%</span>
          <a
            href="#"
            onClick={(e) => {
              e.preventDefault()
              zoomIn()
            }}
            className="zoomBtn"
            title="Zoom In"
          >
            <span className="material-icons">zoom_in</span>
          </a>
          <a
            href="#"
            onClick={(e) => {
              e.preventDefault()
              resetZoom()
            }}
            className="zoomBtn zoomReset"
            title="Reset Zoom"
          >
            <span className="material-icons">refresh</span>
          </a>
        </div>
      )}

      {imageSrc && !isLoadingImage && !isLoadingAnnotations && (
        <div ref={roiState.transformWrapperRef} className="transformWrapper" style={canvasTransformStyle}>
          <img
            src={imageSrc}
            alt="ROI Background"
            className="roiImage"
            ref={roiState.roiImageRef}
            onLoad={setImageDimensions}
            onError={() => console.error('Image load error')}
            style={{
              width: `${imageDims.width}px`,
              height: `${imageDims.height}px`,
              display: 'block',
            }}
            draggable={false}
          />

          {annotations.map((anno: Annotation, index: number) =>
            anno.type === 'rectangle' && anno.isVisible !== false ? (
              <div
                key={`rect-${index}`}
                className={`annotationRect ${selectedAnnotationIndex === index ? 'selected' : ''} ${
                  anno.isLocked ? 'locked' : ''
                }`}
                style={getRectStyle(anno)}
                onClick={(e) => {
                  e.stopPropagation()
                  if (drawingMode === 'edit' && !anno.isLocked) {
                    roiState.setSelectedAnnotationIndex(index)
                  }
                }}
              >
                <span className="annotationLabel">
                  ROI {index + 1} ({anno.type.charAt(0).toUpperCase()})
                  {anno.isLocked && <span className="material-icons">lock</span>}
                </span>
              </div>
            ) : null
          )}

          <svg
            className="annotationsSvgLayer"
            style={{
              width: `${imageDims.width}px`,
              height: `${imageDims.height}px`,
            }}
          >
            {polygonAnnotations.map((anno: PolygonAnnotation & { originalIndex: number }) =>
              anno.isVisible !== false ? (
                <g
                  key={`poly-${anno.originalIndex}`}
                  onClick={(e) => {
                    e.stopPropagation()
                    if (drawingMode === 'edit' && !anno.isLocked) {
                      roiState.setSelectedAnnotationIndex(anno.originalIndex)
                    }
                  }}
                >
                  <polygon
                    points={polygonPointsToString(anno.points)}
                    className={`annotationPolygon ${selectedAnnotationIndex === anno.originalIndex ? 'selected' : ''} ${
                      anno.isLocked ? 'locked' : ''
                    }`}
                  />
                </g>
              ) : null
            )}

            {isPolygonDrawing && currentPolygonPoints.length > 0 && !showLabelingModal && (
              <g>
                <polyline
                  points={currentPolygonPoints.map((p: Point) => `${p.x},${p.y}`).join(' ')}
                  className="currentPolygonOutline"
                  fill="none"
                  stroke="#3498db"
                  strokeWidth="4"
                />
                {currentPolygonPoints.length > 0 && (
                  <line
                    x1={currentPolygonPoints[currentPolygonPoints.length - 1].x}
                    y1={currentPolygonPoints[currentPolygonPoints.length - 1].y}
                    x2={tempPoint.x}
                    y2={tempPoint.y}
                    className="currentPolygonRubberband"
                    stroke="#3498db"
                    strokeWidth="2"
                    strokeDasharray="4,4"
                  />
                )}
                {currentPolygonPoints.map((point: Point, i: number) => (
                  <circle
                    key={`point-${i}`}
                    cx={point.x}
                    cy={point.y}
                    r="4"
                    className="polygonPoint"
                    fill="#3498db"
                    stroke="white"
                    strokeWidth="2"
                  />
                ))}
              </g>
            )}
          </svg>

          {polygonAnnotations.map((anno: PolygonAnnotation & { originalIndex: number }) =>
            anno.isVisible !== false ? (
              <span
                key={`label-${anno.originalIndex}`}
                className="annotationLabel polygonLabel"
                style={getPolygonLabelStyle(anno.points)}
                onClick={(e) => {
                  e.stopPropagation()
                  if (drawingMode === 'edit' && !anno.isLocked) {
                    roiState.setSelectedAnnotationIndex(anno.originalIndex)
                  }
                }}
              >
                ROI {anno.originalIndex + 1} ({anno.type.charAt(0).toUpperCase()})
                {anno.isLocked && <span className="material-icons">lock</span>}
              </span>
            ) : null
          )}

          {isDrawing && !showLabelingModal && <div className="drawingRect" style={getRectStyle(currentRect)} />}

          {drawingMode === 'edit' &&
            selectedAnnotation &&
            selectedAnnotation.type === 'rectangle' &&
            !selectedAnnotation.isLocked &&
            selectedAnnotation.isVisible !== false && (
              <>
                {[
                  { x: selectedAnnotation.x1, y: selectedAnnotation.y1, cls: 'topLeft' },
                  { x: selectedAnnotation.x2, y: selectedAnnotation.y1, cls: 'topRight' },
                  { x: selectedAnnotation.x2, y: selectedAnnotation.y2, cls: 'bottomRight' },
                  { x: selectedAnnotation.x1, y: selectedAnnotation.y2, cls: 'bottomLeft' },
                ].map((vertex, idx) => (
                  <div
                    key={`rect-vertex-${idx}`}
                    className={`editHandle ${vertex.cls}`}
                    style={{ left: `${vertex.x}px`, top: `${vertex.y}px` }}
                    onMouseDown={(e) => {
                      e.preventDefault()
                      e.stopPropagation()
                      startVertexDrag(`rect-${idx}`, { x: vertex.x, y: vertex.y })
                    }}
                  />
                ))}
              </>
            )}

          {drawingMode === 'edit' &&
            selectedAnnotation &&
            selectedAnnotation.type === 'polygon' &&
            !selectedAnnotation.isLocked &&
            selectedAnnotation.isVisible !== false && (
              <svg
                className="editSvgLayer"
                style={{
                  width: `${imageDims.width}px`,
                  height: `${imageDims.height}px`,
                }}
              >
                {selectedAnnotation.points.map((point, idx) => (
                  <circle
                    key={`poly-vertex-${idx}`}
                    cx={point.x}
                    cy={point.y}
                    r="6"
                    className="editVertex"
                    onMouseDown={(e) => {
                      e.preventDefault()
                      e.stopPropagation()
                      startVertexDrag(`poly-${idx}`, { x: point.x, y: point.y })
                    }}
                  />
                ))}
              </svg>
            )}
        </div>
      )}

      {showLabelingModal && (
        <UseCaseModal
          isOpen={showLabelingModal}
          availableUsecases={labelOptions}
          tempUseCases={selectedUseCases}
          setTempUseCases={setSelectedUseCases}
          onSave={saveAnnotation}
          onClose={cancelLabeling}
        />
      )}
    </div>
  )
}
