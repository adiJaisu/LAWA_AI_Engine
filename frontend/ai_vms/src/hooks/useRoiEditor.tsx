import { useState, useRef, useCallback, useMemo, useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'
import { Toast } from 'primereact/toast'
import { getCameraDetails, getCameraFrame, refreshCameraFrame, updateRoi } from '../api/camera.api'
import { getAuthToken } from '../api/auth.api'
import type { CameraApiItem } from '../types/camera.types'
import type {
  Annotation,
  RectAnnotation,
  PolygonAnnotation,
  UseCase,
  Tool,
  ImageDimensions,
  Point,
  RectCoords,
  DrawingMode,
} from '../types/roi.types'
import { ROI_EDITOR_CONFIG } from '../constants/roiEditorConfig'
import { GEOMETRY_UTILS } from '../utils/geometryUtils'

const ROI_TOOLS: Tool[] = [
  { label: 'Rectangle ROI', icon: 'crop_square' },
  { label: 'Polygon ROI', icon: 'polyline' },
  { label: 'Edit ROI', icon: 'edit' },
]

export const useRoiEditor = (cameraIdProp?: string | number) => {
  const [searchParams] = useSearchParams()
  const cameraId = cameraIdProp || searchParams.get('cameraId') || searchParams.get('id') || '1'

  const [selectedTool, setSelectedTool] = useState<Tool | null>(null)
  const [drawingMode, setDrawingMode] = useState<DrawingMode>(null)
  const [showLabelingModal, setShowLabelingModal] = useState(false)
  const [showUseCaseModal, setShowUseCaseModal] = useState(false)

  const [isLoadingImage, setIsLoadingImage] = useState(false)
  const [imageLoadError, setImageLoadError] = useState('')
  const [isLoadingAnnotations, setIsLoadingAnnotations] = useState(false)
  const [isSavingAnnotations, setIsSavingAnnotations] = useState(false)

  const [imageSrc, setImageSrc] = useState('')
  const [labelOptions, setLabelOptions] = useState<string[]>([])
  const [useCases, setUseCases] = useState<UseCase[]>([])
  const [cameraDetails, setCameraDetails] = useState<CameraApiItem | null>(null)
  const [annotations, setAnnotations] = useState<Annotation[]>([])

  const [selectedAnnotationIndex, setSelectedAnnotationIndex] = useState<number | null>(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [isPolygonDrawing, setIsPolygonDrawing] = useState(false)
  const [isDraggingHandle, setIsDraggingHandle] = useState(false)

  const [currentRect, setCurrentRect] = useState<RectCoords>({ x1: 0, y1: 0, x2: 0, y2: 0 })
  const [currentPolygonPoints, setCurrentPolygonPoints] = useState<Point[]>([])
  const [tempPoint, setTempPoint] = useState<Point>({ x: 0, y: 0 })

  const [activeHandle, setActiveHandle] = useState<string | null>(null)
  const [lastMousePos, setLastMousePos] = useState<Point>({ x: 0, y: 0 })

  const [lastSavedAnnotation, setLastSavedAnnotation] = useState<Annotation | null>(null)
  const [selectedUseCases, setSelectedUseCases] = useState<string[]>([])

  const [editingUseCaseIndex, setEditingUseCaseIndex] = useState<number | null>(null)
  const [tempUseCases, setTempUseCases] = useState<string[]>([])

  const [scale, setScale] = useState(1.0)
  const [isPanning, setIsPanning] = useState(false)
  const [panOffset, setPanOffset] = useState<Point>({ x: 0, y: 0 })
  const [panStart, setPanStart] = useState<Point>({ x: 0, y: 0 })

  const imageAreaRef = useRef<HTMLDivElement | null>(null)
  const roiImageRef = useRef<HTMLImageElement | null>(null)
  const transformWrapperRef = useRef<HTMLDivElement | null>(null)
  const toastRef = useRef<Toast>(null)
  const offsetRef = useRef({ x: 0, y: 0 })
  const initialCameraRoiRef = useRef<CameraApiItem['roi'] | null>(null)
  const initialRoiAppliedRef = useRef(false)
  const frameObjectUrlRef = useRef<string | null>(null)

  const [imageDims, setImageDims] = useState<ImageDimensions>({
    width: 0,
    height: 0,
    naturalWidth: 0,
    naturalHeight: 0,
  })

  const tools = ROI_TOOLS


  const initialFitScale = useMemo(() => {
    if (!imageAreaRef.current || imageDims.width === 0 || imageDims.height === 0) {
      return 1
    }
    const areaRect = imageAreaRef.current.getBoundingClientRect()
    const scaleToFitWidth = areaRect.width / imageDims.width
    const scaleToFitHeight = areaRect.height / imageDims.height
    return Math.min(scaleToFitWidth, scaleToFitHeight)
  }, [imageDims.width, imageDims.height])

  const isZoomedIn = useMemo(() => {
    return scale > initialFitScale
  }, [scale, initialFitScale])


  const polygonAnnotations = useMemo(() => {
    return annotations
      .map((anno, index) => ({ ...anno, originalIndex: index }))
      .filter((anno): anno is PolygonAnnotation & { originalIndex: number } =>
        anno.type === 'polygon'
      )
  }, [annotations])


  const canvasTransformStyle = useMemo(() => {
    let cursor = 'default'

    if (isPanning) {
      cursor = 'grabbing'
    } else if (drawingMode === 'rectangle' || drawingMode === 'polygon') {
      cursor = 'crosshair'
    } else if (drawingMode === 'edit') {
      cursor = selectedAnnotationIndex !== null ? 'move' : 'default'
    } else if (isZoomedIn) {
      cursor = 'grab'
    }

    return {
      left: `${offsetRef.current.x}px`,
      top: `${offsetRef.current.y}px`,
      width: `${imageDims.width}px`,
      height: `${imageDims.height}px`,
      position: 'absolute' as const,
      transform: `translate(${panOffset.x}px, ${panOffset.y}px) scale(${scale})`,
      transformOrigin: '0 0',
      cursor,
    }
  }, [isPanning, drawingMode, selectedAnnotationIndex, isZoomedIn, imageDims, panOffset, scale])

  const notify = useCallback(
    (severity: 'success' | 'warn' | 'info' | 'error', message: string) => {
      if (toastRef.current) {
        toastRef.current.show({
          severity,
          summary: 'ROI',
          detail: message,
          life: 2500,
        })
        return
      }

      const prefix = `[ROI ${severity.toUpperCase()}]`
      console.log(prefix, message)
    },
    []
  )

  const clearFrameObjectUrl = useCallback(() => {
    if (frameObjectUrlRef.current) {
      URL.revokeObjectURL(frameObjectUrlRef.current)
      frameObjectUrlRef.current = null
    }
  }, [])

  const resolveFrameSrc = useCallback(
    (frameFile: Blob | string | null): string | null => {
      if (!frameFile) return null

      if (frameFile instanceof Blob) {
        clearFrameObjectUrl()
        const objectUrl = URL.createObjectURL(frameFile)
        frameObjectUrlRef.current = objectUrl
        return objectUrl
      }

      if (typeof frameFile === 'string') {
        const frameValue = frameFile.trim()
        if (!frameValue) return null
        if (frameValue.startsWith('data:image')) return frameValue
        return `data:image/jpeg;base64,${frameValue}`
      }

      return null
    },
    [clearFrameObjectUrl]
  )

  useEffect(() => {
    return () => clearFrameObjectUrl()
  }, [clearFrameObjectUrl])

  const loadCanvasFrame = useCallback(
    async (parsedCameraId: number, shouldRefresh = false) => {
      setIsLoadingImage(true)
      setImageLoadError('')

      try {
        const frameResponse = shouldRefresh
          ? await refreshCameraFrame(parsedCameraId)
          : await getCameraFrame(parsedCameraId)
        const nextImageSrc =
          frameResponse.code === 200 ? resolveFrameSrc(frameResponse.frameFile) : null

        if (frameResponse.code === 200 && nextImageSrc) {
          setImageSrc(nextImageSrc)
          setImageLoadError('')

          if (shouldRefresh) {
            notify('success', 'Camera frame refreshed.')
          }
          return true
        }

        setImageSrc('')
        setImageLoadError('Failed to fetch camera frame')

        if (shouldRefresh) {
          notify('error', frameResponse.message || 'Failed to refresh camera frame.')
        }
        return false
      } catch (error) {
        console.error('Failed to load camera frame for ROI:', error)
        setImageSrc('')
        setImageLoadError('Failed to fetch camera frame')

        if (shouldRefresh) {
          notify('error', 'Failed to refresh camera frame.')
        }
        return false
      } finally {
        setIsLoadingImage(false)
      }
    },
    [notify, resolveFrameSrc]
  )

  const refreshCanvasFrame = useCallback(async () => {
    const token = getAuthToken()
    const parsedCameraId = Number(cameraId)

    if (!token || Number.isNaN(parsedCameraId)) {
      setImageSrc('')
      setImageLoadError('Invalid camera ID or auth token.')
      notify('error', 'Unable to refresh camera frame.')
      return false
    }

    clearFrameObjectUrl()
    return loadCanvasFrame(parsedCameraId, true)
  }, [cameraId, clearFrameObjectUrl, loadCanvasFrame, notify])

  useEffect(() => {
    const token = getAuthToken()
    const parsedCameraId = Number(cameraId)
    let isMounted = true

    if (!token || Number.isNaN(parsedCameraId)) {
      setLabelOptions([])
      setUseCases([])
      setCameraDetails(null)
      setImageSrc('')
      setImageLoadError('Invalid camera ID or auth token.')
      initialCameraRoiRef.current = null
      return
    }

    setIsLoadingAnnotations(true)
    clearFrameObjectUrl()

    Promise.allSettled([
      getCameraDetails(parsedCameraId),
      loadCanvasFrame(parsedCameraId),
    ])
      .then(([detailsResult, frameResult]) => {
        if (!isMounted) return

        if (detailsResult.status === 'fulfilled') {
          const res = detailsResult.value
          if (res.code === 200 && res.cameraDetails) {
            const details = res.cameraDetails
            setCameraDetails(details)

            const cameraUsecases = details.usecases || []
            setLabelOptions(cameraUsecases.map((u) => u.usecaseName))
            setUseCases(
              cameraUsecases.map((u) => ({
                usecaseId: u.usecaseId,
                usecaseName: u.usecaseName,
                is_active: true,
              }))
            )

            initialCameraRoiRef.current = details.roi ?? null
            initialRoiAppliedRef.current = false
          } else {
            setLabelOptions([])
            setUseCases([])
            setCameraDetails(null)
            initialCameraRoiRef.current = null
          }
        } else {
          console.error('Failed to load camera details for ROI:', detailsResult.reason)
          setLabelOptions([])
          setUseCases([])
          setCameraDetails(null)
          initialCameraRoiRef.current = null
        }

        if (frameResult.status === 'rejected') {
          console.error('Failed to load camera frame for ROI:', frameResult.reason)
        }
      })
      .finally(() => {
        if (!isMounted) return
        setIsLoadingAnnotations(false)
      })

    return () => {
      isMounted = false
    }
  }, [cameraId, clearFrameObjectUrl, loadCanvasFrame])

  const applyInitialCameraRoi = useCallback((width: number, height: number) => {
    if (initialRoiAppliedRef.current || !initialCameraRoiRef.current) return

    const toPx = (value: number, max: number) => (value <= 1 ? value * max : value)
    const toUseCaseNames = (item: { usecases?: Array<{ usecaseName?: string }> }) =>
      (item.usecases || [])
        .map((u) => u.usecaseName)
        .filter((name): name is string => Boolean(name))

    const source = initialCameraRoiRef.current
    const roiItems = Array.isArray(source) ? source : [source]

    const parsedAnnotations: Annotation[] = roiItems
      .map((item) => {
        const roi = item as {
          type?: string
          x?: number
          y?: number
          w?: number
          h?: number
          points?: Array<[number, number]>
          usecases?: Array<{ usecaseName?: string }>
        }

        if (roi.type === 'rect') {
          const { x, y, w, h } = roi
          if ([x, y, w, h].some((v) => typeof v !== 'number')) return null

          const x1 = toPx(x as number, width)
          const y1 = toPx(y as number, height)
          const x2 = x1 + toPx(w as number, width)
          const y2 = y1 + toPx(h as number, height)

          const useCases = toUseCaseNames(roi)

          return {
            type: 'rectangle',
            x1,
            y1,
            x2,
            y2,
            label: useCases[0] || '',
            useCases,
            isLocked: false,
            isVisible: true,
          } as RectAnnotation
        }

        if (roi.type === 'polygon' && Array.isArray(roi.points) && roi.points.length >= 3) {
          const points = roi.points
            .filter((p): p is [number, number] => Array.isArray(p) && p.length === 2)
            .map(([px, py]) => ({
              x: toPx(px, width),
              y: toPx(py, height),
            }))

          if (points.length < 3) return null

          const bounds = GEOMETRY_UTILS.getPolygonBounds(points)
          const useCases = toUseCaseNames(roi)

          return {
            type: 'polygon',
            points,
            x1: bounds.x1,
            y1: bounds.y1,
            x2: bounds.x2,
            y2: bounds.y2,
            label: useCases[0] || '',
            useCases,
            isLocked: false,
            isVisible: true,
          } as PolygonAnnotation
        }

        return null
      })
      .filter((anno): anno is Annotation => Boolean(anno))

    if (parsedAnnotations.length > 0) {
      setAnnotations(parsedAnnotations)
    }
    initialRoiAppliedRef.current = true
  }, [])

  useEffect(() => {
    if (!cameraDetails || imageDims.naturalWidth === 0 || imageDims.naturalHeight === 0) return
    applyInitialCameraRoi(imageDims.naturalWidth, imageDims.naturalHeight)
  }, [cameraDetails, imageDims.naturalWidth, imageDims.naturalHeight, applyInitialCameraRoi])

  const setImageDimensions = useCallback(() => {
    if (!roiImageRef.current || !imageAreaRef.current) return

    const img = roiImageRef.current
    const areaRect = imageAreaRef.current.getBoundingClientRect()

    const naturalWidth = img.naturalWidth
    const naturalHeight = img.naturalHeight

    setImageDims({
      width: naturalWidth,
      height: naturalHeight,
      naturalWidth,
      naturalHeight,
    })

    applyInitialCameraRoi(naturalWidth, naturalHeight)

    const scaleToFitWidth = areaRect.width / naturalWidth
    const scaleToFitHeight = areaRect.height / naturalHeight
    const initialScale = Math.min(scaleToFitWidth, scaleToFitHeight)

    setScale(initialScale)

    const scaledWidth = naturalWidth * initialScale
    const scaledHeight = naturalHeight * initialScale

    offsetRef.current.x = (areaRect.width - scaledWidth) / 2
    offsetRef.current.y = (areaRect.height - scaledHeight) / 2
  }, [applyInitialCameraRoi])

  const getRelativeCoordinates = useCallback(
    (event: React.MouseEvent | MouseEvent): Point => {
      if (!roiImageRef.current || !imageAreaRef.current) {
        return { x: 0, y: 0 }
      }

      const areaRect = imageAreaRef.current.getBoundingClientRect()
      const mouseX = event.clientX - areaRect.left
      const mouseY = event.clientY - areaRect.top

      const xOffset = mouseX - offsetRef.current.x - panOffset.x
      const yOffset = mouseY - offsetRef.current.y - panOffset.y

      let x = xOffset / scale
      let y = yOffset / scale

      x = Math.max(0, Math.min(x, imageDims.width))
      y = Math.max(0, Math.min(y, imageDims.height))

      return { x, y }
    },
    [scale, panOffset, imageDims]
  )

  const isPointInROI = useCallback((point: Point, anno: Annotation): boolean => {
    if (anno.type === 'rectangle') {
      const rect = anno as RectAnnotation
      const xMin = Math.min(rect.x1, rect.x2)
      const xMax = Math.max(rect.x1, rect.x2)
      const yMin = Math.min(rect.y1, rect.y2)
      const yMax = Math.max(rect.y1, rect.y2)
      return point.x >= xMin && point.x <= xMax && point.y >= yMin && point.y <= yMax
    } else if (anno.type === 'polygon') {
      return GEOMETRY_UTILS.isPointInPolygon(point, (anno as PolygonAnnotation).points)
    }
    return false
  }, [])

  const finalizeCurrentPolygon = useCallback(() => {
    if (!isPolygonDrawing || currentPolygonPoints.length < 3) return

    const bounds = GEOMETRY_UTILS.getPolygonBounds(currentPolygonPoints)

    setLastSavedAnnotation({
      type: 'polygon',
      points: [...currentPolygonPoints],
      x1: bounds.x1,
      y1: bounds.y1,
      x2: bounds.x2,
      y2: bounds.y2,
      label: '',
      useCases: [],
    } as PolygonAnnotation)
    setIsPolygonDrawing(false)
    setShowLabelingModal(true)
  }, [isPolygonDrawing, currentPolygonPoints])

  const cancelCurrentPolygonDraft = useCallback(() => {
    if (!isPolygonDrawing) return
    setIsPolygonDrawing(false)
    setCurrentPolygonPoints([])
    setTempPoint({ x: 0, y: 0 })
  }, [isPolygonDrawing])

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && drawingMode === 'polygon' && isPolygonDrawing) {
        event.preventDefault()
        cancelCurrentPolygonDraft()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [drawingMode, isPolygonDrawing, cancelCurrentPolygonDraft])

  const startVertexDrag = useCallback((handle: string, point: Point) => {
    if (drawingMode !== 'edit' || selectedAnnotationIndex === null) return
    setActiveHandle(handle)
    setIsDraggingHandle(true)
    setLastMousePos(point)
  }, [drawingMode, selectedAnnotationIndex])

  const selectTool = useCallback((tool: Tool) => {
    setIsDrawing(false)
    setIsPolygonDrawing(false)
    setCurrentPolygonPoints([])
    setIsDraggingHandle(false)
    setActiveHandle(null)
    setShowLabelingModal(false)

    if (selectedTool?.label === tool.label) {
      setSelectedTool(null)
      setDrawingMode(null)
      setSelectedAnnotationIndex(null)
    } else {
      setSelectedTool(tool)
      setSelectedAnnotationIndex(null)

      switch (tool.label) {
        case 'Rectangle ROI':
          setDrawingMode('rectangle')
          break
        case 'Polygon ROI':
          setDrawingMode('polygon')
          break
        case 'Edit ROI':
          setDrawingMode('edit')
          break
        default:
          setDrawingMode(null)
      }
    }
  }, [selectedTool])

  const handleImageMousedown = useCallback(
    (event: React.MouseEvent) => {
      event.preventDefault()
      if (!roiImageRef.current || showLabelingModal) return

      const { x, y } = getRelativeCoordinates(event)

      if (
        !isDrawing &&
        !isPolygonDrawing &&
        isZoomedIn &&
        (event.button === 1 || (event.button === 0 && drawingMode === null))
      ) {
        setIsPanning(true)
        setPanStart({ x: event.clientX - panOffset.x, y: event.clientY - panOffset.y })
        return
      }

      if (drawingMode === 'rectangle') {
        setIsDrawing(true)
        setCurrentRect({ x1: x, y1: y, x2: x, y2: y })
      }

      if (drawingMode === 'polygon' && event.button === 0) {
        const isNearStart =
          currentPolygonPoints.length >= 3 &&
          GEOMETRY_UTILS.getDistance({ x, y }, currentPolygonPoints[0]) <=
          ROI_EDITOR_CONFIG.CLOSE_POLYGON_THRESHOLD

        if (isNearStart) {
          finalizeCurrentPolygon()
        } else {
          setIsPolygonDrawing(true)
          setCurrentPolygonPoints((prev) => [...prev, { x, y }])
        }
      }

      if (drawingMode === 'edit') {
        for (let i = 0; i < annotations.length; i++) {
          if (annotations[i].isLocked) continue

          if (isPointInROI({ x, y }, annotations[i])) {
            setSelectedAnnotationIndex(i)
            setActiveHandle(null)
            setIsDraggingHandle(true)
            setLastMousePos({ x, y })
            return
          }
        }
        setSelectedAnnotationIndex(null)
        setActiveHandle(null)
        setIsDraggingHandle(false)
      }
    },
    [
      showLabelingModal,
      getRelativeCoordinates,
      isDrawing,
      isPolygonDrawing,
      drawingMode,
      isZoomedIn,
      panOffset,
      currentPolygonPoints,
      annotations,
      isPointInROI,
      finalizeCurrentPolygon,
    ]
  )

  const handleImageMousemove = useCallback(
    (event: React.MouseEvent | MouseEvent) => {
      if (!roiImageRef.current) return

      if (isPanning) {
        setPanOffset({
          x: event.clientX - panStart.x,
          y: event.clientY - panStart.y,
        })
        return
      }

      const { x, y } = getRelativeCoordinates(event)

      if (drawingMode === 'rectangle' && isDrawing) {
        setCurrentRect((prev) => ({ ...prev, x2: x, y2: y }))
      } else if (drawingMode === 'polygon' && isPolygonDrawing) {
        setTempPoint({ x, y })
      } else if (drawingMode === 'edit' && isDraggingHandle && selectedAnnotationIndex !== null) {
        setAnnotations((prev) =>
          prev.map((anno, index) => {
            if (index !== selectedAnnotationIndex) return anno

            if (activeHandle?.startsWith('rect-') && anno.type === 'rectangle') {
              const handleIndex = Number(activeHandle.split('-')[1])

              switch (handleIndex) {
                case 0:
                  return { ...anno, x1: x, y1: y }
                case 1:
                  return { ...anno, x2: x, y1: y }
                case 2:
                  return { ...anno, x2: x, y2: y }
                case 3:
                  return { ...anno, x1: x, y2: y }
                default:
                  return anno
              }
            }

            if (activeHandle?.startsWith('poly-') && anno.type === 'polygon') {
              const vertexIndex = Number(activeHandle.split('-')[1])
              if (!Number.isInteger(vertexIndex) || vertexIndex < 0 || vertexIndex >= anno.points.length) {
                return anno
              }

              const updatedPoints = anno.points.map((point, idx) =>
                idx === vertexIndex ? { x, y } : point
              )
              const bounds = GEOMETRY_UTILS.getPolygonBounds(updatedPoints)

              return {
                ...anno,
                points: updatedPoints,
                x1: bounds.x1,
                y1: bounds.y1,
                x2: bounds.x2,
                y2: bounds.y2,
              }
            }

            const dx = x - lastMousePos.x
            const dy = y - lastMousePos.y
            if (dx === 0 && dy === 0) return anno

            if (anno.type === 'rectangle') {
              return {
                ...anno,
                x1: anno.x1 + dx,
                y1: anno.y1 + dy,
                x2: anno.x2 + dx,
                y2: anno.y2 + dy,
              }
            }

            const movedPoints = anno.points.map((point) => ({
              x: point.x + dx,
              y: point.y + dy,
            }))
            const bounds = GEOMETRY_UTILS.getPolygonBounds(movedPoints)

            return {
              ...anno,
              points: movedPoints,
              x1: bounds.x1,
              y1: bounds.y1,
              x2: bounds.x2,
              y2: bounds.y2,
            }
          })
        )
        setLastMousePos({ x, y })
      }
    },
    [
      isPanning,
      panStart,
      getRelativeCoordinates,
      drawingMode,
      isDrawing,
      isPolygonDrawing,
      isDraggingHandle,
      selectedAnnotationIndex,
      lastMousePos.x,
      lastMousePos.y,
      activeHandle,
    ]
  )

  const handleImageMouseup = useCallback(() => {
    if (isPanning) {
      setIsPanning(false)
      return
    }

    if (isDraggingHandle) {
      setIsDraggingHandle(false)
      setActiveHandle(null)
      return
    }

    if (drawingMode === 'rectangle' && isDrawing) {
      setIsDrawing(false)

      const x1 = Math.min(currentRect.x1, currentRect.x2)
      const y1 = Math.min(currentRect.y1, currentRect.y2)
      const x2 = Math.max(currentRect.x1, currentRect.x2)
      const y2 = Math.max(currentRect.y1, currentRect.y2)

      const width = Math.abs(x2 - x1)
      const height = Math.abs(y2 - y1)

      if (width > ROI_EDITOR_CONFIG.MIN_RECTANGLE_SIZE && height > ROI_EDITOR_CONFIG.MIN_RECTANGLE_SIZE) {
        setLastSavedAnnotation({
          type: 'rectangle',
          x1,
          y1,
          x2,
          y2,
          label: '',
          useCases: [],
        } as RectAnnotation)
        setShowLabelingModal(true)
      }
    }
  }, [
    isPanning,
    isDraggingHandle,
    drawingMode,
    isDrawing,
    currentRect,
  ])

  const saveAnnotation = useCallback((useCasesOverride?: string[]) => {
    const useCasesToSave = useCasesOverride ?? selectedUseCases

    if (lastSavedAnnotation && useCasesToSave.length > 0) {
      const newAnnotation = {
        ...lastSavedAnnotation,
        label: useCasesToSave[0],
        useCases: [...useCasesToSave],
        isLocked: false,
        isVisible: true,
      }

      setAnnotations((prev) => [...prev, newAnnotation])
      setLastSavedAnnotation(null)
      setSelectedUseCases([])
      setShowLabelingModal(false)
      setCurrentPolygonPoints([])
    }
  }, [lastSavedAnnotation, selectedUseCases])

  const cancelLabeling = useCallback(() => {
    setShowLabelingModal(false)
    setLastSavedAnnotation(null)
    setSelectedUseCases([])
    setCurrentPolygonPoints([])
  }, [])

  const zoomIn = useCallback(() => {
    setScale((prev) => Math.min(prev + ROI_EDITOR_CONFIG.SCALE_STEP, ROI_EDITOR_CONFIG.MAX_SCALE))
  }, [])

  const zoomOut = useCallback(() => {
    setScale((prev) => Math.max(prev - ROI_EDITOR_CONFIG.SCALE_STEP, ROI_EDITOR_CONFIG.MIN_SCALE))
  }, [])

  const resetZoom = useCallback(() => {
    setScale(initialFitScale)
    setPanOffset({ x: 0, y: 0 })
  }, [initialFitScale])

  const handleZoomWheel = useCallback(
    (event: React.WheelEvent) => {
      event.preventDefault()
      const delta = event.deltaY > 0 ? -ROI_EDITOR_CONFIG.SCALE_STEP : ROI_EDITOR_CONFIG.SCALE_STEP
      setScale((prev) =>
        Math.max(ROI_EDITOR_CONFIG.MIN_SCALE, Math.min(prev + delta, ROI_EDITOR_CONFIG.MAX_SCALE))
      )
    },
    []
  )

  const deleteAnnotation = useCallback((index: number) => {
    setAnnotations((prev) => prev.filter((_, i) => i !== index))
    setSelectedAnnotationIndex(null)
  }, [])

  const toggleLockAnnotation = useCallback((index: number) => {
    setAnnotations((prev) =>
      prev.map((anno, i) => (i === index ? { ...anno, isLocked: !anno.isLocked } : anno))
    )
  }, [])

  const toggleVisibilityAnnotation = useCallback((index: number) => {
    setAnnotations((prev) =>
      prev.map((anno, i) => (i === index ? { ...anno, isVisible: !(anno.isVisible ?? true) } : anno))
    )
  }, [])

  const openUseCaseEditor = useCallback((index: number) => {
    setEditingUseCaseIndex(index)
    setTempUseCases(annotations[index].useCases || [])
    setShowUseCaseModal(true)
  }, [annotations])

  const saveUseCaseEdit = useCallback((useCasesOverride?: string[]) => {
    const useCasesToSave = useCasesOverride ?? tempUseCases

    if (editingUseCaseIndex !== null && useCasesToSave.length > 0) {
      setAnnotations((prev) =>
        prev.map((anno, i) =>
          i === editingUseCaseIndex
            ? { ...anno, useCases: [...useCasesToSave], label: useCasesToSave[0] }
            : anno
        )
      )
    }
    setShowUseCaseModal(false)
    setEditingUseCaseIndex(null)
    setTempUseCases([])
  }, [editingUseCaseIndex, tempUseCases])

  const closeUseCaseModal = useCallback(() => {
    setShowUseCaseModal(false)
    setEditingUseCaseIndex(null)
    setTempUseCases([])
  }, [])

  const saveAnnotations = useCallback(async (): Promise<boolean> => {
    const token = getAuthToken()
    const parsedCameraId = Number(cameraId)

    if (!token || Number.isNaN(parsedCameraId)) {
      notify('error', 'Unable to save ROI: invalid camera or auth token.')
      return false
    }

    setIsSavingAnnotations(true)

    try {
      await new Promise((resolve) => setTimeout(resolve, 500))

      const baseWidth = imageDims.naturalWidth || imageDims.width
      const baseHeight = imageDims.naturalHeight || imageDims.height

      if (!baseWidth || !baseHeight) {
        notify('error', 'Unable to save ROI: image dimensions are not ready.')
        return false
      }

      const roiPayload = annotations.map((annotation) => {
        const mappedUsecases = (annotation.useCases || []).map((usecaseName) => {
          const matched = useCases.find((usecase) => usecase.usecaseName === usecaseName)
          return matched
            ? { usecaseId: matched.usecaseId, usecaseName }
            : { usecaseName }
        })

        if (annotation.type === 'rectangle') {
          const x1 = Math.min(annotation.x1, annotation.x2)
          const y1 = Math.min(annotation.y1, annotation.y2)
          const x2 = Math.max(annotation.x1, annotation.x2)
          const y2 = Math.max(annotation.y1, annotation.y2)

          return {
            type: 'rect',
            x: x1 / baseWidth,
            y: y1 / baseHeight,
            w: (x2 - x1) / baseWidth,
            h: (y2 - y1) / baseHeight,
            usecases: mappedUsecases,
          }
        }

        return {
          type: 'polygon',
          points: annotation.points.map((point) => [
            point.x / baseWidth,
            point.y / baseHeight,
          ]),
          usecases: mappedUsecases,
        }
      })

      const response = await updateRoi({
        cameraId: parsedCameraId,
        roi: roiPayload,
      })

      if (response.code !== 200) {
        notify('error', response.message || 'Failed to save annotations')
        return false
      }

      notify('success', response.message || 'Annotations saved successfully!')
      return true
    } catch {
      notify('error', 'Failed to save annotations')
      return false
    } finally {
      setIsSavingAnnotations(false)
    }
  }, [annotations, imageDims, cameraId, notify, useCases])

  const getRectStyle = useCallback(
    (rect: RectCoords) => {
      const x = Math.min(rect.x1, rect.x2)
      const y = Math.min(rect.y1, rect.y2)
      const width = Math.abs(rect.x2 - rect.x1)
      const height = Math.abs(rect.y2 - rect.y1)

      return {
        left: `${x}px`,
        top: `${y}px`,
        width: `${width}px`,
        height: `${height}px`,
      }
    },
    []
  )

  const getPolygonLabelStyle = useCallback((points: Point[]) => {
    const center = GEOMETRY_UTILS.getPolygonCenter(points)
    return {
      left: `${center.x}px`,
      top: `${center.y}px`,
      transform: 'translate(-50%, -50%)',
    }
  }, [])

  const polygonPointsToString = useCallback((points: Point[]) => {
    return points.map((p) => `${p.x},${p.y}`).join(' ')
  }, [])

  return {
    cameraId,
    selectedTool,
    drawingMode,
    showLabelingModal,
    showUseCaseModal,
    isLoadingImage,
    imageLoadError,
    isLoadingAnnotations,
    isSavingAnnotations,
    imageSrc,
    cameraDetails,
    labelOptions,
    useCases,
    annotations,
    selectedAnnotationIndex,
    isDrawing,
    isPolygonDrawing,
    isDraggingHandle,
    currentRect,
    currentPolygonPoints,
    tempPoint,
    activeHandle,
    lastMousePos,
    lastSavedAnnotation,
    selectedUseCases,
    editingUseCaseIndex,
    tempUseCases,
    scale,
    isPanning,
    panOffset,
    panStart,
    imageDims,
    tools,
    initialFitScale,
    isZoomedIn,
    polygonAnnotations,
    canvasTransformStyle,

    imageAreaRef,
    roiImageRef,
    transformWrapperRef,
    toastRef,

    setSelectedTool,
    setDrawingMode,
    setShowLabelingModal,
    setShowUseCaseModal,
    setAnnotations,
    setSelectedAnnotationIndex,
    setIsDrawing,
    setIsPolygonDrawing,
    setSelectedUseCases,
    setTempUseCases,
    setImageSrc,

    setImageDimensions,
    getRelativeCoordinates,
    isPointInROI,
    selectTool,
    zoomIn,
    zoomOut,
    resetZoom,
    handleZoomWheel,
    deleteAnnotation,
    toggleLockAnnotation,
    toggleVisibilityAnnotation,
    openUseCaseEditor,
    saveUseCaseEdit,
    closeUseCaseModal,
    saveAnnotations,
    saveAnnotation,
    cancelLabeling,
    finalizeCurrentPolygon,
    refreshCanvasFrame,
    startVertexDrag,
    handleImageMousedown,
    handleImageMousemove,
    handleImageMouseup,
    getRectStyle,
    getPolygonLabelStyle,
    polygonPointsToString,
  }
}
