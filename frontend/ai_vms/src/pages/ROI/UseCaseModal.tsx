import React, { useEffect, useState } from 'react'
import '../../styles/modal.css'

interface UseCaseModalProps {
  isOpen: boolean
  availableUsecases: string[]
  tempUseCases: string[]
  setTempUseCases: (useCases: string[]) => void
  onSave: (useCases?: string[]) => void
  onClose: () => void
}

export const UseCaseModal: React.FC<UseCaseModalProps> = ({
  isOpen,
  availableUsecases,
  tempUseCases,
  setTempUseCases,
  onSave,
  onClose,
}) => {
  const [selectedUseCases, setSelectedUseCases] = useState<string[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!isOpen) return
    if (availableUsecases.length === 0) {
      setSelectedUseCases([])
      return
    }

    const allowed = new Set(availableUsecases)
    setSelectedUseCases(tempUseCases.filter((usecase) => allowed.has(usecase)))
  }, [isOpen, availableUsecases, tempUseCases])

  if (!isOpen) return null

  const toggleUsecase = (usecaseName: string) => {
    setSelectedUseCases((prev) =>
      prev.includes(usecaseName)
        ? prev.filter((x) => x !== usecaseName)
        : [...prev, usecaseName]
    )
  }

  const handleSave = async () => {
    if (selectedUseCases.length === 0) return

    setLoading(true)

    try {
      setTempUseCases(selectedUseCases)
      onSave(selectedUseCases)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal usecase-modal" onClick={(e) => e.stopPropagation()}>
        <div className="usecase-title">Edit Use Cases</div>

        <div className="usecase-list">
          {availableUsecases.length === 0 && (
            <div className="usecase-empty">No use cases available</div>
          )}
          {availableUsecases.map((usecaseName) => (
            <label key={usecaseName} className="usecase-item">
              <input
                type="checkbox"
                className="usecase-checkbox"
                checked={selectedUseCases.includes(usecaseName)}
                onChange={() => toggleUsecase(usecaseName)}
              />
              {usecaseName}
            </label>
          ))}
        </div>

        <div className="modal-actions">
          <button className="btn btn-secondary" onClick={onClose}>
            Cancel
          </button>
          <button
            className="btn btn-primary"
            onClick={handleSave}
            disabled={loading || selectedUseCases.length === 0}
          >
            {loading ? 'Saving...' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  )
}
