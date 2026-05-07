import { createContext, useContext } from 'react'
import type { useRoiEditor } from '../hooks/useRoiEditor'

export type RoiEditorContextType = ReturnType<typeof useRoiEditor>

export const RoiEditorContext = createContext<RoiEditorContextType | null>(null)

export const useRoiEditorContext = () => {
  const context = useContext(RoiEditorContext)
  if (!context) {
    throw new Error('useRoiEditorContext must be used within RoiEditorProvider')
  }
  return context
}
