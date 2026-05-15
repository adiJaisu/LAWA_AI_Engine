import React from 'react'
import { useRoiEditorContext } from '../../context/RoiEditorContext'
import { ROI_EDITOR_CONFIG } from '../../constants/roiEditorConfig'
import type { Tool } from '../../types/roi.types'
import '../../styles/ROI/LeftPanel.css'

export const LeftPanel: React.FC = () => {
  const { tools, selectedTool, selectTool, annotations } = useRoiEditorContext()

  const hasAnnotations = annotations.length > 0

  const isToolDisabled = (tool: Tool) => {
    const toolLabel = tool.label

    if (toolLabel === ROI_EDITOR_CONFIG.TOOL_NAMES.EDIT && !hasAnnotations) {
      return true
    }

    return false
  }

  const getToolTooltip = (tool: Tool) => {
    const toolLabel = tool.label

    if (toolLabel === ROI_EDITOR_CONFIG.TOOL_NAMES.EDIT && !hasAnnotations) {
      return 'Create at least one annotation to enable Edit Tool'
    }

    return tool.label
  }

  return (
    <div className="leftSidebar">
      {(tools as Tool[]).map((tool: Tool, index: number) => (
        <button
          key={`tool-${index}`}
          className={`toolBtn ${selectedTool?.label === tool.label ? 'active' : ''} ${
            isToolDisabled(tool) ? 'disabled' : ''
          }`}
          onClick={() => !isToolDisabled(tool) && selectTool(tool)}
          disabled={isToolDisabled(tool)}
          title={getToolTooltip(tool)}
          aria-label={tool.label}
        >
          <span className="material-icons">{tool.icon}</span>
        </button>
      ))}
    </div>
  )
}
