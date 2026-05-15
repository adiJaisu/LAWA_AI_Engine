import { Navigate } from "react-router-dom"
import { checkPermission } from "../utils/permissionUtils"

interface Props {
  resource: string
  scope: string
  children: React.ReactNode
}

const PermissionRoute = ({ resource, scope, children }: Props) => {
  const permission = checkPermission(resource, scope)
  if (!permission) {
    return <Navigate to="/auth/error" replace />
  }

  return <>{children}</>
}

export default PermissionRoute