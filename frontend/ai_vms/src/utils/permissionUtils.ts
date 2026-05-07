export const getPermissions = (): string[] => {
  const stored = sessionStorage.getItem("permissions")
  return stored ? JSON.parse(stored) : []
}

export const getRoleId = (): number | null => {
  const stored = sessionStorage.getItem("user")
  if (stored) {
    try {
      const parsedUser = JSON.parse(stored)
      return parsedUser.role_id || null
    } catch (error) {
      console.error("Invalid user data in sessionStorage")
      return null
    }
  }
  return null
}

export const hasPermission = (permission: string): boolean => {
  const permissions = getPermissions()
  if (permissions.includes("*:*") || permissions.includes(permission)) {
    return true
  }

  const [resource, scope] = permission.split(":")
  if (!resource || !scope) {
    return false
  }

  return permissions.includes(`${resource}:*`) || permissions.includes(`*:${scope}`)
}

export const checkPermission = (resource: string, scope: string) => {
  return hasPermission(`${resource}:${scope}`)
}
