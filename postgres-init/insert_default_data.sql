-- =============================================================================
-- AI VMS Default Data Insertion
-- Inserts default roles, resources, scopes, usecases, users, and permissions
-- =============================================================================

-- =============================================================================
-- INSERT DEFAULT ROLES
-- =============================================================================

INSERT INTO roles (name, is_active, created_at, updated_at)
VALUES 
    ('Super Admin', true, NOW(), NOW()),
    ('Admin', true, NOW(), NOW()),
    ('Manager', true, NOW(), NOW()),
    ('Operator', true, NOW(), NOW())
ON CONFLICT (name) DO NOTHING;

-- =============================================================================
-- INSERT DEFAULT RESOURCES
-- =============================================================================

INSERT INTO resources (name, is_active, created_at, updated_at)
VALUES 
    ('camera', true, NOW(), NOW()),
    ('usecase', true, NOW(), NOW()),
    ('user', true, NOW(), NOW()),
    ('role', true, NOW(), NOW()),
    ('report', true, NOW(), NOW()),
    ('setting', true, NOW(), NOW())
ON CONFLICT (name) DO NOTHING;

-- =============================================================================
-- INSERT DEFAULT SCOPES
-- =============================================================================

INSERT INTO scopes (name, is_active, created_at, updated_at)
VALUES 
    ('create', true, NOW(), NOW()),
    ('read', true, NOW(), NOW()),
    ('update', true, NOW(), NOW()),
    ('delete', true, NOW(), NOW()),
    ('*', true, NOW(), NOW())
ON CONFLICT (name) DO NOTHING;

-- =============================================================================
-- INSERT DEFAULT USECASES
-- =============================================================================

INSERT INTO usecases (name, description, is_active, created_at, updated_at)
VALUES 
    ('Cash Handling', 'Monitor cash handling operations', true, NOW(), NOW()),
    ('Teller Operations', 'Teller desk monitoring and transaction verification', true, NOW(), NOW()),
    ('Vault Access', 'Vault access and security monitoring', true, NOW(), NOW()),
    ('Customer Interactions', 'Monitor customer service areas', true, NOW(), NOW()),
    ('Exit/Entry', 'Track entry and exit points', true, NOW(), NOW()),
    ('Storage Area', 'Monitor storage and archive areas', true, NOW(), NOW())
ON CONFLICT (name) DO NOTHING;

-- =============================================================================
-- INSERT DEFAULT USECASE CLASSES
-- =============================================================================

INSERT INTO usecase_classes (usecase_id, class_name, created_at)
SELECT u.id, v.class_name, NOW()
FROM usecases u
JOIN (
    VALUES
        ('Cash Handling', 'person'),
        ('Cash Handling', 'cash'),
        ('Teller Operations', 'person'),
        ('Teller Operations', 'counter'),
        ('Vault Access', 'person'),
        ('Vault Access', 'vault'),
        ('Customer Interactions', 'person'),
        ('Exit/Entry', 'person'),
        ('Exit/Entry', 'door'),
        ('Storage Area', 'person'),
        ('Storage Area', 'box')
) AS v(usecase_name, class_name)
    ON u.name = v.usecase_name
ON CONFLICT (usecase_id, class_name) DO NOTHING;


-- =============================================================================
-- INSERT ROLE PERMISSIONS FOR SUPER ADMIN
-- =============================================================================

INSERT INTO role_permissions (role_id, resource_id, scope_id, created_at, updated_at)
SELECT 
    r.id,
    res.id,
    s.id,
    NOW(),
    NOW()
FROM roles r
CROSS JOIN resources res
CROSS JOIN scopes s
WHERE r.name = 'Super Admin'
ON CONFLICT (role_id, resource_id, scope_id) DO NOTHING;

-- =============================================================================
-- INSERT ROLE PERMISSIONS FOR ADMIN
-- =============================================================================

INSERT INTO role_permissions (role_id, resource_id, scope_id, created_at, updated_at)
SELECT 
    r.id,
    res.id,
    s.id,
    NOW(),
    NOW()
FROM roles r
CROSS JOIN resources res
CROSS JOIN scopes s
WHERE r.name = 'Admin' 
    AND s.name IN ('read', 'update', 'create')
ON CONFLICT (role_id, resource_id, scope_id) DO NOTHING;

-- =============================================================================
-- INSERT ROLE PERMISSIONS FOR MANAGER
-- =============================================================================

INSERT INTO role_permissions (role_id, resource_id, scope_id, created_at, updated_at)
SELECT 
    r.id,
    res.id,
    s.id,
    NOW(),
    NOW()
FROM roles r
CROSS JOIN resources res
CROSS JOIN scopes s
WHERE r.name = 'Manager'
    AND res.name IN ('camera', 'usecase', 'report')
    AND s.name IN ('read', 'update')
ON CONFLICT (role_id, resource_id, scope_id) DO NOTHING;

-- =============================================================================
-- INSERT ROLE PERMISSIONS FOR OPERATOR
-- =============================================================================

INSERT INTO role_permissions (role_id, resource_id, scope_id, created_at, updated_at)
SELECT 
    r.id,
    res.id,
    s.id,
    NOW(),
    NOW()
FROM roles r
CROSS JOIN resources res
CROSS JOIN scopes s
WHERE r.name = 'Operator'
    AND res.name IN ('camera', 'report')
    AND s.name = 'read'
ON CONFLICT (role_id, resource_id, scope_id) DO NOTHING;
-- =============================================================================
-- INSERT DEFAULT SUPER ADMIN USER
-- =============================================================================

INSERT INTO users (
    email,
    hashed_password,
    first_name,
    last_name,
    role_id,
    is_active,
    created_at,
    updated_at,
    created_by,
    updated_by,
    is_delete
)
SELECT 
    'superadmin@ai-vms.com',
    '$2b$12$niUXbdx9b9AdVnJf59MjfuiSrE6j52nvlT.3g0u9YnIakNGmBW6iK', -- Password: admin@123
    'Super',
    'Admin',
    r.id,
    true,
    NOW(),
    NOW(),
    NULL,
    NULL,
    false
FROM roles r
WHERE r.name = 'Super Admin'
ON CONFLICT (email) DO NOTHING;

-- Default data insertion complete!
