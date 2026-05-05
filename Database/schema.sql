-- =============================================================================

-- AI VMS Database Schema Creation

-- Creates all tables, enums, and indexes

-- =============================================================================

-- Enable necessary extensions

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================

-- CREATE ENUMS

-- =============================================================================

DO $$ 

BEGIN

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'camera_type') THEN

        CREATE TYPE camera_type AS ENUM ('ip', 'usb');

    END IF;

END

$$;

-- =============================================================================

-- CREATE ROLES TABLE

-- =============================================================================

CREATE TABLE IF NOT EXISTS roles (

    id BIGSERIAL PRIMARY KEY,

    name VARCHAR(100) NOT NULL UNIQUE,

    is_active BOOLEAN NOT NULL DEFAULT true,

    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    created_by BIGINT,

    updated_by BIGINT

);

CREATE INDEX IF NOT EXISTS ix_roles_name ON roles(name);

CREATE INDEX IF NOT EXISTS ix_roles_is_active ON roles(is_active);

-- =============================================================================

-- CREATE RESOURCES TABLE

-- =============================================================================

CREATE TABLE IF NOT EXISTS resources (

    id BIGSERIAL PRIMARY KEY,

    name VARCHAR(120) NOT NULL UNIQUE,

    is_active BOOLEAN NOT NULL DEFAULT true,

    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    created_by BIGINT,

    updated_by BIGINT

);

CREATE INDEX IF NOT EXISTS ix_resources_name ON resources(name);

CREATE INDEX IF NOT EXISTS ix_resources_is_active ON resources(is_active);

-- =============================================================================

-- CREATE SCOPES TABLE

-- =============================================================================

CREATE TABLE IF NOT EXISTS scopes (

    id BIGSERIAL PRIMARY KEY,

    name VARCHAR(120) NOT NULL UNIQUE,

    is_active BOOLEAN NOT NULL DEFAULT true,

    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    created_by BIGINT,

    updated_by BIGINT

);

CREATE INDEX IF NOT EXISTS ix_scopes_name ON scopes(name);

CREATE INDEX IF NOT EXISTS ix_scopes_is_active ON scopes(is_active);

-- =============================================================================

-- CREATE USECASES TABLE

-- =============================================================================

CREATE TABLE IF NOT EXISTS usecases (

    id BIGSERIAL PRIMARY KEY,

    name VARCHAR(120) NOT NULL UNIQUE,

    description VARCHAR(500),

    is_active BOOLEAN NOT NULL DEFAULT true,

    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    created_by BIGINT,

    updated_by BIGINT

);

CREATE INDEX IF NOT EXISTS ix_usecases_name ON usecases(name);

CREATE INDEX IF NOT EXISTS ix_usecases_is_active ON usecases(is_active);

-- =============================================================================

-- CREATE USERS TABLE

-- =============================================================================

CREATE TABLE IF NOT EXISTS users (

    id BIGSERIAL PRIMARY KEY,

    email VARCHAR(255) NOT NULL UNIQUE,

    hashed_password VARCHAR(255),

    first_name VARCHAR(120),

    last_name VARCHAR(120),

    role_id BIGINT NOT NULL REFERENCES roles(id),

    is_active BOOLEAN NOT NULL DEFAULT true,

    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    created_by BIGINT,

    updated_by BIGINT,

    is_delete BOOLEAN NOT NULL DEFAULT false

);

CREATE INDEX IF NOT EXISTS ix_users_email ON users(email);

CREATE INDEX IF NOT EXISTS ix_users_role ON users(role_id);

CREATE INDEX IF NOT EXISTS ix_users_is_active ON users(is_active);

CREATE INDEX IF NOT EXISTS ix_users_is_delete ON users(is_delete);

-- =============================================================================

-- CREATE CAMERAS TABLE

-- =============================================================================

CREATE TABLE IF NOT EXISTS cameras (

    id BIGSERIAL PRIMARY KEY,

    name VARCHAR(150) NOT NULL UNIQUE,

    type camera_type NOT NULL,

    rtsp_url VARCHAR(500),

    roi_frame_blob BYTEA,

    resolution VARCHAR(50),

    fps BIGINT,

    codec VARCHAR(50),

    roi JSONB,

    is_active BOOLEAN NOT NULL DEFAULT true,

    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    created_by BIGINT,

    updated_by BIGINT,

    is_delete BOOLEAN NOT NULL DEFAULT false

);

CREATE INDEX IF NOT EXISTS ix_cameras_name ON cameras(name);

CREATE INDEX IF NOT EXISTS ix_cameras_is_active ON cameras(is_active);

CREATE INDEX IF NOT EXISTS ix_cameras_is_delete ON cameras(is_delete);

CREATE INDEX IF NOT EXISTS ix_cameras_created_at ON cameras(created_at);

-- =============================================================================

-- CREATE ROLE PERMISSIONS TABLE

-- =============================================================================

CREATE TABLE IF NOT EXISTS role_permissions (

    id BIGSERIAL PRIMARY KEY,

    role_id BIGINT NOT NULL REFERENCES roles(id),

    resource_id BIGINT NOT NULL REFERENCES resources(id),

    scope_id BIGINT NOT NULL REFERENCES scopes(id),

    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    created_by BIGINT,

    updated_by BIGINT,

    CONSTRAINT uq_role_permissions UNIQUE (role_id, resource_id, scope_id)

);

CREATE INDEX IF NOT EXISTS ix_role_permissions_role ON role_permissions(role_id);

CREATE INDEX IF NOT EXISTS ix_role_permissions_resource ON role_permissions(resource_id);

CREATE INDEX IF NOT EXISTS ix_role_permissions_scope ON role_permissions(scope_id);

-- =============================================================================

-- CREATE CAMERA USECASE TABLE

-- =============================================================================

CREATE TABLE IF NOT EXISTS camera_usecase (

    id BIGSERIAL PRIMARY KEY,

    camera_id BIGINT NOT NULL REFERENCES cameras(id),

    usecase_id BIGINT NOT NULL REFERENCES usecases(id),

    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    created_by BIGINT,

    updated_by BIGINT,

    CONSTRAINT uq_camera_usecase UNIQUE (camera_id, usecase_id)

);

CREATE INDEX IF NOT EXISTS ix_camera_usecase_camera ON camera_usecase(camera_id);

CREATE INDEX IF NOT EXISTS ix_camera_usecase_usecase ON camera_usecase(usecase_id);

-- All tables created successfully!
 
-- =============================================================================
-- CREATE EVENTS TABLE
-- =============================================================================
CREATE TABLE IF NOT EXISTS events (
    id BIGSERIAL PRIMARY KEY,
    camera_id VARCHAR(150),
    usecase_name VARCHAR(120),
    event_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    evidence_path VARCHAR(500),
    event_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_events_camera ON events(camera_id);
CREATE INDEX IF NOT EXISTS ix_events_time ON events(event_time);
