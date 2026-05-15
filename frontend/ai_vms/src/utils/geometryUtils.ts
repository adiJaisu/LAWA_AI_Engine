import type { Point, RectCoords } from '../types/roi.types';

export const GEOMETRY_UTILS = {
  isPointInPolygon(point: Point, vertices: Point[]): boolean {
    if (!vertices || vertices.length < 3) return false;

    const { x, y } = point;
    let isInside = false;

    for (let i = 0, j = vertices.length - 1; i < vertices.length; j = i++) {
      const xi = vertices[i].x;
      const yi = vertices[i].y;
      const xj = vertices[j].x;
      const yj = vertices[j].y;

      const intersect = (yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi;
      if (intersect) isInside = !isInside;
    }

    return isInside;
  },

  getDistance(p1: Point, p2: Point): number {
    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    return Math.sqrt(dx * dx + dy * dy);
  },

  getRectCenter(rect: RectCoords): Point {
    return {
      x: (rect.x1 + rect.x2) / 2,
      y: (rect.y1 + rect.y2) / 2,
    };
  },

  getPolygonCenter(vertices: Point[]): Point {
    const sumX = vertices.reduce((sum, p) => sum + p.x, 0);
    const sumY = vertices.reduce((sum, p) => sum + p.y, 0);
    return {
      x: sumX / vertices.length,
      y: sumY / vertices.length,
    };
  },

  getPolygonBounds(vertices: Point[]): RectCoords {
    const xCoords = vertices.map((p) => p.x);
    const yCoords = vertices.map((p) => p.y);

    return {
      x1: Math.min(...xCoords),
      y1: Math.min(...yCoords),
      x2: Math.max(...xCoords),
      y2: Math.max(...yCoords),
    };
  },

  normalizeRect(rect: RectCoords): RectCoords {
    return {
      x1: Math.min(rect.x1, rect.x2),
      y1: Math.min(rect.y1, rect.y2),
      x2: Math.max(rect.x1, rect.x2),
      y2: Math.max(rect.y1, rect.y2),
    };
  },
} as const;
