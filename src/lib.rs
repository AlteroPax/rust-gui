//! Core math and rendering primitives for the demo GUI.
//!
//! This crate provides a tiny 3D toolkit: vectors, meshes, a camera,
//! a software canvas, and simple color helpers.

use std::f32::consts::PI;

/// 3D vector in a right‑handed coordinate system.
#[derive(Copy, Clone)]
pub struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3 {

    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Vec3 {
            x, y, z
        }
    }

    /// Returns this vector rotated around the Y axis by `angle` radians.
    pub fn rotate_y(self, angle: f32) -> Vec3 {
        Vec3 {
            x: self.x * angle.cos() - self.z * angle.sin(),
            y: self.y,
            z: self.x * angle.sin() + self.z * angle.cos(),
        }
    }

    /// Returns this vector rotated around the X axis by `angle` radians.
    pub fn rotate_x(self, angle: f32) -> Vec3 {
        Vec3 {
            x: self.x,
            y: self.y * angle.cos() - self.z * angle.sin(),
            z: self.y * angle.sin() + self.z * angle.cos(),
        }
    }

    /// Returns this vector rotated around the Z axis by `angle` radians.
    pub fn rotate_z(self, angle: f32) -> Vec3 {
        Vec3 {
            x: self.x * angle.cos() - self.y * angle.sin(),
            y: self.x * angle.sin() + self.y * angle.cos(),
            z: self.z,
        }
    }

    pub fn translate_y(self, distance: f32) -> Vec3 {
        Vec3 {
            x: self.x,
            y: self.y + distance,
            z: self.z
        }
    }

    pub fn translate_x(self, distance: f32) -> Vec3 {
        Vec3 {
            x: self.x + distance,
            y: self.y,
            z: self.z
        }
    }

    pub fn translate_z(self, distance: f32) -> Vec3 {
        Vec3 {
            x: self.x,
            y: self.y,
            z: self.z + distance,
        }
    }
}

/// Simple indexed mesh made of shared vertices, edges, and quad faces.
pub struct Mesh {
    vertices: Vec<Vec3>,
    edges: Vec<(usize, usize)>,
    faces: Vec<[usize; 4]>,
}

impl Mesh {
    /// Creates an axis‑aligned cube centered at the origin.
    ///
    /// The cube has side length `2 * half_extent`.
    pub fn cube(half_extent: f32) -> Self {
        let h = half_extent;

        let vertices = vec![
            Vec3 {
                x: -h,
                y: -h,
                z: -h,
            }, // 0
            Vec3 { x: h, y: -h, z: -h }, // 1
            Vec3 { x: h, y: h, z: -h },  // 2
            Vec3 { x: -h, y: h, z: -h }, // 3
            Vec3 { x: -h, y: -h, z: h }, // 4
            Vec3 { x: h, y: -h, z: h },  // 5
            Vec3 { x: h, y: h, z: h },   // 6
            Vec3 { x: -h, y: h, z: h },  // 7
        ];

        let edges = vec![
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0), // back
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4), // front
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7), // connecting
        ];

        let faces = vec![
            [0, 1, 2, 3], // back
            [5, 4, 7, 6], // front
            [0, 4, 5, 1], // bottom
            [2, 6, 7, 3], // top
            [0, 3, 7, 4], // left
            [1, 5, 6, 2], // right
        ];

        Self {
            vertices,
            edges,
            faces,
        }
    }

    /// Creates a UV sphere centered at the origin.
    ///
    /// `num_stacks` controls vertical subdivision, `num_slices` horizontal.
    pub fn sphere(radius: f32, num_stacks: usize, num_slices: usize) -> Self {
        let mut vertices = Vec::new();

        vertices.push(Vec3 {
            x: 0.0,
            y: radius,
            z: 0.0,
        }); // 0 = top pole

        for i in 1..num_stacks {
            let phi = PI * i as f32 / num_stacks as f32;
            for j in 0..num_slices {
                let theta = 2.0 * PI * j as f32 / num_slices as f32;
                vertices.push(Vec3 {
                    x: radius * phi.sin() * theta.cos(),
                    y: radius * phi.cos(),
                    z: radius * phi.sin() * theta.sin(),
                });
            }
        }

        vertices.push(Vec3 {
            x: 0.0,
            y: -radius,
            z: 0.0,
        }); // bottom pole

        // Build edges
        let mut edges: Vec<(usize, usize)> = Vec::new();
        let last_stack_start: usize = 1 + (num_stacks - 2) * num_slices;
        let bottom_pole = vertices.len() - 1;

        // Top pole -> first middle stack
        for j in 0..num_slices {
            edges.push((0, 1 + j));
        }

        // Middle stacks: horizontal rings + vertical connections
        for i in 0..(num_stacks - 2) {
            let row_start = 1 as usize + i * num_slices as usize;
            for j in 0..num_slices {
                let curr = row_start + j as usize;
                let next_in_row = row_start + ((j + 1) % num_slices);
                let below = row_start + num_slices as usize + j as usize;
                edges.push((curr, next_in_row));
                edges.push((curr, below));
            }
        }

        // Bottom row: horizontal ring
        for j in 0..num_slices {
            let curr = last_stack_start + j;
            let next_in_row = last_stack_start + ((j + 1) % num_slices);
            edges.push((curr, next_in_row));
            edges.push((curr, bottom_pole));
        }

        Self {
            vertices,
            edges,
            faces: Vec::new(), // placeholder for later
        }
    }

    /// Renders this mesh using a transform, camera, and target canvas.
    ///
    /// Vertices are transformed, projected, then drawn as points and/or edges.
    pub fn render<F>(
        &self,
        canvas: &mut Canvas,
        camera: &Camera,
        transform: F,
        draw_vertices: bool,
        draw_edges: bool,
        draw_faces: bool,
        color: Color,
    ) where
        F: Fn(Vec3) -> Vec3,
    {
        let verts_world: Vec<Vec3> = self.vertices.iter().map(|&v| transform(v)).collect();
        let verts_view: Vec<Vec3> = verts_world.iter().map(|&v| camera.world_to_view(v)).collect();
        let projected: Vec<(i32, i32)> = verts_view.iter().map(|&v| camera.project_view(v)).collect();

        if draw_faces && !self.faces.is_empty() {
            struct FaceToDraw {
                indices: [usize; 4],
                depth: f32,
            }

            let mut faces_to_draw = Vec::<FaceToDraw>::new();

            for face in &self.faces {
                let [i0, i1, i2, i3] = *face;

                let v0 = verts_view[i0];
                let v1 = verts_view[i1];
                let v2 = verts_view[i3];

                // Edge vectors in camera space
                let e1 = Vec3 { x: v1.x - v0.x, y: v1.y - v0.y, z: v1.z - v0.z };
                let e2 = Vec3 { x: v2.x - v0.x, y: v2.y - v0.y, z: v2.z - v0.z };

                // Face normal (right‑handed cross product)
                let normal = Vec3 {
                    x: e1.y * e2.z - e1.z * e2.y,
                    y: e1.z * e2.x - e1.x * e2.z,
                    z: e1.x * e2.y - e1.y * e2.x,
                };

                // Simple back-face culling: skip faces whose normal points away.
                if normal.z <= 0.0 {
                    continue;
                }

                // Average depth for painter's algorithm
                let depth = (v0.z + v1.z + v2.z + verts_view[i3].z) / 4.0;

                faces_to_draw.push(FaceToDraw {
                    indices: [i0, i1, i2, i3],
                    depth
                });

            }

            faces_to_draw.sort_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap());

            for face in faces_to_draw {
                let [i0, i1, i2, i3] = face.indices;

                let p0 = projected[i0];
                let p1 = projected[i1];
                let p2 = projected[i2];
                let p3 = projected[i3];

                canvas.fill_triangle(p0, p1, p2, color);
                canvas.fill_triangle(p0, p2, p3, color);

            } 
        }

        if draw_vertices {
            for &(px, py) in &projected {
                canvas.draw_pixel(px, py, color);
            }
        }

        if draw_edges {
            for &(i, j) in self.edges.iter() {
                canvas.draw_line(projected[i], projected[j], color);
            }
        }
    }
}

/// Pinhole camera positioned on the Z axis looking toward +Z.
pub struct Camera {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub width: u32,
    pub height: u32,

}

impl Camera {
    /// Creates a new camera at position `(0, 0, z)` with the given viewport size.
    pub fn new(position: Vec3, yaw: f32, pitch: f32, width: u32, height: u32, ) -> Self {
        Self { position, yaw, pitch, width, height   }
    }

    /// Projects a 3D point into screen space using a simple perspective divide.
    pub fn project(&self, world: Vec3) -> (i32, i32) {
        self.project_view(self.world_to_view(world))
    }

    pub fn project_view(&self, view: Vec3) -> (i32, i32) {
        let depth = view.z;
        if depth <= 0.0 {
            return (-1,-1);
        }
        let projected_x = view.x / depth;
        let projected_y = view.y / depth;

        let screen_x = ((projected_x + 1.0) * 0.5 * self.width as f32) as i32;
        let screen_y = ((1.0 - (projected_y + 1.0) *0.5) * self.height as f32) as i32;

        (screen_x, screen_y) 

    }

    pub fn world_to_view(&self, world: Vec3) -> Vec3 {
        let translated =  Vec3::new(world.x - self.position.x, world.y - self.position.y, world.z - self.position.z);
        return translated.rotate_x(-self.pitch).rotate_y(-self.yaw)
    }
}

/// Mutable view into a RGBA frame buffer for software drawing.
pub struct Canvas<'a> {
    frame: &'a mut [u8],
    width: u32,
    height: u32,
}

impl<'a> Canvas<'a> {
    /// Creates a new canvas that draws into the given frame buffer.
    pub fn new(frame: &'a mut [u8], width: u32, height: u32) -> Self {
        Self {
            frame,
            width,
            height,
        }
    }

    pub fn draw_pixel(&mut self, x: i32, y: i32, color: Color) {
        if x < 0 || y < 0 || x >= self.width as i32 {
            return;
        }
        let idx = (y as u32 * self.width + x as u32) as usize * 4;
        if idx + 3 >= self.frame.len() {
            return;
        }
        self.frame[idx..idx + 4].copy_from_slice(&color.0);
    }

    fn draw_line(&mut self, (x0, y0): (i32, i32), (x1, y1): (i32, i32), color: Color) {
        let mut x0 = x0;
        let mut y0 = y0;
        let dx = (x1 - x0).abs();
        let dy = (y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx - dy;

        loop {
            self.draw_pixel(x0, y0, color);
            if x0 == x1 && y0 == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 > -dy {
                err -= dy;
                x0 += sx;
            }
            if e2 < dx {
                err += dx;
                y0 += sy;
            }
        }
    }

    pub fn fill_triangle(
        &mut self,
        (x0, y0): (i32, i32),
        (x1, y1): (i32, i32),
        (x2, y2): (i32, i32),
        color: Color,
    ) {
        //Edge function
        let edge = |(ax, ay): (i32, i32), (bx, by): (i32, i32), (px, py): (i32, i32)| -> i32 {
            (px - ax) * (by - ay) - (py - ay) * (bx - ax)
        };

        let p0 = (x0, y0);
        let p1 = (x1, y1);
        let p2 = (x2, y2);

        let area = edge(p0, p1, p2) as f32;
        if area == 0.0 {
            return;
        }

        // Bounding box, clamped to screen
        let min_x = x0.min(x1).min(x2).max(0);
        let max_x = x0.max(x1).max(x2).min(self.width as i32 - 1);
        let min_y = y0.min(y1).min(y2).max(0);
        let max_y = y0.max(y1).max(y2).min(self.height as i32 - 1);

        let area_sign = if area > 0.0 { 1.0 } else { -1.0 };

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let w0 = edge(p1, p2, (x, y)) as f32 * area_sign;
                let w1 = edge(p2, p0, (x, y)) as f32 * area_sign;
                let w2 = edge(p0, p1, (x, y)) as f32 * area_sign;

                if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                    self.draw_pixel(x, y, color);
                }
            }
        }
    }

    /// Fills an axis‑aligned rectangle with a solid color.
    pub fn draw_rect(&mut self, x: u32, y: u32, w: u32, h: u32, color: [u8; 4]) {
        for iy in y..y + h {
            for ix in x..x + w {
                let idx = ((iy * self.width + ix) * 4) as usize;
                self.frame[idx..idx + 4].copy_from_slice(&color);
            }
        }
    }
}

/// RGBA color stored as 8‑bit channels.
#[derive(Copy, Clone)]
pub struct Color(pub [u8; 4]);

impl Color {
    /// Creates a fully saturated RGB color from a hue in `[0, 1)`.
    pub fn from_hue(h: f32) -> Self {
        let h = h - h.floor();
        let h6 = h * 6.0;
        let x = 1.0 - (h6.fract() - 0.5).abs() * 2.0; // 1 at peak, 0 at ends
        let x = x.clamp(0.0, 1.0);

        let (r, g, b) = match (h6 / 1.0) as i32 % 6 {
            0 => (1.0, x, 0.0),
            1 => (x, 1.0, 0.0),
            2 => (0.0, 1.0, x),
            3 => (0.0, x, 1.0),
            4 => (x, 0.0, 1.0),
            _ => (1.0, 0.0, x),
        };

        Color([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8, 255])
    }

    /// Extracts the approximate HSV hue in `[0, 1)` from this RGB color.
    pub fn to_hue(&self) -> f32 {
        let r = self.0[0] as f32 / 255.0;
        let g = self.0[1] as f32 / 255.0;
        let b = self.0[2] as f32 / 255.0;

        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let delta = max - min;

        if delta == 0.0 {
            return 0.0;
        }

        let mut h = if max == r {
            ((g - b) / delta).rem_euclid(6.0)
        } else if max == g {
            (b - r) / delta + 2.0
        } else {
            (r - g) / delta + 4.0
        };

        h /= 6.0;
        if h < 0.0 { h + 1.0 } else { h }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn vec3_rotate_x_90_degrees() {
        let v = Vec3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        };
        let r = v.rotate_x(FRAC_PI_2);
        assert!(approx_eq(r.x, 0.0, 1e-5));
        assert!(approx_eq(r.y, 0.0, 1e-5));
        assert!(approx_eq(r.z, 1.0, 1e-5));
    }

    #[test]
    fn vec3_rotate_y_90_degrees() {
        let v = Vec3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        };
        let r = v.rotate_y(FRAC_PI_2);
        assert!(approx_eq(r.x, 0.0, 1e-5));
        assert!(approx_eq(r.y, 0.0, 1e-5));
        assert!(approx_eq(r.z, 1.0, 1e-5));
    }

    #[test]
    fn vec3_rotate_z_90_degrees() {
        let v = Vec3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        };
        let r = v.rotate_z(FRAC_PI_2);
        assert!(approx_eq(r.x, 0.0, 1e-5));
        assert!(approx_eq(r.y, 1.0, 1e-5));
        assert!(approx_eq(r.z, 0.0, 1e-5));
    }

    #[test]
    fn cube_mesh_has_expected_counts_and_vertices() {
        let h = 1.0;
        let cube = Mesh::cube(h);
        assert_eq!(cube.vertices.len(), 8);
        assert_eq!(cube.edges.len(), 12);
        assert_eq!(cube.faces.len(), 6);

        // Check a couple of vertices
        assert!(
            cube.vertices
                .iter()
                .any(|v| v.x == -h && v.y == -h && v.z == -h)
        );
        assert!(
            cube.vertices
                .iter()
                .any(|v| v.x == h && v.y == h && v.z == h)
        );
    }

    #[test]
    fn sphere_mesh_has_expected_counts_and_radius() {
        let radius = 1.5;
        let stacks = 6;
        let slices = 8;
        let sphere = Mesh::sphere(radius, stacks, slices);

        // 2 poles + (stacks - 1) * slices middle vertices
        let expected_vertices = 2 + (stacks - 1) * slices;
        assert_eq!(sphere.vertices.len(), expected_vertices);

        // Basic radius check
        for v in &sphere.vertices {
            let r = (v.x * v.x + v.y * v.y + v.z * v.z).sqrt();
            assert!(approx_eq(r, radius, 1e-3));
        }

        // Edge count: top + middle + bottom
        let expected_edges = slices + (stacks - 2) * (2 * slices) + slices * 2;
        assert_eq!(sphere.edges.len(), expected_edges);
    }

    #[test]
    fn camera_projects_center_point_to_screen_center() {
        let width = 640;
        let height = 480;
        let camera = Camera::new(Vec3::new(0.0,0.0,-3.0), 0.0, 0.0, width, height);
        let v = Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };

        let (sx, sy) = camera.project(v);
        assert_eq!(sx, (width as i32) / 2);
        assert_eq!(sy, (height as i32) / 2);
    }

    #[test]
    fn canvas_draw_pixel_writes_correct_bytes() {
        let width = 4;
        let height = 4;
        let mut buf = vec![0u8; (width * height * 4) as usize];
        let mut canvas = Canvas::new(&mut buf, width, height);
        let color = Color([10, 20, 30, 40]);

        canvas.draw_pixel(1, 2, color);

        let idx = ((2 * width + 1) * 4) as usize;
        assert_eq!(&buf[idx..idx + 4], &color.0);
    }

    #[test]
    fn canvas_draw_line_draws_endpoints() {
        let width = 4;
        let height = 4;
        let mut buf = vec![0u8; (width * height * 4) as usize];
        let mut canvas = Canvas::new(&mut buf, width, height);
        let color = Color([255, 255, 255, 255]);

        canvas.draw_line((0, 0), (3, 0), color);

        let idx_start = ((0 * width + 0) * 4) as usize;
        let idx_end = ((0 * width + 3) * 4) as usize;
        assert_eq!(&buf[idx_start..idx_start + 4], &color.0);
        assert_eq!(&buf[idx_end..idx_end + 4], &color.0);
    }

    #[test]
    fn color_from_hue_basic_primaries() {
        // Red
        let red = Color::from_hue(0.0);
        assert!(red.0[0] > 200 && red.0[1] < 10 && red.0[2] < 10);

        // Green (~1/3)
        let green = Color::from_hue(1.0 / 3.0);
        assert!(green.0[1] > 200);

        // Blue (~2/3)
        let blue = Color::from_hue(2.0 / 3.0);
        assert!(blue.0[2] > 200);
    }

    #[test]
    fn color_to_hue_basic_primaries() {
        let red = Color([255, 0, 0, 255]);
        let h_red = red.to_hue();
        assert!(h_red < 0.1 || h_red > 0.9, "red hue={}", h_red);

        let green = Color([0, 255, 0, 255]);
        let h_green = green.to_hue();
        assert!(approx_eq(h_green, 1.0 / 3.0, 0.05), "green hue={}", h_green);

        let blue = Color([0, 0, 255, 255]);
        let h_blue = blue.to_hue();
        assert!(approx_eq(h_blue, 2.0 / 3.0, 0.05), "blue hue={}", h_blue);
    }
}
