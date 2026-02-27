use pixels::{Pixels, SurfaceTexture};
use std::sync::Arc;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

use gui::{Camera, Canvas, Color, Mesh, Vec3};

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;

struct App {
    window: Option<Arc<Window>>,
    pixels: Option<Pixels<'static>>,
    start_time: Instant,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            pixels: None,
            start_time: Instant::now(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let size = LogicalSize::new(WIDTH as f64, HEIGHT as f64);
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Rust GUI")
                        .with_inner_size(size),
                )
                .expect("Failed to create window"),
        );

        let window_size = window.inner_size();
        let surface_texture =
            SurfaceTexture::new(window_size.width, window_size.height, window.clone());
        let pixels =
            Pixels::new(WIDTH, HEIGHT, surface_texture).expect("Failed to create pixels");

        self.start_time = Instant::now();
        self.window = Some(window);
        self.pixels = Some(pixels);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = &self.window else {
            return;
        };

        if window.id() != window_id {
            return;
        }

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                let Some(pixels) = self.pixels.as_mut() else {
                    return;
                };

                let frame = pixels.frame_mut();
                let elapsed = self.start_time.elapsed().as_secs_f32();

                for pixel in frame.chunks_exact_mut(4) {
                    pixel[0] = 0; //R
                    pixel[1] = 0; //G
                    pixel[2] = 0; //B
                    pixel[3] = 255; //A
                }

                let camera = Camera::new(-3.0, WIDTH, HEIGHT);
                let mut canvas = Canvas::new(frame, WIDTH, HEIGHT);

                let cube1 = Mesh::cube(0.5);
                let cube2 = Mesh::cube(0.5);
                let cube3 = Mesh::cube(0.5);
                let sphere = Mesh::sphere(1.0, 80, 80);

                let angle = elapsed * 0.5;

                let cube1_transform =
                    |v: Vec3| v.rotate_x(angle).rotate_y(-angle).translate_x(-1.5);
                let cube2_transform = |v: Vec3| v.rotate_y(-angle);
                let cube3_transform =
                    |v: Vec3| v.rotate_y(-angle).rotate_x(angle).translate_x(1.5);

                let speed = 0.3;
                let amplitude = 2.0;

                let t = elapsed * speed;

                let tri01 = 1.0 - ((t % 2.0) - 1.0).abs();

                let y_offset = (tri01 * 2.0 - 1.0) * amplitude;

                let sphere_transform =
                    |v: Vec3| v.rotate_y(angle).translate_y(-2.0).translate_x(y_offset);

                let base_hue = elapsed * 0.01;
                let color = Color::from_hue(base_hue);
                let color2 = Color::from_hue(base_hue + 0.5);

                sphere.render(
                    &mut canvas,
                    &camera,
                    sphere_transform,
                    true,
                    true,
                    false,
                    color,
                );
                cube1.render(
                    &mut canvas,
                    &camera,
                    cube1_transform,
                    true,
                    false,
                    false,
                    color2,
                );

                cube2.render(
                    &mut canvas,
                    &camera,
                    cube2_transform,
                    true,
                    true,
                    false,
                    color2,
                );
                cube3.render(
                    &mut canvas,
                    &camera,
                    cube3_transform,
                    false,
                    true,
                    true,
                    color2,
                );

                if pixels.render().is_err() {
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new();

    if let Err(err) = event_loop.run_app(&mut app) {
        eprintln!("Event loop error: {err}");
    }
}
