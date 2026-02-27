use pixels::{Pixels, SurfaceTexture};
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use std::time::Instant;

use gui::{Camera, Canvas, Color, Mesh, Vec3};

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;
fn main() {
    let event_loop = EventLoop::new();

    let window = WindowBuilder::new()
        .with_title("Rust GUI")
        .with_inner_size(LogicalSize::new(WIDTH, HEIGHT))
        .build(&event_loop)
        .unwrap();

    let surface_texture = SurfaceTexture::new(WIDTH, HEIGHT, &window);

    let mut pixels = Pixels::new(WIDTH, HEIGHT, surface_texture).expect("Failed to create pixels");

    let start_time = Instant::now();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                _ => {}
            },

            Event::RedrawRequested(_) => {
                // update(&mut app);
                // draw(pixels.frame_mut(), &app);

                let frame = pixels.frame_mut();
                let elapsed = start_time.elapsed().as_secs_f32();

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

                let cube1_transform = |v: Vec3| v.rotate_x(angle).rotate_y(-angle).translate_x(-1.5);
                let cube2_transform = |v: Vec3| v.rotate_y(-angle);
                let cube3_transform = |v: Vec3| v.rotate_y(-angle).rotate_x(angle).translate_x(1.5);

                let speed = 0.3;
                let amplitude = 2.0;

                let t = elapsed * speed;

                let tri01 = 1.0 - ((t % 2.0) - 1.0).abs();

                let y_offset = (tri01 * 2.0 - 1.0) * amplitude;

                let sphere_transform = |v: Vec3| v.rotate_y(angle).translate_y(-2.0).translate_x(y_offset);

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
                    *control_flow = ControlFlow::Exit;
                }
            }

            Event::MainEventsCleared => {
                window.request_redraw();
            }

            _ => {}
        }
    });
}
