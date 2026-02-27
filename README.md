## Rust GUI Learning Project

This repository is a **personal learning project** for experimenting with basic GUI and rendering concepts in Rust.  
It currently uses the `winit` and `pixels` crates to open a window and draw animated 2D content, and is in a very early, exploratory stage.

### Status

- **Project stage**: Extremely early / experimental  
- **Audience**: Personal learning, not yet intended for general reuse  
- **Stability**: APIs and structure are likely to change frequently

### Prerequisites

- **Rust** (recommended: latest stable from `rustup`)  
- A **desktop OS with a GUI** (Windows, macOS, or a Linux distribution with a graphical environment)

### Getting Started

Clone the repository and run the project with Cargo:

```bash
git clone <your-repo-url>
cd gui
cargo run
```

This should open a window titled **"Rust GUI"** and start the current experiment/animation.

### Project Structure (High-Level)

- `Cargo.toml` – crate configuration and dependencies (`winit`, `pixels`, etc.)  
- `src/main.rs` – sets up the window, event loop, and rendering surface  
- `src/lib.rs` – core types such as camera, mesh, canvas, and math utilities backing the rendering

As this is a learning project, the structure and modules may be refactored aggressively as new ideas are tried.

### Goals

- Learn the basics of **GUI and window management** in Rust  
- Experiment with **rendering pixels to a window** and simple 2D/3D scenes  
- Explore patterns for organizing rendering, camera, and geometry code in Rust

### License

This project is licensed under the **Mozilla Public License 2.0 (MPL-2.0)**.  
See the `LICENSE` file for the full text.


