# Dyne: Universal Physics Container & Distributed Runtime

**Deploy any Physics, Anywhere.**

Dyne is a lightweight toolchain that turns physics models—whether **Physics-Informed Neural Networks (PINNs)**, **Neural Operators**, or classical **Numerical Solvers**—into portable WebAssembly (WASM) actors.

It orchestrates these "Physics Containers" across heterogeneous clusters (Browsers, ESP32/IoT, Edge Servers), creating a unified computing mesh where devices synchronize boundary conditions in real-time.

## 🏗️ The 3-Layer Architecture

Dyne abstracts the complexity of distributed physics through three core layers:

1.  **Compiler (The Builder):**
    * Converts Python (PyTorch/JAX) models or Rust code into optimized WASM binaries.
    * Injects **"Boundary Sync"** logic automatically, transforming standalone models into communicable actors.
    * Supports quantization and model distillation for microcontroller targets.

2.  **Runtime (The Orchestrator):**
    * **Edge & Browser Compatible:** Runs on `dyne-runtime-web` or embedded environments (ESP32/WASM32).
    * **Physics-Aware Sync:** Manages time-stepping (`dt`) and exchanges boundary data (energy/mass conservation) between split domains.
    * **Dynamic Partitioning:** (Planned) Re-distributes computational load based on network latency and device capability.

3.  **Visualization (The Observer):**
    * Aggregates fragmented simulation data from distributed nodes into a global field view.
    * Provides real-time monitoring of cluster topology and energy error rates.

## 🧪 Research Context

Dyne serves as the foundational platform for the **Micro-SciML** research initiatives:
* **Theme 1: Physics-Aware Edge Runtime** (Real-time data assimilation on sensors)
* **Theme 2: Physical Cluster Orchestrator** (K8s alternative for physics-based mesh computing)
* **Theme 5: Twinkernel** (OS kernel with embedded digital twin capabilities)

> **Current Status (v0.1):** Proof of Concept demonstrating a Wave Equation solver running across dual WASM instances with boundary synchronization.


## ⚡ Quickstart (v0.2)
In v0.2, we introduced the dyne CLI to automate the build and run process.

Prerequisites
- Rust (latest stable)

- Python 3.12.4

- wasm-pack (Required for building WASM: cargo install wasm-pack)

Installation (Dev Mode)
Clone the repository:

```Bash
git clone https://github.com/gunjou-like/Dyne.git
cd Dyne
Install the Dyne CLI:
```
```Bash
pip install -e .
Run the Demo
Navigate to the example project and use the CLI to build and serve the simulation.
```
```Bash

cd examples/wave_demo
dyne run
```
This command will automatically:

1. Read dyne.toml.

2. Generate optimized Rust code (constants.rs) based on your config.

3. Compile the WASM module using wasm-pack.

4. Start a local server at http://localhost:8000.

## ⚡ Quickstart (v0.1 Demo)

To run the **Dual-WASM Wave Equation Demo**, you need to build the project using Cargo and serve the files locally.

### Prerequisites
* **Rust** (latest stable)
* **Python 3** (for local server)

### Build & Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/gunjou-like/Dyne.git
   cd Dyne
    ```
2. Build WASM modules: Use wasm-pack to compile the Rust code and generate the JavaScript bindings for the web.

    ```bash
    wasm-pack build --target web
    ```
3. Start Local Server: Launch a simple HTTP server to serve the static files and the generated WASM binary.

    ```bash
    python3 -m http.server 8000
    ```
4. Open Demo: Open your browser and navigate to: http://localhost:8000


## 🗺️ Roadmap

We are aiming for a lightweight, distributed runtime for Physics-Informed Machine Learning (SciML) on Edge devices.

- [x] **v0.1.x: Proof of Concept**
    - [x] Demo: Wave equation continuity across 2 WASM modules.
- [x] **v0.2.x: Developer Experience (DX) & Foundation**
    - [x] **CLI Toolchain:** `dyne run` for automated building and serving.
    - [x] **Configuration:** `dyne.toml` support with Parameter Injection.
    - [x] **Core Refactoring:** `DyneEngine` trait for universal solver interface.
- [ ] **v0.3.x: Core Technology**
    - [ ] **Multi-Model Support:** Adding Heat Equation / Fluid solvers.
    - [ ] **PDE-aware Partitioning (Beta):** Automated domain decomposition.
- [ ] **v0.4 - v0.6: Edge & Protocol**
    - [ ] **Edge Runtime:** Support for ESP32 and Linux Embedded.
    - [ ] **Boundary Sync Protocol 1.0:** Robust data exchange specification.
    - [ ] **Observability:** Energy error & latency visualization.
- [ ] **v1.0.0: Stable Release**
    - [ ] Production-ready API.
    - [ ] Comprehensive documentation.
    - [ ] Ready for 3rd-party adoption.

## Directory structure 
```
Dyne/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
│
├── dyne/
│   ├── __init__.py
│   ├── cli.py
│   └── config.py
│
│   ├── compiler/
│   │   ├── __init__.py
│   │   ├── parser.py
│   │   └── partitioner/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       ├── simple_split.py
│   │       └── pde_aware.py
│   │
│   │   └── compressor/
│   │       ├── __init__.py
│   │       ├── base.py
│   │       └── noop.py
│   │
│   ├── runtime/
│   │   ├── __init__.py
│   │   ├── wasm_export.py
│   │   ├── boundary_protocol.py
│   │   └── profiles.py
│   │
│   └── viz/
│       ├── __init__.py
│       ├── server.py
│       └── assets/
│           ├── index.html
│           ├── app.js
│           └── style.css
│
├── wasm/
│   ├── README.md
│   ├── modules/
│   │   └── dummy_module.wasm
│   └── src/
│
├── examples/
│   └── wave_pinn/
│       ├── wave_pinn.onnx
│       ├── run_demo.sh
│       └── notebook.ipynb
│
├── tests/
│   ├── __init__.py
│   ├── test_parser.py
│   ├── test_simple_split.py
│   └── test_integration_demo.py
│
├── docs/
│   ├── index.md
│   ├── architecture.md
│   ├── partitioner.md
│   └── roadmap.md
│
└── .github/
    └── workflows/
        └── ci.yml

```