# Dyne
**Lightweight PINN Inference Toolchain for IoT/Edge/WASM**

Dyne enables Physics-Informed Neural Networks (PINNs) inference on resource-constrained devices (ESP32, browser WASM, edge GPUs) through a 3-layer toolchain:

1. Compiler/Optimizer: ONNX model partitioning with PDE-aware logic – splits models by physical subdomains while preserving boundary continuity.
2. Runtime: WASM inference engine + Boundary Sync Protocol for multi-device coordination.
3. Visualization: Real-time mesh rendering of distributed simulations.

MVP splits 1 ONNX into 2 WASM modules running on dual browser canvases. Future: compressor plugins, on-device fine-tuning.

Powers Theme1 Physics-Aware Edge Runtime , Theme2 Physical Cluster Orchestrator (WASM on ESP32 clusters), and Theme5 Twinkernel. Demonstrates "safe AI control" – sensor noise filtered by physics laws (energy conservation).

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
- [ ] **v0.2.x: Developer Experience (DX)**
    - [ ] CLI for scaffolding and building.
    - [ ] Configuration file support (`.yaml` / `.toml`).
    - [ ] Stable browser demo environment.
- [ ] **v0.3.x: Core Technology**
    - [ ] **PDE-aware Partitioning (Beta):** Automated domain decomposition.
    - [ ] Simple Edge integration testing.
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
dyne-pinn/
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