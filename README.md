# Dyne
**Lightweight PINN Inference Toolchain for IoT/Edge/WASM**

Dyne enables Physics-Informed Neural Networks (PINNs) inference on resource-constrained devices (ESP32, browser WASM, edge GPUs) through a 3-layer toolchain:

1. Compiler/Optimizer: ONNX model partitioning with PDE-aware logic – splits models by physical subdomains while preserving boundary continuity.
2. Runtime: WASM inference engine + Boundary Sync Protocol for multi-device coordination.
3. Visualization: Real-time mesh rendering of distributed simulations.

MVP splits 1 ONNX into 2 WASM modules running on dual browser canvases. Future: compressor plugins, on-device fine-tuning.

Powers Theme1 Physics-Aware Edge Runtime , Theme2 Physical Cluster Orchestrator (WASM on ESP32 clusters), and Theme5 Twinkernel. Demonstrates "safe AI control" – sensor noise filtered by physics laws (energy conservation).

# directory structure 
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