from setuptools import setup, find_packages

# Define extras here so we can derive "all" programmatically
extras = {
    # Evaluation pipeline (single-backend runner + adapters)
    "eval": [
        # Orchestration / config
        "hydra-core>=1.3.2,<1.4",
        "omegaconf>=2.3,<2.4",

        # Provider adapters use one lightweight async HTTP transport. This
        # deliberately avoids pinning an OpenAI SDK version against SLIME.
        "httpx>=0.27",

        # Vision utilities for PIL <-> PNG data URLs
        "Pillow>=10.0.0,<12",
    ],

    # You can add more extras later, e.g., "dev": [...], "render": [...]
}

# "all" aggregates every extra listed above (unique + sorted)
extras["all"] = sorted({pkg for group in extras.values() for pkg in group})

setup(
    name="view_suite",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        # Core runtime deps for your package (keep your originals)
        "numpy",
        "requests",
        "open3d==0.19.0",
        "uvicorn<=0.40.0",
        "fastapi",
        "websockets==15.0.1",
        "fire",
        "huggingface-hub",
        "opencv-python-headless>=4.8",
        "tenacity",
        "gym-sokoban",
        "python-multipart",
        "ai2thor",
        "ray"
    ],
    extras_require=extras,
)
