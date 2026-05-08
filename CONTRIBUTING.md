# Contributing to DIAO Drawer

Thanks for your interest in DIAO Drawer (UDD — Ultimate DIAO Drawer). This is
a small toy project: parametric 3D mesh generation driven by a Gradio UI.
Contributions of any size are welcome.

## Table of Contents

- [How to Contribute](#how-to-contribute)
- [Getting Started](#getting-started)
- [Submitting Changes](#submitting-changes)
- [Issues](#issues)
- [Pull Requests](#pull-requests)

## How to Contribute

There are several ways to contribute:

- Report bugs or rendering glitches
- Suggest new shape / surface / preset parameters
- Improve documentation or examples
- Submit pull requests for fixes or features
- Review pull requests from others

## Getting Started

### 1. Fork the repository

Fork the repo to your own GitHub account.

### 2. Clone your fork

```bash
git clone https://github.com/<your-username>/diao-drawer
cd diao-drawer
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run locally

```bash
python diao_3d.py        # 3D version (Gradio + trimesh, exports .glb)
python diao_matplotlib.py # 2D version (matplotlib PNG)
```

Then open the URL Gradio prints (default `http://127.0.0.1:7860`).

### 5. Create a feature branch

```bash
git checkout -b feat/your-feature-name
```

## Submitting Changes

### 1. Commit

```bash
git add <files>
git commit -m "short description of the change"
```

Prefer small, focused commits. Avoid `git add .` if you have unrelated local
artifacts (`*.glb`, `output.png`, etc.) — those are gitignored but it's still
easy to slip something in.

### 2. Push

```bash
git push origin feat/your-feature-name
```

### 3. Open a Pull Request

Open a PR against `main` (or the current default branch). Please include:

- A short description of what the change does and why
- Before/after screenshots if you changed rendering output
- Steps to reproduce, if you're fixing a bug

## Issues

When filing an issue, include:

- OS and Python version
- Versions of the key deps (`gradio`, `trimesh`, `pydantic`, `numpy`)
- Steps to reproduce
- What you expected vs. what happened
- Screenshots / logs if relevant

## Pull Requests

- Match existing code style (it's plain Python — no formatter enforced).
- Keep PRs focused on one issue or feature.
- If you add a new geometry feature, default it to off / minimal so existing
  outputs don't change unexpectedly.

---

Thanks for contributing to **DIAO Drawer**.
