<a id="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]



<br />
<div align="center">
  <a href="https://github.com/mei-diao/diao-drawer">
    <img src="https://github.com/user-attachments/assets/1bdb6672-8ece-497c-bc65-06f45f5b68c2" alt="Logo" width="130" height="180">
  </a>

  <h3 align="center">Ultimate DIAO Drawer (UDD)</h3>

  <p align="center">
    <b>D</b>igital <b>I</b>maging <b>A</b>nalysis <b>O</b>peration
    <br />
    A parametric 3D model generator with a Gradio UI. Tweak shape, color,
    surface and hair sliders; get an interactive `.glb` you can rotate in the
    browser.
    <br />
    <br />
    <a href="https://huggingface.co/spaces/meidiao/diao-drawer">
      <img src="https://img.shields.io/badge/%F0%9F%A4%97-Open_Gradio_Demo-blue">
    </a>
    <a href="https://github.com/mei-diao/diao-drawer/issues/new?labels=bug&template=bug-report---.md">
      <img src="https://img.shields.io/badge/Report_Bug-red">
    </a>
    <a href="https://github.com/mei-diao/diao-drawer/issues/new?labels=enhancement&template=feature-request---.md">
      <img src="https://img.shields.io/badge/Request_Feature-green">
    </a>
  </p>
</div>



## Table of Contents

- [About](#about)
- [Built With](#built-with)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [3D version (`diao_3d.py`)](#3d-version-diao_3dpy)
  - [2D version (`diao_matplotlib.py`)](#2d-version-diao_matplotlibpy)
  - [Parameters](#parameters)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)



## About

UDD is a small toy project that generates a parametric 3D mesh from a handful
of sliders (shaft length / radius, head, scrotum, surface wrinkles, hair, …)
and serves it through a Gradio interface. Two entry points are included:

- `diao_3d.py` — interactive **3D** version. Samples a signed-distance field
  and extracts a single watertight body with marching cubes, applies
  per-vertex colors and surface perturbations, optionally adds hair strands
  and bakes a skin pack into textures, and exports `.glb` for the Gradio
  `Model3D` viewer.
- `diao_matplotlib.py` — older **2D** version. Renders a static PNG via
  `matplotlib`'s 3D toolkit. Kept around as a lightweight fallback.

This is an educational / for-fun project. See [Acknowledgments](#acknowledgments).



### Built With

- [Python 3.10+](https://python.org/)
- [Gradio](https://www.gradio.app/) — UI + model serving
- [trimesh](https://trimesh.org/) — mesh construction and `.glb` export
- [NumPy](https://numpy.org/) / [SciPy](https://scipy.org/) — geometry math, gaussian wrinkle filter
- [scikit-image](https://scikit-image.org/) — marching cubes for the SDF body
- [Pillow](https://python-pillow.org/) — encoding the skin-pack textures
- [Matplotlib](https://matplotlib.org/) + [Seaborn](https://seaborn.pydata.org/) — 2D version only



## Getting Started

### Prerequisites

- Python **3.10 or newer** (tested on 3.10–3.12)
- `pip`
- A modern browser (the 3D viewer is WebGL-based)

### Installation

```bash
git clone https://github.com/mei-diao/diao-drawer.git
cd diao-drawer
pip install -r requirements.txt
```

If you hit a `pydantic` schema error when launching Gradio, upgrade pydantic:

```bash
pip install -U "pydantic>=2.13"
```



## Usage

### 3D version (`diao_3d.py`)

```bash
python diao_3d.py
```

Then open the URL Gradio prints (default `http://127.0.0.1:7860`). The UI
exposes a set of grouped controls for shape, color, material, texture,
surface, hair, optional veins, a few one-click presets and a 🎲 **Random**
button. The generated `.glb` is rendered in-page and can be downloaded.

#### Material packs

The **Material** tab swaps the whole look in one click. `Original` keeps the
colors from the Color tab; every other pack replaces them with its own
pattern and sets its own metalness / roughness (and glow, for Neon):

`Gold` · `Chrome` · `Marble` · `Wood` · `Rainbow` · `Candy cane` · `Camo` ·
`Galaxy` · `Neon` · `Chocolate`

Packs are plain functions in `diao_3d.py` (`MATERIAL_PACKS`): a pattern that
colors each vertex, plus the PBR numbers injected into the `.glb`. Adding one
is a few lines.

The **Themes** buttons under the presets are one-click combos of a pack and
a shape that suits it: 🏆 Trophy (gold), 🗿 Statue (marble), 🤖 Robot
(chrome, with raised "cables"), 🌌 Galaxy, 🍭 Candy (candy cane, hooked) and
🌈 Rainbow. Each one is a normal entry in `PRESETS`, built from an existing
preset with `_theme(base, **changes)`; themes always use skin pack `None`.

#### Skin packs

The **Texture** tab swaps the flat vertex colors for baked textures. The body
gets a UV atlas, and every texel is rendered from procedural skin layers —
mottling, veins under the skin, creases (rings around the shaft, rugae on the
scrotum), follicles, freckles and fine grain — into albedo, normal and
roughness maps embedded in the `.glb`:

`None` (flat vertex colors, as before) · `Natural` · `Silky` · `Weathered` ·
`Freckled` · `Goosebumps`

Skin packs take their base tone from the Color tab. With a material pack other
than `Original`, only the skin's relief carries over — painted skin, or a cast.
Packs are dicts of layer strengths in `diao_3d.py` (`SKIN_PACKS`), so adding
one is a single line. The first render of a new shape takes about a second
(UV atlas + bake); changes that keep the shape reuse it.

#### Recipe codes

Every render also produces a **Recipe code** (`UDD1-…`) under the presets —
all the parameters, compressed into one string. Send it to a friend; they
paste it into the same box and click **Load** to get the identical model.
Codes are validated on load: out-of-range numbers are clamped and anything
unknown falls back to the Default preset.

### 2D version (`diao_matplotlib.py`)

```bash
python diao_matplotlib.py
```

Outputs a static `output.png` and shows it in the Gradio panel. If the title
font appears as boxes (`口口`), see the
[CJK font note](#cjk-font-on-the-2d-version) below.

### Parameters

The 3D UI groups parameters into sections:

| Group     | Parameter                  | What it does                                        |
| --------- | -------------------------- | --------------------------------------------------- |
| Shape     | Shaft length / radius      | Length and thickness of the shaft                   |
|           | Head radius                | Glans sphere radius                                 |
|           | Scrotum radius             | Each testicle sphere radius                         |
|           | Curvature                  | Bend the shaft along its length (0 = straight)      |
|           | Corona ridge               | Adds a torus ring under the glans                   |
| Color     | Shaft / Head / Scrotum     | Hex pickers, blended smoothly between regions       |
| Material  | Material pack              | `Original` or one of the packs above                |
| Texture   | Skin pack                  | `None` (flat color) or a baked skin texture pack    |
| Surface   | Wrinkle intensity (×3)     | Per-region vertex perturbation amplitude            |
|           | Wrinkle smoothness         | Gaussian σ smoothing the perturbations              |
|           | Wrinkle seed               | Deterministic seed for reproducible output          |
| Hair      | Density / length           | Number and length of curly strands on the scrotum   |
| Veins     | Vein count / thickness     | Number and thickness of raised ridges on the shaft  |

#### CJK font on the 2D version

`diao_matplotlib.py` titles the plot in Chinese (`<name>的屌`). If the bundled
fonts on your machine don't include a CJK family the title renders as `口口`.
The script tries `PingFang SC` (macOS), `Heiti SC` / `STHeiti` (macOS),
`Microsoft YaHei` / `SimHei` (Windows), `WenQuanYi Zen Hei` /
`Noto Sans CJK SC` (common Linux). Install one of those and re-run.



## Roadmap

- [x] Generate `.glb` 3D model
- [x] Gradio UI for parameter sliders
- [x] Color pickers for each part
- [x] Surface wrinkles (gaussian-filtered vertex displacement)
- [x] Single watertight body (SDF + marching cubes) instead of separate primitives
- [x] Hair on the scrotum (randomly bent strands grown from surface normals)
- [x] Reproducible wrinkle seed
- [x] Shaft curvature
- [x] Corona ridge under the glans
- [x] Surface veins on the shaft
- [x] One-click presets
- [x] Material packs (gold, chrome, marble, wood, rainbow, …)
- [x] 🎲 Random button
- [x] Shareable recipe codes
- [x] One-click theme presets (Trophy, Statue, Robot, Galaxy, Candy, Rainbow)
- [x] Texture-based skin packs (instead of flat vertex color)
- [ ] Real-time WebGL preview without round-tripping `.glb`
- [ ] Natural-language-driven generation

See the [open issues](https://github.com/mei-diao/diao-drawer/issues) for more.



## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Briefly: fork, branch, PR. Please keep
PRs focused — one feature or fix at a time.



## Acknowledgments

- This project is built **for educational / entertainment purposes only.**
- Distribution of this project including its outputs without explicit
  permission is not allowed.
- Consequences brought by individuals using this project are not attributed
  to the developers.



<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/mei-diao/diao-drawer.svg?style=for-the-badge
[contributors-url]: https://github.com/mei-diao/diao-drawer/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/mei-diao/diao-drawer.svg?style=for-the-badge
[forks-url]: https://github.com/mei-diao/diao-drawer/network/members
[stars-shield]: https://img.shields.io/github/stars/mei-diao/diao-drawer.svg?style=for-the-badge
[stars-url]: https://github.com/mei-diao/diao-drawer/stargazers
[issues-shield]: https://img.shields.io/github/issues/mei-diao/diao-drawer.svg?style=for-the-badge
[issues-url]: https://github.com/mei-diao/diao-drawer/issues
