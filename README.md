# Luma Geospatial Engine

A Python package that serves as the geospatial engine for the Epistem Land Use Mapping for All (Luma) platform.

## File Structure

- **`src/luma_ge/`**: The core Python package for this project. It contains all the backend logic, helper functions, and modules for interacting with Google Earth Engine.
- **`notebooks/`**: Jupyter notebooks used for development, experimentation, and demonstrating the functionality of the core modules.
- **`pyproject.toml`**: The standard Python project configuration file. It defines project metadata and core dependencies for `pip`.

## Getting Started

Follow these instructions to set up and run the project on your local machine.

### 1. Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Git**: A version control system for cloning the repository. [Installation Guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).
- **Python environment manager**: If you do not yet have one installed, we recommend [Miniforge](https://github.com/conda-forge/miniforge); it is lightweight, no-frills compared to Anaconda, and works well for this project. If you already have another Conda-compatible manager, you can continue using it.

To confirm these tools are available in your shell, run:

```powershell
git --version
conda --version
```

**Warning for Windows Users: Do not add Python or Conda to your system PATH.** This causes conflicts and prevents the luma_ge environment from working correctly. For details, see [FAQ- Should I add Anaconda to the Windows PATH?](https://www.anaconda.com/docs/getting-started/working-with-conda/reference/faq#should-i-add-anaconda-to-the-windows-path).

### 2. Set Up the Python Environment

Choose one of the following setup methods based on your needs:

#### Option A: Install from Git

_Recommended for most users - install directly from the repository._

**Direct install (no cloning required):**

```bash
pip install git+https://github.com/epistem-io/EpistemXBackend.git
```

**Or clone and install in editable mode:**

1. Clone the repository:

   ```bash
   git clone https://github.com/epistem-io/EpistemXBackend.git
   cd EpistemXBackend
   ```

2. Install the package using pip:

   ```bash
   pip install -e .
   ```

   This will install `luma_ge` and all its dependencies as specified in `pyproject.toml`.

3. Proceed to [Usage](#4-usage).

#### Option B: Docker Container

_Best for deployment or isolated environments._

1. Build the Docker image:

   ```bash
   docker build -t luma-ge .
   ```

2. Run the container:

   ```bash
   docker run -p 7860:7860 luma-ge
   ```

3. The application will be available at `http://localhost:7860`.

#### Option C: GitHub Codespaces (Cloud-based, No Local Setup)

_Best for quick experimentation without local installation, or when working on different machines._

1. **Create a Codespace** from the repository:

   - Navigate to the [EpistemXBackend repository](https://github.com/epistem-io/EpistemXBackend) on GitHub
   - Click the green **Code** button
   - Select **Codespaces** tab → **Create codespace on update_main**
   - Wait for the environment to initialize (typically 2-3 minutes)

2. **Start working with notebooks**:
   - Open the `notebooks/Module_implementation.ipynb` file in VS Code's notebook editor within the Codespace
   - When prompted, select the Python kernel provided by the Codespace environment
   - The required environment and package setup are already configured in Codespaces
   - Work through the notebook for step-by-step examples and module testing

**Codespaces Tips:**

- Notebooks run directly in VS Code's built-in notebook editor
- Long-running Earth Engine operations may take several minutes; monitor progress in cell outputs
- Use the VS Code terminal (Ctrl+\`) for additional commands if needed
- Codespaces automatically save your work; you can return to the same Codespace later

### 4. Usage

#### Running the Jupyter Notebooks

Before launching, install the `epistemx` package into the active environment so notebooks can import the source modules:

```bash
python -m pip install -e .
```

Launch Jupyter Lab from the project root to explore the project's modules and workflows:

```bash
jupyter lab
```

Start with `Module_implementation.ipynb` in the `notebooks/` directory for a focused guide on individual module testing and development. This notebook provides step-by-step examples for working with the core luma_ge modules.
