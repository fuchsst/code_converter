# C++ to Godot Conversion Assistant

This project utilizes a multi-agent system powered by CrewAI and Large Language Models (LLMs) like Google Gemini to assist in the complex process of converting C++ codebases (specifically game logic) to Godot Engine 4.x projects, targeting GDScript (or potentially C# with configuration changes).

The core philosophy emphasizes minimizing LLM API calls through deterministic preprocessing, direct context provision, and a structured, iterative workflow.

## Core Concepts

*   **CrewAI Framework:** Orchestrates specialized AI agents for different tasks.
*   **5-Step Workflow:** The conversion process is broken down into distinct, manageable steps:
    1.  **Dependency Analysis:** Deterministically analyzes C++ `#include` directives using Clang.
    2.  **Work Package Identification:** LLM proposes logical groups of C++ files for conversion.
    3.  **Godot Structure Definition:** LLM proposes a target Godot scene/node/script structure for a work package.
    4.  **Mapping Definition:** LLM defines a high-level strategy and a detailed task list for converting a package to the proposed structure.
    5.  **Code Processing:** Executes the task list, using an LLM agent to generate/modify Godot code, applying changes, and validating syntax.
*   **API Call Minimization:** Achieved via deterministic Step 1, single Crew executions per logical step (2-4), and executor-driven iteration in Step 5.
*   **Direct Context (No RAG):** Relies on providing relevant file content directly to LLMs, managed by the `ContextManager`.
*   **Iterative Refinement:** Includes a feedback loop (Step 5 -> Step 4) for correcting mapping issues based on code generation/validation failures.

## Prerequisites

*   **Python:** Version 3.10 or higher recommended.
*   **Pip:** Python package installer.
*   **Dependencies:** Install required packages:
    ```bash
    pip install -r requirements.txt
    ```
*   **Clang:** Libclang library and Python bindings (`pip install clang`). Ensure libclang is accessible in your system's PATH or configure its path if needed (see `src/utils/dependency_analyzer.py`).
*   **libclang.dll** e.g. on Windows install LLVM (`choco install LLVM`)
*   **C++ Project:**
    *   The source code of the C++ project you want to convert.
    *   A `compile_commands.json` file in the root of the C++ project directory. This is crucial for accurate dependency analysis by Clang. (CMake can generate this with `CMAKE_EXPORT_COMPILE_COMMANDS=ON`).
*   **Godot Engine:** Godot 4.x executable. Ensure it's in your system's PATH or set the `GODOT_EXECUTABLE_PATH` environment variable.
*   **LLM API Key:** An API key for the desired LLM provider (e.g., Google Gemini). Set the `GEMINI_API_KEY` environment variable.

## Configuration

*   **Environment Variables:** Create a `.env` file in the project root to store sensitive information like API keys:
    ```dotenv
    GEMINI_API_KEY=YOUR_API_KEY_HERE
    # Optional: Override default paths or models
    # CPP_PROJECT_DIR=path/to/your/cpp/project
    # GODOT_PROJECT_DIR=path/to/your/godot/project
    # GODOT_EXECUTABLE_PATH=path/to/godot.exe
    # ANALYZER_MODEL=google/gemini-1.5-flash-latest
    # MAPPER_MODEL=google/gemini-1.5-pro-latest
    # GENERATOR_EDITOR_MODEL=google/gemini-1.5-flash-latest
    ```
*   **`src/config.py`:** Contains default model names, path configurations (which can be overridden by `.env` or CLI args), token limits, and other settings.

## Usage (CLI)

The primary interface is through `src/main.py` using `fire`.

**General Syntax:**

```bash
python src/main.py <command> [options]
```

**Required Arguments for most commands:**

*   `--cpp-dir path/to/cpp/project`: Specifies the C++ project root.
*   `--godot-dir path/to/godot/project`: Specifies the target Godot project directory. This directory will be used for context (reading existing files) and as the output directory for generated/modified files.

**Key Commands:**

*   **`analyze-deps`**: (Step 1) Runs C++ include dependency analysis.
    ```bash
    python src/main.py analyze-deps --cpp-dir <path>
    ```
*   **`identify-packages`**: (Step 2) Identifies work packages from the dependency graph. Requires Step 1 to be complete.
    ```bash
    python src/main.py identify-packages --cpp-dir <path>
    ```
*   **`define-structure`**: (Step 3) Proposes Godot structure for packages. Requires Step 2 to be complete.
    *   Process all eligible packages:
        ```bash
        python src/main.py define-structure --cpp-dir <path> --godot-dir <path>
        ```
    *   Process specific package(s):
        ```bash
        python src/main.py define-structure --package-id <ID1> [--package-id <ID2> ...] --cpp-dir <path> --godot-dir <path>
        ```
*   **`define-mapping`**: (Step 4) Defines C++ to Godot mapping for packages. Requires Step 3 to be complete for the package.
    *   Process all eligible packages:
        ```bash
        python src/main.py define-mapping --cpp-dir <path> --godot-dir <path>
        ```
    *   Process specific package(s):
        ```bash
        python src/main.py define-mapping --package-id <ID1> [--package-id <ID2> ...] --cpp-dir <path> --godot-dir <path>
        ```
*   **`process-code`**: (Step 5) Generates/modifies Godot code based on mapping. Requires Step 4 to be complete for the package.
    *   Process all eligible packages:
        ```bash
        python src/main.py process-code --cpp-dir <path> --godot-dir <path>
        ```
    *   Process specific package(s):
        ```bash
        python src/main.py process-code --package-id <ID1> [--package-id <ID2> ...] --cpp-dir <path> --godot-dir <path>
        ```
*   **`run-all`**: Runs the full pipeline sequentially (Steps 1-5).
    ```bash
    python src/main.py run-all --cpp-dir <path> --godot-dir <path>
    ```
*   **`resume`**: Attempts to resume the pipeline from the last saved state (useful after interruptions or failures). Handles remapping loops if necessary.
    ```bash
    python src/main.py resume --cpp-dir <path> --godot-dir <path>
    ```

**Optional Arguments:**

*   `--analysis-dir <path>`: Specify a different directory for analysis outputs (dependency graph, state file, intermediate artifacts). Defaults to `./analysis_output`.
*   `--target-language <lang>`: Specify target language (e.g., `GDScript`, `CSharp`). Defaults to `GDScript`.

## Workflow & State

*   The tool operates iteratively. You can run steps individually or use `run-all`.
*   The workflow state (package status, artifacts) is saved in `orchestrator_state.json` within the analysis directory.
*   The `resume` command uses this state file to continue where it left off.
*   Intermediate artifacts (structure proposals, mapping strategies, task lists, task results) are saved in the analysis directory.

## Logging

*   Logs are written to `logs/conversion.log`.
*   Log level can be configured in `src/logger_setup.py`.
