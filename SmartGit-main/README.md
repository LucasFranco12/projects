# SmartGit: A Smart Version Control Tool

SmartGit is a minimal Git-like version control system with integrated AI features designed to help you manage your code, track changes, and even guide you through project goals.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Compiling from Source (Windows)](#compiling-from-source-windows)
  - [Compiling from Source (Linux/macOS)](#compiling-from-source-linuxmacos)
- [Getting Started](#getting-started)
- [Commands](#commands)
  - [`smartgit init`](#smartgit-init)
  - [`smartgit add <file>`](#smartgit-add-file)
  - [`smartgit commit --message <message>`](#smartgit-commit---message-message)
  - [`smartgit status`](#smartgit-status)
  - [`smartgit log`](#smartgit-log)
  - [`smartgit restore <file>`](#smartgit-restore-file)
  - [`smartgit diff [--from <commit_idx>] [--to <commit_idx>]`](#smartgit-diff---from-commit_idx---to-commit_idx)
  - [`smartgit history <file>`](#smartgit-history-file)
  - [`smartgit watch --on|--off|--time <minutes>`](#smartgit-watch---on---off---time-minutes)
  - [`smartgit goal`](#smartgit-goal)
    - [`smartgit goal set <description>`](#smartgit-goal-set-description)
    - [`smartgit goal show`](#smartgit-goal-show)
    - [`smartgit goal tasks`](#smartgit-goal-tasks)
    - [`smartgit goal task list`](#smartgit-goal-task-list)
    - [`smartgit goal task set <number> <description>`](#smartgit-goal-task-set-number-description)
    - [`smartgit goal task prompt <number> <prompt>`](#smartgit-goal-task-prompt-number-prompt)
  - [`smartgit progress`](#smartgit-progress)
  - [`smartgit signup --email <email> --password <password>`](#smartgit-signup---email-email---password-password)
  - [`smartgit login --email <email> --password <password>`](#smartgit-login---email-email---password-password)
  - [`smartgit sync`](#smartgit-sync)
  - [`smartgit pull [--project <project_name>]`](#smartgit-pull---project-project_name)
  - [`smartgit list-projects`](#smartgit-list-projects)
  - [`smartgit switch-project --name <project_name>`](#smartgit-switch-project---name-project_name)
  - [`smartgit current-project`](#smartgit-current-project)
  - [`smartgit revert <commit_idx>`](#smartgit-revert-commit_idx)
  - [`smartgit reset <commit_idx>`](#smartgit-reset-commit_idx)
  - [`smartgit export [--output <path>]`](#smartgit-export---output-path)
- [Development & Contributing](#development--contributing)
  - [Firebase Configuration](#firebase-configuration)
  - [GroqCloud API Configuration](#groqcloud-api-configuration)

## Features
*   **Basic VCS Operations**: `init`, `add`, `commit`, `status`, `log`, `restore`.
*   **Advanced VCS Features**: `diff`, `history`, `revert`, `reset`, `export`.
*   **File Watcher**: Auto-commit changes after inactivity.
*   **AI-Powered Goal Management**: Set project goals, generate tasks, and track progress using LLM integration.
*   **Cloud Sync**: Synchronize commits, progress, and tasks with Firestore.
*   **Multi-User & Multi-Project**: Securely manage your projects and data in the cloud with user authentication.

## Installation

### Prerequisites
*   [Rust](https://www.rust-lang.org/tools/install) (latest stable version recommended)
*   For Windows cross-compilation: [MSYS2](https://www.msys2.org/) with `mingw-w64-x86_64-gcc` installed.

### Compiling from Source (Windows)

1.  **Install Rust:**
    ```bash
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    ```
2.  **Add Linux Cross-Compilation Target (if compiling for Linux from Windows):**
    ```bash
    rustup target add x86_64-unknown-linux-gnu
    ```
3.  **Install MSYS2 and MinGW GCC (for C/C++ dependencies during cross-compilation):**
    *   Download and install [MSYS2](https://www.msys2.org/).
    *   Open "MSYS2 MinGW x64" from the Start menu and run:
        ```bash
        pacman -Syu
        pacman -S mingw-w64-x86_64-gcc
        ```
    *   Add `C:\msys64\mingw64\bin` to your system's `Path` environment variable. Restart your terminal after this.
4.  **Navigate to the project root directory** (where `Cargo.toml` is).
5.  **Build the project:**
    *   For Windows native binary:
        ```bash
        cargo build --release
        ```
    *   For Linux binary (cross-compile from Windows):
        ```bash
        cargo build --release --target x86_64-unknown-linux-gnu
        ```
    The executable will be in `target/release/smartgit.exe` (Windows) or `target/x86_64-unknown-linux-gnu/release/smartgit` (Linux).

### Compiling from Source (Linux/macOS)

1.  **Install Rust:**
    ```bash
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    ```
2.  **Navigate to the project root directory** (where `Cargo.toml` is).\
3.  **Build the project:**
    ```bash
    cargo build --release
    ```
    The executable will be in `target/release/smartgit`.

## Getting Started

1.  **Initialize a new repository:**
    ```bash
    smartgit init
    ```
    This creates a `.smartgit` directory and a `.smartgitignore` file.

2.  **Create an account or login:**
    ```bash
    smartgit signup --email your_email@example.com --password your_password
    ```
    Or if you already have an account:
    ```bash
    smartgit login --email your_email@example.com --password your_password
    ```

3.  **Start using SmartGit for version control!**

## Commands

### `smartgit init`
Initializes a new SmartGit repository in the current directory. This creates a `.smartgit` directory to store all repository data (objects, commits, index) and a `.smartgitignore` file with default patterns.

### `smartgit add <file>`
Adds a specified file to the SmartGit staging area (index). This tracks the file's content and prepares it for the next commit.
*   **Example:** `smartgit add src/main.rs`

### `smartgit commit --message <message>`
Records the staged changes as a new commit in the repository's history. Each commit includes a message, timestamp, list of files, and system metadata (hostname, OS, username).
*   **Example:** `smartgit commit --message "Implement initial add and commit commands"`

### `smartgit status`
Displays the current state of the working directory and staging area. It shows:
*   Files staged for commit.
*   Files modified since the last commit but not staged.
*   Files deleted from the working directory but still in the index.
*   Untracked files.

### `smartgit log`
Shows the commit history for the current repository, listing commits in reverse chronological order. Each entry includes the commit index, timestamp, message, and the files included in that commit.

### `smartgit restore <file>`
Restores a specified file to its state from the most recent commit in which it appeared.
*   **Example:** `smartgit restore config.rs`

### `smartgit diff [--from <commit_idx>] [--to <commit_idx>]`
Shows line-by-line changes between commits or between the working directory and a commit.
*   If no arguments are provided, it compares the working directory with the HEAD commit.
*   `--from <commit_idx>`: Specifies the starting commit for the comparison (1-based index).
*   `--to <commit_idx>`: Specifies the ending commit for the comparison (1-based index).
*   **Examples:**
    *   `smartgit diff` (compares working directory with HEAD)
    *   `smartgit diff --from 1 --to 3` (compares commit 1 with commit 3)
    *   `smartgit diff --from 5` (compares commit 5 with HEAD)
    *   `smartgit diff --to 2` (compares working directory with commit 2)

### `smartgit history <file>`
Displays the commit history for a specific file. It lists all commits that modified the file, including their commit index, timestamp, message, and the file's hash in that commit.
*   **Example:** `smartgit history src/cli.rs`

### `smartgit watch --on|--off|--time <minutes>`
Manages the automatic file watcher that auto-commits changes after a period of inactivity.
*   `--on`: Starts the watcher in the background.
*   `--off`: Stops the currently running watcher.
*   `--time <minutes>`: Sets the inactivity timeout for auto-commits in minutes (default is 10 minutes).
*   **Examples:**
    *   `smartgit watch --on`
    *   `smartgit watch --time 5`
    *   `smartgit watch --off`

### `smartgit goal`
A subcommand for managing project goals and tasks with AI assistance.

#### `smartgit goal set <description>`
Sets a new overall goal for your project. This description will be used by the AI to generate tasks.
*   **Example:** `smartgit goal set "Build a fully functional web server in Rust"`

#### `smartgit goal show`
Displays the currently set project goal.

#### `smartgit goal tasks`
Generates a list of checkpoint tasks based on the current project goal using an LLM. If tasks already exist, it will list them instead.
*   **Example:** `smartgit goal tasks`

#### `smartgit goal task list`
Lists all generated tasks, showing their completion status, difficulty, and XP.

#### `smartgit goal task set <number> <description>`
Sets or updates the description for a specific task by its number. The AI will also rate its difficulty.
*   **Example:** `smartgit goal task set 1 "Implement HTTP request parsing"`

#### `smartgit goal task prompt <number> <prompt>`
Allows you to prompt the AI about a specific task with a custom question.
*   **Example:** `smartgit goal task prompt 2 "What Rust crates are suitable for handling routing?"`

### `smartgit progress`
Displays your current development progress, including total XP gained and a progress bar for completed tasks.

### `smartgit signup --email <email> --password <password>`
Creates a new SmartGit account. Your project data will be isolated and associated with this account in Firestore.
*   **Example:** `smartgit signup --email newuser@example.com --password mysecretpassword`

### `smartgit login --email <email> --password <password>`
Logs in to an existing SmartGit account. This fetches your authentication tokens and sets up the current user session.
*   **Example:** `smartgit login --email existing@example.com --password mypassword`

### `smartgit sync`
Manually synchronizes your local progress data with Firestore. Commits are automatically synced after each `commit` operation.

### `smartgit pull [--project <project_name>]`
Pulls the latest project data (commits, objects, progress, tasks) from Firestore for the current or specified project. This ensures your local repository reflects the most recent cloud state.
*   If no project name is specified, it pulls for the currently active project.
*   **Example:** `smartgit pull` or `smartgit pull --project my_web_server`

### `smartgit list-projects`
Lists all project names associated with your logged-in user account in Firestore.

### `smartgit switch-project --name <project_name>`
Switches the active project for the current session. Subsequent commands will operate on this project.
*   **Example:** `smartgit switch-project --name my_new_feature`

### `smartgit current-project`
Displays the name of the currently active project.

### `smartgit revert <commit_idx>`
Creates a new commit that undoes the changes introduced by a specified commit. The original commit remains in the history.
*   **Example:** `smartgit revert 3` (undoes changes from the 3rd commit)

### `smartgit reset <commit_idx>`
Performs a hard reset of the working directory and local history to a specified commit. All commits made after the target commit will be discarded from local history and the working directory will be updated to match the target commit's state.
*   **Example:** `smartgit reset 2` (resets to the 2nd commit, discarding commits 3 and later)

### `smartgit export [--output <path>]`
Bundles the entire SmartGit repository into a zip archive. This includes the `.smartgit` folder (containing commits, objects, etc.), the `.smartgitignore` file, and the most recent version of each tracked file.
*   If no `--output` path is provided, the default is `smartgit-backup.zip`.
*   **Example:** `smartgit export` or `smartgit export --output my_project_backup.zip`

## Development & Contributing

### Firebase Configuration
This project uses Firebase Firestore for cloud synchronization and authentication.
The Firebase API Key and Project ID are **hardcoded** in `src/main.rs` for convenience.
*   `FIREBASE_API_KEY`: `AIzaSyDHecGyk7UK5JCsgoLE1e8MUmjAbb2Fn0I`
*   `FIREBASE_PROJECT_ID`: `smartgit-abe0d`
If you wish to use your own Firebase project, you will need to replace these values in `src/main.rs`.

### GroqCloud API Configuration
The AI features (goal setting, task generation, task completion checking) rely on the GroqCloud API.
You need to set the following environment variables for the AI features to work:
*   `GROQ_API_KEY`: Your API key from GroqCloud.
*   `GROQ_MODEL`: (Optional) The model to use (e.g., `llama-3.3-70b-versatile`). Defaults to `llama-3.3-70b-versatile` if not set.

**Example for setting environment variables (Linux/macOS):**
```bash
export GROQ_API_KEY="your_groq_api_key_here"
export GROQ_MODEL="llama-3.3-70b-versatile" # Optional
```
**Example for setting environment variables (Windows PowerShell):**
```powershell
$env:GROQ_API_KEY="your_groq_api_key_here"
$env:GROQ_MODEL="llama-3.3-70b-versatile" # Optional
```
```