mod auth;
mod config;
mod sync;
mod cli;

use base64;
use base64::engine::general_purpose::URL_SAFE;
use base64::Engine;
use clap::{Parser, Subcommand};
use std::fs::{self};
use std::io::{Read};
use std::path::Path;
use sha1::{Sha1, Digest};
use serde::{Serialize, Deserialize};
use std::io::Write;
use reqwest;
use std::env;
use serde_json;
use similar;
use similar::{TextDiff, ChangeTag};
use hostname;
use whoami;
use crate::cli::{success, warning, error, info, create_progress_bar, create_spinner};
use crate::sync::FirestoreSync;
use sysinfo::{System, SystemExt, ProcessExt};

/// SmartGit: A minimal Git-like tool with AI integration
#[derive(Parser)]
#[command(name = "smartgit")]
#[command(about = "A smart git-like version control tool", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}


#[derive(Serialize, Deserialize, Clone)]
struct IndexEntry {
    filename: String,
    hash: String,
}

#[derive(Serialize, Deserialize, Clone)]
struct Commit {
    message: String,
    timestamp: String,
    files: Vec<IndexEntry>,
    hostname: String,
    os: String,
    username: String,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize a new SmartGit repository
    Init,

    /// Add a file to the SmartGit staging area
    Add {
        /// File to add
        file: String,
    },

    Commit {
        /// Commit message
        #[arg(short, long)]
        message: String,
    },
    Status,
    Log,
    Restore {
    /// File to restore
    file: String,
},
    /// Show line-by-line changes between commits or working directory
    Diff {
        /// First commit index (optional, defaults to working directory)
        #[arg(long)]
        from: Option<usize>,
        /// Second commit index (optional, defaults to HEAD)
        #[arg(long)]
        to: Option<usize>,
    },
    /// Show commit history for a specific file
    History {
        /// File to show history for
        file: String,
    },
/// Auto-commit files when changes stop
Watch {
    /// Turn on the watcher
    #[arg(long)]
    on: bool,

    /// Turn off the watcher
    #[arg(long)]
    off: bool,

    /// Set inactivity timeout (in minutes)
    #[arg(long)]
    time: Option<u64>,
},
Goal {
    #[command(subcommand)]
    cmd: GoalCommand,
},
InternalWatch, 
Progress,
/// Create a new account
Signup {
    #[arg(long)] email: String,
    #[arg(long)] password: String,
},
/// Login to your account
Login {
    #[arg(long)] email: String,
    #[arg(long)] password: String,
},
Sync,
Pull {
        #[arg(long)] project: Option<String>,
    },
    /// List all projects for the logged-in user
    ListProjects,
    /// Switch to a different project
    SwitchProject {
        /// Name of the project to switch to
        #[arg(long)]
        name: String,
    },
    /// Show the current project name
    CurrentProject,
    /// Revert changes from a specific commit by creating a new commit
    Revert {
        /// Commit index to revert (1-based)
        commit: usize,
    },
    /// Reset working directory and history to a specific commit
    Reset {
        /// Commit index to reset to (1-based)
        commit: usize,
    },
    /// Export repository to a zip archive
    Export {
        /// Output file path (default: smartgit-backup.zip)
        #[arg(short, long)]
        output: Option<String>,
    },
}

#[derive(Subcommand)]
enum GoalCommand {
    /// Set a new project goal description
    Set { description: Vec<String> },  // Vec<String> to capture multi-word
    /// Show the current project goal
    Show,
    
    /// Generate a list of checkpoint tasks for the goal
    Tasks,
    /// Manage/checkpoint individual tasks
    Task {
        #[command(subcommand)]
        cmd: TaskSubCommand,
    },
}

#[derive(Subcommand)]
enum TaskSubCommand {
    /// Show all tasks
    List,
    /// Set or change a specific task by number
    Set {
        number: usize,
        description: Vec<String>,
    },
    /// Prompt about a specific task with a custom prompt
    Prompt {
        number: usize,
        prompt: Vec<String>,
    },
}

#[derive(Serialize, Deserialize, Clone)]
struct Task {
    id: usize,
    description: String,
    completed: bool,
    difficulty: u32,
    xp: u32,
    completion_reason: Option<String>,
}

#[derive(Serialize, Deserialize, Default)]
struct ProjectTasks {
    tasks: Vec<Task>,
}

impl ProjectTasks {
    fn load() -> Self {
        let path = ".smartgit/tasks.json";
        if let Ok(data) = fs::read_to_string(path) {
            serde_json::from_str(&data).unwrap_or_default()
        } else {
            ProjectTasks::default()
        }
    }
    fn save(&self) {
        let _ = fs::write(
            ".smartgit/tasks.json",
            serde_json::to_string_pretty(self).unwrap(),
        );
    }
}

#[derive(Serialize, Deserialize)]
struct ProjectGoal {
    description: String,
}

impl ProjectGoal {
    fn load() -> Option<Self> {
        let path = ".smartgit/goal.json";
        if let Ok(data) = fs::read_to_string(path) {
            serde_json::from_str(&data).ok()
        } else {
            None
        }
    }
    fn save(&self) {
        let _ = fs::write(
            ".smartgit/goal.json",
            serde_json::to_string_pretty(self).unwrap(),
        );
    }
}

#[derive(Serialize, Deserialize, Default)]
struct ProgressData {
    xp: u32,
    completed_tasks: u32,
    total_tasks: u32,
}

impl ProgressData {
    fn from_tasks(tasks: &ProjectTasks) -> Self {
        let xp: u32 = tasks.tasks.iter().map(|t| t.xp).sum();
        let completed_tasks = tasks.tasks.iter().filter(|t| t.completed).count() as u32;
        let total_tasks = tasks.tasks.len() as u32;
        ProgressData { xp, completed_tasks, total_tasks }
    }
    fn save(&self) {
        let _ = std::fs::write(
            ".smartgit/progress.json",
            serde_json::to_string_pretty(self).unwrap(),
        );
    }
}

fn print_progress_bar(completed: u32, total: u32) {
    let percent = if total == 0 { 1.0 } else { completed as f32 / total as f32 };
    let bar_len = 30;
    let filled = (percent * bar_len as f32).round() as usize;
    let bar = format!("[{}{}]", "#".repeat(filled), "-".repeat(bar_len - filled));
    println!("Tasks completed: {}/{}", completed, total);
    println!("{} {:.0}%", bar, percent * 100.0);
    if total > 0 && completed == total {
        println!("🎉 All tasks complete! 🎉");
    }
}

fn run_internal_watcher() {
    use std::fs;
    use std::path::Path;
    use std::sync::{mpsc::channel, Arc, Mutex};
    use std::time::{Duration, Instant};
    use std::thread;
    use std::collections::HashSet;
    use notify::{recommended_watcher, RecursiveMode, Watcher, EventKind, Result as NotifyResult};
    use crate::{load_config, is_ignored};

    // ANSI color codes
    const RED: &str = "\x1b[31m";
    const GREEN: &str = "\x1b[32m";
    const YELLOW: &str = "\x1b[33m";
    const BLUE: &str = "\x1b[34m";
    const MAGENTA: &str = "\x1b[35m";
    const CYAN: &str = "\x1b[36m";
    const RESET: &str = "\x1b[0m";

    // Banner
    println!("{}   ╔═══════════════════════════════════╗{}", CYAN, RESET);
    println!("{}   ║    🚀 {}SMARTGIT WATCHER{}   ║{}", CYAN, MAGENTA, CYAN, RESET);
    println!("{}   ║      Watching for changes...      ║{}", CYAN, RESET);
    println!("{}   ╚═══════════════════════════════════╝{}", CYAN, RESET);

    let cfg = load_config();
    let delay = Duration::from_secs(cfg.timeout_seconds);

    // Shared set of changed files
    let changed_files = Arc::new(Mutex::new(HashSet::new()));
    let (tx, rx) = channel();
    let cf_clone = Arc::clone(&changed_files);

    // File watcher
    let mut watcher = recommended_watcher(move |res: NotifyResult<notify::Event>| {
        if let Ok(event) = res {
            if matches!(event.kind, EventKind::Modify(_) | EventKind::Create(_)) {
                for path in event.paths.iter() {
                    if let Some(name) = path.file_name().and_then(|f| f.to_str()) {
                        if !is_ignored(name) && !name.starts_with(".smartgit") {
                            cf_clone.lock().unwrap().insert(name.to_string());
                            let _ = tx.send(());
                        }
                    }
                }
            }
        }
    }).unwrap();
    watcher.watch(Path::new("."), RecursiveMode::Recursive).unwrap();
    fs::write(".smartgit/watch.pid", std::process::id().to_string()).unwrap();

    let mut last_change = Instant::now();
    // Determine the path to the built binary (Windows: target/debug/smartgit.exe)
    // let exe_path = if cfg!(windows) {
    //     "target/debug/smartgit.exe"
    // } else {
    //     "target/debug/smartgit"
    // };
    loop {
        thread::sleep(Duration::from_secs(1));

        // Drain pending change events and reset timer once
        if rx.try_recv().is_ok() {
            while rx.try_recv().is_ok() {}
            last_change = Instant::now();
            println!("\n{}📁 {}Change detected, timer reset.{}", BLUE, CYAN, RESET);
        }

        // Compute remaining time
        let elapsed = Instant::now().duration_since(last_change);
        let remaining = if delay > elapsed { delay - elapsed } else { Duration::ZERO };
        let secs = remaining.as_secs();

        // Display countdown
        print!("\r{}Time until auto-commit: {}{:>3}s{}", YELLOW, RED, secs, RESET);
        std::io::stdout().flush().unwrap();

        // When time is up, commit once and reset
        if remaining.is_zero() {
            let mut files = changed_files.lock().unwrap();
            if !files.is_empty() {
                let file_list: Vec<String> = files.drain().collect();
                println!("\n{}⏰ {}Auto-committing: {:?}{}", MAGENTA, GREEN, file_list, RESET);
                // Stage and commit changed files using the built binary directly
                for file in &file_list {
                    let exe_path = if cfg!(windows) {
                        "target\\debug\\smartgit.exe"
                    } else {
                        "target/debug/smartgit"
                    };
                    let _ = std::process::Command::new(exe_path)
                        .args(["add", file])
                        .status();
                }
                let msg = format!("Auto-commit: {:?}", file_list);
                let exe_path = if cfg!(windows) {
                    "target\\debug\\smartgit.exe"
                } else {
                    "target/debug/smartgit"
                };
                let _ = std::process::Command::new(exe_path)
                    .args(["commit", "--message", &msg])
                    .status();
                println!("{}✅ {}Commit complete. Resuming watcher...{}", GREEN, CYAN, RESET);
            }
            last_change = Instant::now();
        }
    }
}


fn hash_file(path: &Path) -> Option<String> {
    let mut file_data = Vec::new();
    if let Ok(mut f) = fs::File::open(path) {
        if f.read_to_end(&mut file_data).is_ok() {
            let mut hasher = Sha1::new();
            hasher.update(&file_data);
            return Some(format!("{:x}", hasher.finalize()));
        }
    }
    None
}


fn is_ignored(filename: &str) -> bool {
    let ignore_path = ".smartgitignore";
    if !Path::new(ignore_path).exists() {
        return false;
    }

    if let Ok(lines) = fs::read_to_string(ignore_path) {
        for pattern in lines.lines().map(|l| l.trim()) {
            if pattern.is_empty() || pattern.starts_with('#') {
                continue;
            }

            if pattern == filename {
                return true;
            }

            if pattern.ends_with("/*") {
                let dir = pattern.trim_end_matches("/*");
                if filename.starts_with(dir) {
                    return true;
                }
            }
        }
    }

    false
}

#[derive(Serialize, Deserialize)]
struct Config {
    timeout_seconds: u64,
}

fn load_config() -> Config {
    let path = ".smartgit/config.json";
    if Path::new(path).exists() {
        if let Ok(data) = fs::read_to_string(path) {
            if let Ok(cfg) = serde_json::from_str::<Config>(&data) {
                return cfg;
            }
        }
    }
    Config { timeout_seconds: 600 } // default: 10 min
}

fn save_config(cfg: &Config) {
    let _ = fs::write(".smartgit/config.json", serde_json::to_string_pretty(cfg).unwrap());
}

fn get_system_metadata() -> (String, String, String) {
    let hostname = hostname::get()
        .unwrap_or_else(|_| "unknown".into())
        .to_string_lossy()
        .to_string();
    
    let os = format!("{} {}", 
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    
    let username = whoami::username();
    
    (hostname, os, username)
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Init => {
            let repo_path = ".smartgit";
            let objects_path = ".smartgit/objects";
            info("Tip: run `smartgit watch` to enable auto-commits after file changes.");

            // Always ensure objects/ exists
            if !Path::new(repo_path).exists() {
                match fs::create_dir_all(objects_path) {
                    Ok(_) => success("Initialized empty SmartGit repository in .smartgit/"),
                    Err(e) => error(&format!("Failed to create repo: {}", e)),
                }
                // Create .smartgitignore with sensible defaults if it doesn't exist
                let ignore_path = ".smartgitignore";
                if !Path::new(ignore_path).exists() {
                    let default = "Cargo.lock\n*.log\ntarget/\n.gitignore\nCargo.toml\nwatch.pid\n";
                    let _ = fs::write(ignore_path, default);
                    info("Created .smartgitignore with default patterns.");
                }
            } else {
                if !Path::new(objects_path).exists() {
                    match fs::create_dir_all(objects_path) {
                        Ok(_) => success("Fixed missing .smartgit/objects/ directory"),
                        Err(e) => error(&format!("Error repairing repo structure: {}", e)),
                    }
                } else {
                    info("Repository already initialized.");
                }
            }
        }

        Commands::Add { file } => {
            if !Path::new(".smartgit").exists() {
                error("Not a SmartGit repository. Run `smartgit init` first.");
                return;
            }

            let path = Path::new(&file);
            if !path.exists() {
                error(&format!("File '{}' not found.", file));
                return;
            }

            let mut file_data = Vec::new();
            match fs::File::open(path) {
                Ok(mut f) => {
                    if f.read_to_end(&mut file_data).is_err() {
                        error("Error reading file.");
                        return;
                    }
                }
                Err(_) => {
                    error(&format!("Could not open file '{}'.", file));
                    return;
                }
            }
            if is_ignored(&file) {
                warning(&format!("'{}' is ignored and will not be added.", file));
                return;
            }
            // Hash contents
            let mut hasher = Sha1::new();
            hasher.update(&file_data);
            let hash = hasher.finalize();
            let hash_str = format!("{:x}", hash);

            // Write to .smartgit/objects/<hash>
            let object_path = format!(".smartgit/objects/{}", hash_str);
            if let Err(e) = fs::write(&object_path, &file_data) {
                error(&format!("Failed to write object: {}", e));
                return;
            }

            // Write to index
            let index_path = ".smartgit/index.json";
            let mut index: Vec<IndexEntry> = if Path::new(index_path).exists() {
                let data = fs::read_to_string(index_path).unwrap_or_default();
                serde_json::from_str(&data).unwrap_or_default()
            } else {
                Vec::new()
            };

            // Don't add the file if it's already in the index
            let already_added = index.iter().any(|e| e.filename == *file);
            if already_added {
                info(&format!("'{}' is already staged.", file));
                return;
            }

            index.push(IndexEntry {
                filename: file.clone(),
                hash: hash_str.clone(),
            });

            if let Ok(json) = serde_json::to_string_pretty(&index) {
                let _ = fs::write(index_path, json);
            }

            success(&format!("Added '{}', stored as object {}", file, hash_str));
        }

        Commands::Commit { message } => {
            let index_path = ".smartgit/index.json";
            let commits_path = ".smartgit/commits.json";

            if !Path::new(index_path).exists() {
                info("Nothing to commit. Staging area is empty.");
                return;
            }

            let index_data = fs::read_to_string(index_path).unwrap_or_default();
            let staged_files: Vec<IndexEntry> = serde_json::from_str(&index_data).unwrap_or_default();

            if staged_files.is_empty() {
                info("Nothing to commit. Index is empty.");
                return;
            }

            // Get current time
            let timestamp = chrono::Utc::now().to_rfc3339();

            // Get system metadata
            let (hostname, os, username) = get_system_metadata();

            let commit = Commit {
                message: message.clone(),
                timestamp: timestamp.clone(),
                files: staged_files.clone(),
                hostname,
                os,
                username,
            };

            // Load existing commits
            let mut commits: Vec<Commit> = if Path::new(commits_path).exists() {
                let data = fs::read_to_string(commits_path).unwrap_or_default();
                serde_json::from_str(&data).unwrap_or_default()
            } else {
                Vec::new()
            };

            commits.push(commit.clone());

            if let Ok(json) = serde_json::to_string_pretty(&commits) {
                let _ = fs::write(commits_path, json);
                success(&format!("Committed {} files with message: \"{}\"", staged_files.len(), message));
            } else {
                error("Failed to write commit.");
            }

            // --- Store actual objects for each committed file ---
            let pb = create_progress_bar(staged_files.len() as u64, "Storing objects");
            for entry in &staged_files {
                let src_path = Path::new(&entry.filename);
                let obj_path = Path::new(".smartgit/objects").join(&entry.hash);
                if src_path.exists() {
                    if let Ok(data) = fs::read(&src_path) {
                        let _ = fs::write(&obj_path, &data);
                    }
                }
                pb.inc(1);
            }
            pb.finish_with_message("Objects stored");

            // Clear index (optional for now)
            let _ = fs::write(index_path, "[]");

            // --- Firestore Sync: Push commit to Firestore and update progress/tasks/objects ---
            use crate::config::FirebaseConfig;
            use crate::sync::FirestoreSync;
            let mut config = FirebaseConfig::load().expect("Run smartgit login first");
            config.ensure_project_name();
            let user_id = match config.id_token.split('.').nth(1) {
                Some(payload_b64) => {
                    let payload = URL_SAFE.decode(payload_b64).unwrap_or_default();
                    let json: serde_json::Value = serde_json::from_slice(&payload).unwrap_or_default();
                    json.get("user_id").and_then(|v| v.as_str()).unwrap_or("me").to_string()
                },
                None => "me".to_string(),
            };
            let project_name = config.project_name.clone().unwrap_or_else(|| "default".to_string());
            let mut sync = FirestoreSync::new(config.project_id.clone(), config.id_token.clone());
            
            // Show spinner for Firestore operations
            let spinner = create_spinner("Syncing with Firestore");
            
            // Ensure the project document exists in Firestore
            let _ = sync.ensure_project_doc(&user_id, &project_name);
            
            // Push commit with metadata
            if let Err(e) = sync.push_commit(&user_id, &project_name, &commit) {
                if let Some(resp) = e.downcast_ref::<reqwest::Error>() {
                    if let Some(status) = resp.status() {
                        error(&format!("Failed to sync commit to Firestore: {}", status));
                    } else {
                        error(&format!("Failed to sync commit to Firestore: {}", resp));
                    }
                } else {
                    error(&format!("Failed to sync commit to Firestore: {e}"));
                }
            }

            // Push progress data
            let project_tasks = ProjectTasks::load();
            let progress = ProgressData::from_tasks(&project_tasks);
            if let Err(e) = sync.push_progress_doc(&user_id, &project_name, &progress, &timestamp) {
                error(&format!("Failed to upload progress to Firestore: {e}"));
}

            spinner.finish_with_message("Sync complete");

            // Upload tasks.json as a document in Firestore under /progress/tasks
            let tasks_endpoint = format!(
                "https://firestore.googleapis.com/v1/projects/{}/databases/(default)/documents/users/{}/projects/{}/progress/tasks",
                config.project_id, user_id, project_name
            );
            let tasks_json = serde_json::to_string_pretty(&project_tasks).unwrap_or_default();
            let tasks_body = serde_json::json!({
                "fields": {
                    "json": { "stringValue": tasks_json }
                }
            });
            let client = reqwest::blocking::Client::new();
            let tasks_resp = client.patch(&tasks_endpoint)
                .bearer_auth(&config.id_token)
                .json(&tasks_body)
                .send();
            match &tasks_resp {
                Ok(r) if !r.status().is_success() => {
                    eprintln!("Failed to upload tasks to Firestore: {}", r.status());
                    // Optionally: let text = r.text().unwrap_or_default(); eprintln!("Response: {}", text);
                },
                Err(e) => eprintln!("Failed to upload tasks to Firestore: {e}"),
                _ => (),
            }
            // Upload progress.json as a new document in Firestore under /progress (one per commit)
            let progress_collection_endpoint = format!(
                "https://firestore.googleapis.com/v1/projects/{}/databases/(default)/documents/users/{}/projects/{}/progress",
                config.project_id, user_id, project_name
            );
            let progress_json = serde_json::to_string_pretty(&progress).unwrap_or_default();
            let progress_body = serde_json::json!({
                "fields": {
                    "json": { "stringValue": progress_json },
                    "timestamp": { "stringValue": timestamp.clone() }
                }
            });
            let _ = client.post(&progress_collection_endpoint)
                .bearer_auth(&config.id_token)
                .json(&progress_body)
                .send();
            // Optionally, upload tasks.json as a new document in /progress as well (snapshot per commit)
            let tasks_collection_endpoint = format!(
                "https://firestore.googleapis.com/v1/projects/{}/databases/(default)/documents/users/{}/projects/{}/progress",
                config.project_id, user_id, project_name
            );
            let tasks_json = serde_json::to_string_pretty(&project_tasks).unwrap_or_default();
            let tasks_body = serde_json::json!({
                "fields": {
                    "json": { "stringValue": tasks_json },
                    "type": { "stringValue": "tasks" },
                    "timestamp": { "stringValue": timestamp.clone() }
                }
            });
            let _ = client.post(&tasks_collection_endpoint)
                .bearer_auth(&config.id_token)
                .json(&tasks_body)
                .send();
            // Upload all committed objects (file hashes) as a document in Firestore under /objects
            let mut objects_map = serde_json::Map::new();
            for entry in &staged_files {
                let obj_path = Path::new(".smartgit/objects").join(&entry.hash);
                if obj_path.exists() {
                    if let Ok(data) = fs::read(&obj_path) {
                        // Encode as base64 for Firestore
                        let b64 = base64::engine::general_purpose::STANDARD.encode(&data);
                        objects_map.insert(entry.hash.clone(), serde_json::json!({"stringValue": b64}));
                    }
                }
            }
            if !objects_map.is_empty() {
                let objects_endpoint = format!(
                    "https://firestore.googleapis.com/v1/projects/{}/databases/(default)/documents/users/{}/projects/{}/objects/objects",
                    config.project_id, user_id, project_name
                );
                let objects_body = serde_json::json!({
                    "fields": objects_map
                });
                let _ = client.patch(&objects_endpoint)
                    .bearer_auth(&config.id_token)
                    .json(&objects_body)
                    .send();
            }
            // --- End Firestore Sync ---

            // After commit, check tasks with LLM
            let mut project_tasks = ProjectTasks::load();
            let goal = ProjectGoal::load().map(|g| g.description).unwrap_or_default();
            // Read all committed files' contents
            let mut codebase = String::new();
            for entry in &staged_files {
                let object_path = format!(".smartgit/objects/{}", entry.hash);
                if let Ok(content) = fs::read_to_string(&object_path) {
                    codebase.push_str(&format!("\n// File: {}\n{}\n", entry.filename, content));
                }
            }
            let api_key = std::env::var("GROQ_API_KEY").expect("Please set the GROQ_API_KEY environment variable");
            let model = std::env::var("GROQ_MODEL").unwrap_or_else(|_| "llama-3.3-70b-versatile".into());
            for task in project_tasks.tasks.iter_mut() {
                if task.completed { continue; }
                let prompt = format!(
                    "Project goal: '{}'.\nTask: {}\nCodebase:\n{}\nIs this task now complete? Answer 'Yes' or 'No', and give a short reason. If yes, also rate the code quality for this task (1-10). Output format: Complete: Yes/No\nReason: ...\nQuality: N (if complete)",
                    goal, task.description, codebase
                );
                let payload = serde_json::json!({
                    "model": model,
                    "messages": [
                        { "role": "user", "content": prompt }
                    ],
                    "max_tokens": 256,
                    "temperature": 0.1
                });
                let client = reqwest::blocking::Client::new();
                let resp = client
                    .post("https://api.groq.com/openai/v1/chat/completions")
                    .bearer_auth(&api_key)
                    .json(&payload)
                    .send();
                if let Ok(r) = resp {
                    if r.status().is_success() {
                        if let Ok(json) = r.json::<serde_json::Value>() {
                            if let Some(choices) = json.get("choices").and_then(|v| v.as_array()) {
                                if let Some(choice) = choices.get(0) {
                                    if let Some(content) = choice
                                        .get("message")
                                        .and_then(|m| m.get("content"))
                                        .and_then(|v| v.as_str())
                                    {
                                        let mut complete = false;
                                        let mut reason = String::new();
                                        let mut quality = 0u32;
                                        for line in content.lines() {
                                            if line.to_lowercase().contains("complete:") {
                                                if line.to_lowercase().contains("yes") {
                                                    complete = true;
                                                }
                                            } else if line.to_lowercase().contains("reason:") {
                                                reason = line.splitn(2, ':').nth(1).unwrap_or("").trim().to_string();
                                            } else if line.to_lowercase().contains("quality:") {
                                                let n = line.splitn(2, ':').nth(1).unwrap_or("").trim().split_whitespace().next().unwrap_or("");
                                                quality = n.parse().unwrap_or(0);
                                            }
                                        }
                                        if complete {
                                            task.completed = true;
                                            task.xp += task.difficulty * 10;
                                            task.completion_reason = Some(reason.clone());
                                            println!("✅ Task {} complete! +{} XP\nReason: {}\nQuality: {}", task.id, task.difficulty * 10, reason, quality);
                                        } else {
                                            task.completion_reason = Some(reason.clone());
                                            println!("❌ Task {} not complete. Reason: {}", task.id, reason);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            project_tasks.save();
        }
        

        Commands::Status => {
            let index_path = ".smartgit/index.json";
            let mut staged: Vec<IndexEntry> = Vec::new();
            let mut modified = Vec::new();
            let mut deleted = Vec::new();

            if Path::new(index_path).exists() {
                let index_data = fs::read_to_string(index_path).unwrap_or_default();
                staged = serde_json::from_str(&index_data).unwrap_or_default();

                for entry in &staged {
                    let path = Path::new(&entry.filename);
                    if !path.exists() {
                        deleted.push(entry.filename.clone());
                    } else if let Some(current_hash) = hash_file(path) {
                        if current_hash != entry.hash {
                            modified.push(entry.filename.clone());
                        }
                    }
                }
            }

            // Detect untracked files
            let mut untracked = Vec::new();
            if let Ok(entries) = fs::read_dir(".") {
                for entry in entries.flatten() {
                    let path = entry.path();
                    let name = entry.file_name().to_string_lossy().to_string();

                    if name.starts_with(".smartgit") || name == "target" {
                        continue; // ignore internal files
                    }

                    let already_tracked = staged.iter().any(|e| e.filename == name);
                    if !already_tracked && path.is_file() && !is_ignored(&name) {
                        untracked.push(name);
                    }
                }
            }

            // Print results
            println!("=== SmartGit Status ===");
            if !staged.is_empty() {
                println!("📦 Staged:");
                for f in &staged {
                    println!("  {}", f.filename);
                }
            }

            if !modified.is_empty() {
                println!("🔄 Modified:");
                for f in &modified {
                    println!("  {}", f);
                }
            }

            if !deleted.is_empty() {
                println!("❌ Deleted:");
                for f in &deleted {
                    println!("  {}", f);
                }
            }

            if !untracked.is_empty() {
                println!("🆕 Untracked:");
                for f in &untracked {
                    println!("  {}", f);
                }
            }

            if staged.is_empty() && modified.is_empty() && deleted.is_empty() && untracked.is_empty() {
                println!("✅ Working directory clean.");
            }
        }

        Commands::Log => {
            let commits_path = ".smartgit/commits.json";

            if !Path::new(commits_path).exists() {
                println!("No commits found.");
                return;
            }

            let data = fs::read_to_string(commits_path).unwrap_or_default();
            let commits: Vec<Commit> = serde_json::from_str(&data).unwrap_or_default();

            if commits.is_empty() {
                println!("No commits found.");
                return;
            }

            println!("=== SmartGit Commit History ===");
            for (i, commit) in commits.iter().enumerate().rev() {
                println!("🔖 Commit {}:", i + 1);
                println!("  📅 {}", commit.timestamp);
                println!("  📝 {}", commit.message);
                println!("  📦 Files:");
                for file in &commit.files {
                    println!("     - {}", file.filename);
                }
                println!();
            }
        }

        Commands::Restore { file } => {
            let commits_path = ".smartgit/commits.json";
            if !Path::new(commits_path).exists() {
                println!("No commits to restore from.");
                return;
            }

            let data = fs::read_to_string(commits_path).unwrap_or_default();
            let commits: Vec<Commit> = serde_json::from_str(&data).unwrap_or_default();
            if commits.is_empty() {
                println!("No commits found.");
                return;
            }

            // Find the last commit containing the file
            for commit in commits.iter().rev() {
                if let Some(entry) = commit.files.iter().find(|e| e.filename == *file) {
                    let object_path = format!(".smartgit/objects/{}", entry.hash);
                    if Path::new(&object_path).exists() {
                        if let Ok(content) = fs::read(&object_path) {
                            let _ = fs::write(&file, content);
                            println!("Restored '{}' from last commit.", file);
                            return;
                        }
                    }
                }
            }

            println!("File '{}' not found in any commit.", file);
        }

        Commands::Diff { from, to } => {
            let commits_path = ".smartgit/commits.json";
            if !Path::new(commits_path).exists() {
                println!("No commits found.");
                return;
            }

            let data = fs::read_to_string(commits_path).unwrap_or_default();
            let commits: Vec<Commit> = serde_json::from_str(&data).unwrap_or_default();

            if commits.is_empty() {
                println!("No commits found.");
                return;
            }

            // Get the two commits to compare
            let (from_commit, to_commit) = match (from, to) {
                (None, None) => {
                    // Compare working directory with HEAD
                    let head = commits.last().unwrap();
                    let mut working_files = std::collections::HashMap::new();
                    
                    // Read current working directory files
                    for entry in &head.files {
                        let path = Path::new(&entry.filename);
                        if path.exists() {
                            if let Ok(content) = fs::read_to_string(path) {
                                working_files.insert(entry.filename.clone(), content);
                            }
                        }
                    }
                    
                    // Compare each file
                    for entry in &head.files {
                        let path = Path::new(&entry.filename);
                        if path.exists() {
                            if let Ok(current_content) = fs::read_to_string(path) {
                                let obj_path = Path::new(".smartgit/objects").join(&entry.hash);
                                if let Ok(old_content) = fs::read_to_string(&obj_path) {
                                    if current_content != old_content {
                                        println!("\n=== Changes in {} ===", entry.filename);
                                        let diff = TextDiff::from_lines(&old_content, &current_content);
                                        for change in diff.iter_all_changes() {
                                            let sign = match change.tag() {
                                                ChangeTag::Delete => "-",
                                                ChangeTag::Insert => "+",
                                                ChangeTag::Equal => " ",
                                            };
                                            let style = match change.tag() {
                                                ChangeTag::Delete => "\x1b[31m", // Red
                                                ChangeTag::Insert => "\x1b[32m", // Green
                                                ChangeTag::Equal => "\x1b[0m",   // Reset
                                            };
                                            print!("{}{}{}{}", style, sign, change, "\x1b[0m");
                                        }
                                    }
                                }
                            }
                        }
                    }
                    return;
                },
                (Some(from_idx), Some(to_idx)) => {
                    if from_idx >= commits.len() || to_idx >= commits.len() {
                        println!("Invalid commit indices.");
                        return;
                    }
                    (&commits[from_idx], &commits[to_idx])
                },
                (Some(from_idx), None) => {
                    if from_idx >= commits.len() {
                        println!("Invalid commit index.");
                        return;
                    }
                    (&commits[from_idx], commits.last().unwrap())
                },
                (None, Some(to_idx)) => {
                    if to_idx >= commits.len() {
                        println!("Invalid commit index.");
                        return;
                    }
                    (commits.last().unwrap(), &commits[to_idx])
                },
            };

            // Compare files between the two commits
            let mut from_files = std::collections::HashMap::new();
            let mut to_files = std::collections::HashMap::new();

            // Load files from the "from" commit
            for entry in &from_commit.files {
                let obj_path = Path::new(".smartgit/objects").join(&entry.hash);
                if let Ok(content) = fs::read_to_string(&obj_path) {
                    from_files.insert(entry.filename.clone(), content);
                }
            }

            // Load files from the "to" commit
            for entry in &to_commit.files {
                let obj_path = Path::new(".smartgit/objects").join(&entry.hash);
                if let Ok(content) = fs::read_to_string(&obj_path) {
                    to_files.insert(entry.filename.clone(), content);
                }
            }

            // Compare files
            let all_files: std::collections::HashSet<_> = from_files.keys()
                .chain(to_files.keys())
                .cloned()
                .collect();

            for filename in all_files {
                let from_content = from_files.get(&filename);
                let to_content = to_files.get(&filename);

                match (from_content, to_content) {
                    (Some(from), Some(to)) if from != to => {
                        println!("\n=== Changes in {} ===", filename);
                        let diff = TextDiff::from_lines(from, to);
                        for change in diff.iter_all_changes() {
                            let sign = match change.tag() {
                                ChangeTag::Delete => "-",
                                ChangeTag::Insert => "+",
                                ChangeTag::Equal => " ",
                            };
                            let style = match change.tag() {
                                ChangeTag::Delete => "\x1b[31m", // Red
                                ChangeTag::Insert => "\x1b[32m", // Green
                                ChangeTag::Equal => "\x1b[0m",   // Reset
                            };
                            print!("{}{}{}{}", style, sign, change, "\x1b[0m");
                        }
                    },
                    (Some(_), None) => println!("\n=== Deleted: {} ===", filename),
                    (None, Some(_)) => println!("\n=== Added: {} ===", filename),
                    _ => {},
                }
            }
        }

        Commands::History { file } => {
            let commits_path = ".smartgit/commits.json";
            if !Path::new(commits_path).exists() {
                println!("No commits found.");
                return;
            }

            let data = fs::read_to_string(commits_path).unwrap_or_default();
            let commits: Vec<Commit> = serde_json::from_str(&data).unwrap_or_default();

            if commits.is_empty() {
                println!("No commits found.");
                return;
            }

            // Find all commits that modified this file
            let mut file_history = Vec::new();
            for (i, commit) in commits.iter().enumerate() {
                if commit.files.iter().any(|f| f.filename == file) {
                    file_history.push((i, commit));
                }
            }

            if file_history.is_empty() {
                println!("No history found for file: {}", file);
                return;
            }

            println!("=== History for {} ===", file);
            for (i, commit) in file_history.iter().rev() {
                println!("\nCommit {}:", i + 1);
                println!("  📅 {}", commit.timestamp);
                println!("  📝 {}", commit.message);
                
                // Show the file's hash in this commit
                if let Some(entry) = commit.files.iter().find(|f| f.filename == file) {
                    println!("  🔑 Hash: {}", entry.hash);
                }
            }
        }

        Commands::Watch { on, off, time } => {
            use std::process::{Command as ProcessCommand, Stdio};
            use std::env;

            if off {
                let pid_path = ".smartgit/watch.pid";
                if Path::new(pid_path).exists() {
                    if let Ok(pid) = fs::read_to_string(pid_path) {
                        let pid = pid.trim().parse::<usize>().unwrap_or(0);
                        if pid > 0 {
                            let mut sys = System::new_all();
                            sys.refresh_all();
                            if let Some(process) = sys.process(sysinfo::Pid::from(pid)) {
                                process.kill();
                                println!("❌ SmartGit Watcher stopped.");
                            } else {
                                println!("Process not found.");
                            }
                        }
                        let _ = fs::remove_file(pid_path);
                    } else {
                        println!("Could not read watch.pid.");
                    }
                } else {
                    println!("No watcher is running.");
                }
                return;
            }

            if let Some(min) = time {
                let cfg = Config { timeout_seconds: min * 60 };
                save_config(&cfg);
                println!("⏱️  Timeout set to {} minute(s).", min);
                return;
            }

            if on {
                let exe = env::current_exe().unwrap();
                let exe_path = exe.to_str().unwrap();

                #[cfg(windows)]
                {
                    // Windows-specific process spawning
                    let _ = ProcessCommand::new("cmd")
                        .args(&[
                            "/C",
                            "start",   // launch a new window
                            "",        // window title
                            exe_path,  // path to smartgit.exe
                            "internal-watch",
                        ])
                        .stdout(Stdio::null())
                        .stderr(Stdio::null())
                        .spawn()
                        .expect("failed to spawn watcher");
                }

                #[cfg(not(windows))]
                {
                    // Unix-like systems process spawning
                    let _ = ProcessCommand::new(exe_path)
                        .arg("internal-watch")
                        .stdout(Stdio::null())
                        .stderr(Stdio::null())
                        .spawn()
                        .expect("failed to spawn watcher");
                }

                println!("✅ SmartGit watcher started in background.");
                return;
            }
            println!("Usage:\n  --on\n  --off\n  --time <minutes>");
        }

        Commands::InternalWatch => run_internal_watcher(),

        Commands::Goal { cmd } => handle_goal(cmd),

        Commands::Progress => {
            let tasks = ProjectTasks::load();
            let progress = ProgressData::from_tasks(&tasks);
            progress.save();
            println!("=== Progress ===");
            println!("Total XP: {}", progress.xp);
            print_progress_bar(progress.completed_tasks, progress.total_tasks);
        }

        Commands::Signup { email, password } => {
            use crate::auth::create_account;
            use crate::config::FirebaseConfig;
            
            // Use hardcoded Firebase config
            let api_key = "AIzaSyDHecGyk7UK5JCsgoLE1e8MUmjAbb2Fn0I";
            let project_id = "smartgit-abe0d";
            
            match create_account(&email, &password, api_key) {
                Ok(tokens) => {
                    let mut config = FirebaseConfig::default();
                    config.api_key = api_key.to_string();
                    config.project_id = project_id.to_string();
                    config.id_token = tokens.id_token;
                    config.refresh_token = tokens.refresh_token;
                    config.expires_at = tokens.expires_at;
                    config.user_id = tokens.local_id;
                    config.save();
                    
                    // Create initial project structure in Firestore
                    let mut sync = FirestoreSync::new(config.project_id.clone(), config.id_token.clone());
                    let _ = sync.ensure_project_doc(&config.user_id, "default");
                    
                    success("Account created successfully!");
                }
                Err(e) => error(&format!("Account creation failed: {}", e)),
            }
        }

        Commands::Login { email, password } => {
            use crate::auth::login;
            use crate::config::FirebaseConfig;
            
            // Use hardcoded Firebase config
            let api_key = "AIzaSyDHecGyk7UK5JCsgoLE1e8MUmjAbb2Fn0I";
            let project_id = "smartgit-abe0d";
            
            match login(&email, &password, api_key) {
                Ok(tokens) => {
                    let mut config = FirebaseConfig::default();
                    config.api_key = api_key.to_string();
                    config.project_id = project_id.to_string();
                    config.id_token = tokens.id_token;
                    config.refresh_token = tokens.refresh_token;
                    config.expires_at = tokens.expires_at;
                    config.user_id = tokens.local_id;
                    config.save();
                    success("Login successful and tokens saved.");
                }
                Err(e) => error(&format!("Login failed: {}", e)),
            }
        }

        Commands::Sync => {
            use crate::sync::{FirestoreSync, ProgressSync};
            use crate::config::FirebaseConfig;
            let config = FirebaseConfig::load().expect("Run smartgit login first");
            // Extract user id from id_token (sub claim in JWT)
            let user_id = match config.id_token.split('.').nth(1) {
                Some(payload_b64) => {
                    let payload = URL_SAFE.decode(payload_b64).unwrap_or_default();
                    let json: serde_json::Value = serde_json::from_slice(&payload).unwrap_or_default();
                    json.get("user_id").and_then(|v| v.as_str()).unwrap_or("me").to_string()
                },
                None => "me".to_string(),
            };
            let proj = "default".to_string();
            let data = ProgressData::from_tasks(&ProjectTasks::load());
            let mut sync = FirestoreSync::new(config.project_id.clone(), config.id_token.clone());
            match sync.push(&user_id, &proj, &data) {
                Ok(_) => println!("✅ Progress synced to Firebase."),
                Err(e) => eprintln!("Sync failed: {e}"),
            }
        }

        Commands::Pull { project } => {
            use crate::config::FirebaseConfig;
            use crate::sync::FirestoreSync;
            let config = FirebaseConfig::load().expect("Run smartgit login first");
            let user_id = match config.id_token.split('.').nth(1) {
                Some(payload_b64) => {
                    let payload = URL_SAFE.decode(payload_b64).unwrap_or_default();
                    let json: serde_json::Value = serde_json::from_slice(&payload).unwrap_or_default();
                    json.get("user_id").and_then(|v| v.as_str()).unwrap_or("me").to_string()
                },
                None => "me".to_string(),
            };
            let project_name = project.or(config.project_name.clone()).unwrap_or_else(|| "default".to_string());
            let mut sync = FirestoreSync::new(config.project_id.clone(), config.id_token.clone());
            match sync.pull_project(&user_id, &project_name) {
                Ok(_) => println!("✅ Pulled latest project data from Firestore for project: {}.", project_name),
                Err(e) => eprintln!("Failed to pull project: {e}"),
            }
        }

        Commands::ListProjects => {
            use crate::config::FirebaseConfig;
            use crate::sync::FirestoreSync;
            let config = FirebaseConfig::load().expect("Run smartgit login first");
            let user_id = match config.id_token.split('.').nth(1) {
                Some(payload_b64) => {
                    let payload = URL_SAFE.decode(payload_b64).unwrap_or_default();
                    let json: serde_json::Value = serde_json::from_slice(&payload).unwrap_or_default();
                    json.get("user_id").and_then(|v| v.as_str()).unwrap_or("me").to_string()
                },
                None => "me".to_string(),
            };
            let mut sync = FirestoreSync::new(config.project_id.clone(), config.id_token.clone());
            match sync.list_projects(&user_id) {
                Ok(projects) => {
                    println!("Available projects:");
                    for p in projects { println!("- {}", p); }
                },
                Err(e) => eprintln!("Failed to list projects: {e}"),
            }
        }

        Commands::SwitchProject { name } => {
            use crate::config::FirebaseConfig;
            let mut config = FirebaseConfig::load().expect("Run smartgit login first");
            config.project_name = Some(name.clone());
            config.save();
            println!("Switched to project: {}", name);
        }

        Commands::CurrentProject => {
            use crate::config::FirebaseConfig;
            let config = FirebaseConfig::load().expect("Run smartgit login first");
            let project = config.project_name.clone().unwrap_or_else(|| "default".to_string());
            println!("Current project: {}", project);
        }

        Commands::Revert { commit } => {
            let commits_path = ".smartgit/commits.json";
            if !Path::new(commits_path).exists() {
                println!("No commits found.");
                return;
            }

            let data = fs::read_to_string(commits_path).unwrap_or_default();
            let mut commits: Vec<Commit> = serde_json::from_str(&data).unwrap_or_default();

            if commits.is_empty() {
                println!("No commits found.");
                return;
            }

            // Convert to 0-based index
            let commit_idx = commit - 1;
            if commit_idx >= commits.len() {
                println!("Invalid commit index. Available commits: 1-{}", commits.len());
                return;
            }

            // Get the commit to revert and its parent
            let target_commit = &commits[commit_idx];
            let parent_commit = if commit_idx > 0 {
                Some(&commits[commit_idx - 1])
            } else {
                None
            };

            // Create a map of files in the parent commit
            let mut parent_files = std::collections::HashMap::new();
            if let Some(parent) = parent_commit {
                for entry in &parent.files {
                    let obj_path = Path::new(".smartgit/objects").join(&entry.hash);
                    if let Ok(content) = fs::read_to_string(&obj_path) {
                        parent_files.insert(entry.filename.clone(), content);
                    }
                }
            }

            // Create a map of files in the target commit
            let mut target_files = std::collections::HashMap::new();
            for entry in &target_commit.files {
                let obj_path = Path::new(".smartgit/objects").join(&entry.hash);
                if let Ok(content) = fs::read_to_string(&obj_path) {
                    target_files.insert(entry.filename.clone(), content);
                }
            }

            // Apply reverse changes to working directory
            let mut reverted_files = Vec::new();
            for (filename, _target_content) in &target_files {
                if let Some(parent_content) = parent_files.get(filename) {
                    // File existed in parent, restore it
                    let _ = fs::write(filename, parent_content);
                    reverted_files.push(filename.clone());
                } else {
                    // File was added in target commit, delete it
                    let _ = fs::remove_file(filename);
                }
            }

            // Check for files that were deleted in target commit
            for (filename, _) in &parent_files {
                if !target_files.contains_key(filename) {
                    // File was deleted in target commit, restore it
                    let obj_path = Path::new(".smartgit/objects").join(&parent_files[filename]);
                    if let Ok(content) = fs::read_to_string(&obj_path) {
                        let _ = fs::write(filename, content);
                        reverted_files.push(filename.clone());
                    }
                }
            }

            // Create a new commit for the revert
            let timestamp = chrono::Utc::now().to_rfc3339();
            let message = format!("Revert commit {}: {}", commit, target_commit.message);
            
            // Get system metadata
            let (hostname, os, username) = get_system_metadata();

            // Add all reverted files to staging
            let mut index = Vec::new();
            for filename in reverted_files {
                if let Ok(content) = fs::read_to_string(&filename) {
                    let mut hasher = Sha1::new();
                    hasher.update(content.as_bytes());
                    let hash = format!("{:x}", hasher.finalize());
                    
                    // Write to objects
                    let obj_path = Path::new(".smartgit/objects").join(&hash);
                    let _ = fs::write(&obj_path, content);
                    
                    index.push(IndexEntry {
                        filename,
                        hash,
                    });
                }
            }

            // Create and save the revert commit
            let revert_commit = Commit {
                message,
                timestamp,
                files: index,
                hostname,
                os,
                username,
            };
            commits.push(revert_commit);

            // Save updated commits
            if let Ok(json) = serde_json::to_string_pretty(&commits) {
                let _ = fs::write(commits_path, json);
                println!("✅ Created revert commit for commit {}", commit);
            }
        }

        Commands::Reset { commit } => {
            let commits_path = ".smartgit/commits.json";
            if !Path::new(commits_path).exists() {
                println!("No commits found.");
                return;
            }

            let data = fs::read_to_string(commits_path).unwrap_or_default();
            let mut commits: Vec<Commit> = serde_json::from_str(&data).unwrap_or_default();

            if commits.is_empty() {
                println!("No commits found.");
                return;
            }

            // Convert to 0-based index
            let commit_idx = commit - 1;
            if commit_idx >= commits.len() {
                println!("Invalid commit index. Available commits: 1-{}", commits.len());
                return;
            }

            // Get the target commit
            let target_commit = &commits[commit_idx];

            // Clear working directory (except .smartgit and ignored files)
            let ignored: Vec<String> = fs::read_dir(".")
                .unwrap()
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.file_name().to_string_lossy().to_string())
                .filter(|name| name == ".smartgit" || name == ".smartgitignore" || name == "src" || is_ignored(name))
                .collect();

            for entry in fs::read_dir(".").unwrap() {
                if let Ok(entry) = entry {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if !ignored.contains(&name) {
                        let path = entry.path();
                        if path.is_file() {
                            let _ = fs::remove_file(&path);
                        } else if path.is_dir() {
                            let _ = fs::remove_dir_all(&path);
                        }
                    }
                }
            }

            // Restore files from the target commit
            for entry in &target_commit.files {
                let obj_path = Path::new(".smartgit/objects").join(&entry.hash);
                if let Ok(content) = fs::read(&obj_path) {
                    let file_path = Path::new(&entry.filename);
                    if let Some(parent) = file_path.parent() {
                        let _ = fs::create_dir_all(parent);
                    }
                    let _ = fs::write(&file_path, content);
                }
            }

            // Truncate commits array
            commits.truncate(commit_idx + 1);

            // Save updated commits
            if let Ok(json) = serde_json::to_string_pretty(&commits) {
                let _ = fs::write(commits_path, json);
                println!("✅ Reset to commit {}", commit);
            }
        }

        Commands::Export { output } => {
            use zip::ZipWriter;
            use std::fs::File;
            use std::path::Path;
            use std::collections::HashMap;
            
            let output_path = output.unwrap_or_else(|| "smartgit-backup.zip".to_string());
            let output_file = match File::create(&output_path) {
                Ok(file) => file,
                Err(e) => {
                    error(&format!("Failed to create output file: {}", e));
                    return;
                }
            };
            
            let mut zip = ZipWriter::new(output_file);
            let options = zip::write::FileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated)
                .unix_permissions(0o755);
            
            // Function to add a file to the zip
            fn add_file_to_zip(
                zip: &mut ZipWriter<File>,
                path: &Path,
                base_path: &Path,
                options: zip::write::FileOptions,
            ) -> std::io::Result<()> {
                let name = match path.strip_prefix(base_path) {
                    Ok(name) => name,
                    Err(_) => path,
                };
                
                if path.is_file() {
                    zip.start_file(name.to_string_lossy(), options)?;
                    let mut file = File::open(path)?;
                    std::io::copy(&mut file, zip)?;
                } else if path.is_dir() {
                    zip.add_directory(name.to_string_lossy(), options)?;
                }
                Ok(())
            }
            
            // Add .smartgit directory
            let smartgit_path = Path::new(".smartgit");
            if !smartgit_path.exists() {
                error("No SmartGit repository found. Run 'smartgit init' first.");
                return;
            }
            
            // Create progress bar
            let pb = create_progress_bar(1, "Creating backup archive");
            
            // First, add the .smartgit directory structure
            for entry in walkdir::WalkDir::new(smartgit_path) {
                match entry {
                    Ok(entry) => {
                        if let Err(e) = add_file_to_zip(&mut zip, entry.path(), Path::new("."), options) {
                            error(&format!("Failed to add file to archive: {}", e));
                            return;
                        }
                    }
                    Err(e) => {
                        error(&format!("Failed to read directory entry: {}", e));
                        return;
                    }
                }
            }
            
            // Add .smartgitignore if it exists
            let ignore_path = Path::new(".smartgitignore");
            if ignore_path.exists() {
                if let Err(e) = add_file_to_zip(&mut zip, ignore_path, Path::new("."), options) {
                    error(&format!("Failed to add .smartgitignore to archive: {}", e));
                    return;
                }
            }
            
            // Get the most recent version of each file from commits
            let commits_path = ".smartgit/commits.json";
            if Path::new(commits_path).exists() {
                if let Ok(data) = fs::read_to_string(commits_path) {
                    if let Ok(commits) = serde_json::from_str::<Vec<Commit>>(&data) {
                        // Create a map to track the most recent version of each file
                        let mut latest_files: HashMap<String, &IndexEntry> = HashMap::new();
                        
                        // Iterate through commits in reverse order to get the most recent versions
                        for commit in commits.iter().rev() {
                            for file in &commit.files {
                                // Only add if we haven't seen this file before
                                latest_files.entry(file.filename.clone()).or_insert(file);
                            }
                        }
                        
                        // Add the most recent version of each file
                        for (_, file) in latest_files {
                            let file_path = Path::new(&file.filename);
                            if file_path.exists() {
                                if let Err(e) = add_file_to_zip(&mut zip, file_path, Path::new("."), options) {
                                    error(&format!("Failed to add file to archive: {}", e));
                                    return;
                                }
                            }
                        }
                    }
                }
            }
            
            // Finalize the zip file
            match zip.finish() {
                Ok(_) => {
                    pb.finish_with_message("Backup complete");
                    success(&format!("Repository exported to {}", output_path));
                }
                Err(e) => error(&format!("Failed to finalize archive: {}", e)),
            }
        }
    }
}

fn handle_goal(cmd: GoalCommand) {
    match cmd {
        GoalCommand::Set { description } => {
            let desc = description.join(" ");
            let goal = ProjectGoal { description: desc.clone() };
            goal.save();
            println!("✅ Project goal set to: {}", desc);
        }
        GoalCommand::Show => {
            if let Some(goal) = ProjectGoal::load() {
                println!("📌 Current project goal: {}", goal.description);
            } else {
                println!("No project goal set. Use `smartgit goal set <desc>`." );
            }
        }
        GoalCommand::Tasks => {
            let mut tasks = ProjectTasks::load();
            if !tasks.tasks.is_empty() {
                println!("Tasks have already been generated:");
                for task in &tasks.tasks {
                    let status = if task.completed { "✅" } else { "  " };
                    let xp = if task.xp > 0 { format!(" [XP: {}]", task.xp) } else { String::new() };
                    println!("  {}{}. {} [Difficulty: {}]{}", status, task.id, task.description, task.difficulty, xp);
                    if let Some(reason) = &task.completion_reason {
                        if task.completed {
                            println!("     Reason: {}", reason);
                        }
                    }
                }
                return;
            }
            if let Some(goal) = ProjectGoal::load() {
                println!("🚀 Generating checkpoint tasks for: {}", goal.description);
                let generated = generate_checkpoints(&goal.description);
                let parsed_tasks = parse_tasks_with_difficulty(&generated.join("\n"));
                for task in &parsed_tasks {
                    println!("  {}. {} [Difficulty: {}]", task.id, task.description, task.difficulty);
                }
                // Save generated tasks
                tasks.tasks = parsed_tasks;
                tasks.save();
            } else {
                println!("No project goal set. Use `smartgit goal set <desc>`." );
            }
        }
        GoalCommand::Task { cmd } => handle_task(cmd),
    }
}

fn handle_task(cmd: TaskSubCommand) {
    match cmd {
        TaskSubCommand::List => {
            let tasks = ProjectTasks::load();
            if tasks.tasks.is_empty() {
                println!("No tasks set. Run 'smartgit goal tasks' to generate tasks.");
            } else {
                println!("Current tasks:");
                for t in &tasks.tasks {
                    if t.completed {
                        println!("  {}. ✅ {} [Difficulty: {}] [XP: {}]", t.id, t.description, t.difficulty, t.xp);
                        if let Some(reason) = &t.completion_reason {
                            println!("     Reason: {}", reason);
                        }
                    } else {
                        println!("  {}.    {} [Difficulty: {}] [XP: {}]", t.id, t.description, t.difficulty, t.xp);
                    }
                }
            }
        }
        TaskSubCommand::Set { number, description } => {
            let mut tasks = ProjectTasks::load();
            let desc = description.join(" ");
            if number == 0 {
                println!("Task numbers start at 1.");
                return;
            }
            if tasks.tasks.len() < number {
                // Fill with default tasks up to the requested number
                while tasks.tasks.len() < number {
                    let id = tasks.tasks.len() + 1;
                    tasks.tasks.push(Task {
                        id,
                        description: String::new(),
                        completed: false,
                        difficulty: 1,
                        xp: 0,
                        completion_reason: None,
                    });
                }
            }
            // Get project goal and all task descriptions for context
            let goal = ProjectGoal::load().map(|g| g.description).unwrap_or_default();
            let other_tasks: Vec<String> = tasks.tasks.iter().map(|t| t.description.clone()).collect();
            // Prompt LLM for difficulty rating
            let api_key = std::env::var("GROQ_API_KEY").expect("Please set the GROQ_API_KEY environment variable");
            let model = std::env::var("GROQ_MODEL").unwrap_or_else(|_| "llama-3.3-70b-versatile".into());
            let prompt = format!(
                "Given the project goal: '{}', and these tasks: [{}], rate the following new task for difficulty (1-10) and explain your rating.\nTask: {}\nOutput only: Difficulty: N (as an integer) and a short explanation.",
                goal,
                other_tasks.join("; "),
                desc
            );
            let payload = serde_json::json!({
                "model": model,
                "messages": [
                    { "role": "user", "content": prompt }
                ],
                "max_tokens": 128,
                "temperature": 0.1
            });
            let client = reqwest::blocking::Client::new();
            let resp = client
                .post("https://api.groq.com/openai/v1/chat/completions")
                .bearer_auth(api_key)
                .json(&payload)
                .send();
            let (difficulty, explanation) = if let Ok(r) = resp {
                if r.status().is_success() {
                    if let Ok(json) = r.json::<serde_json::Value>() {
                        if let Some(choices) = json.get("choices").and_then(|v| v.as_array()) {
                            if let Some(choice) = choices.get(0) {
                                if let Some(content) = choice
                                    .get("message")
                                    .and_then(|m| m.get("content"))
                                    .and_then(|v| v.as_str())
                                {
                                    // Try to parse Difficulty: N and explanation
                                    let mut diff = 1u32;
                                    let mut expl = String::new();
                                    for line in content.lines() {
                                        if let Some(idx) = line.find("Difficulty:") {
                                            let n = line[idx+11..].trim().split_whitespace().next().unwrap_or("");
                                            diff = n.parse().unwrap_or(1);
                                        } else if line.to_lowercase().contains("explanation") {
                                            expl = line.splitn(2, ':').nth(1).unwrap_or("").trim().to_string();
                                        } else if expl.is_empty() && !line.trim().is_empty() {
                                            expl = line.trim().to_string();
                                        }
                                    }
                                    (diff, expl)
                                } else { (1, String::new()) }
                            } else { (1, String::new()) }
                        } else { (1, String::new()) }
                    } else { (1, String::new()) }
                } else { (1, String::new()) }
            } else { (1, String::new()) };
            // Update the task at the given index
            let id = number;
            tasks.tasks[number - 1] = Task {
                id,
                description: desc.clone(),
                completed: false,
                difficulty,
                xp: 0,
                completion_reason: None,
            };
            tasks.save();
            println!("Set task {} to: {} [Difficulty: {}]", number, desc, difficulty);
            if !explanation.is_empty() {
                println!("AI explanation: {}", explanation);
            }
        }
        TaskSubCommand::Prompt { number, prompt } => {
            let tasks = ProjectTasks::load();
            if number == 0 || number > tasks.tasks.len() {
                println!("Invalid task number.");
                return;
            }
            let task = &tasks.tasks[number - 1];
            let user_prompt = prompt.join(" ");
            let full_prompt = format!("For the following task: '{}', {}", task.description, user_prompt);
            let api_key = std::env::var("GROQ_API_KEY").expect("Please set the GROQ_API_KEY environment variable");
            let model = std::env::var("GROQ_MODEL").unwrap_or_else(|_| "llama-3.3-70b-versatile".into());
            let payload = serde_json::json!({
                "model": model,
                "messages": [
                    { "role": "user", "content": full_prompt }
                ],
                "max_tokens": 512,
                "temperature": 0.1
            });
            let client = reqwest::blocking::Client::new();
            let resp = client
                .post("https://api.groq.com/openai/v1/chat/completions")
                .bearer_auth(api_key)
                .json(&payload)
                .send();
            if let Ok(r) = resp {
                if r.status().is_success() {
                    if let Ok(json) = r.json::<serde_json::Value>() {
                        if let Some(choices) = json.get("choices").and_then(|v| v.as_array()) {
                            if let Some(choice) = choices.get(0) {
                                if let Some(content) = choice
                                    .get("message")
                                    .and_then(|m| m.get("content"))
                                    .and_then(|v| v.as_str())
                                {
                                    println!("AI response:\n{}", content);
                                    return;
                                }
                            }
                        }
                    }
                } else {
                    eprintln!("GroqCloud API error: {}", r.status());
                }
            } else {
                eprintln!("Failed to call GroqCloud API");
            }
        }
    }
}


/// Generate checkpoint tasks by calling the GroqCloud OpenAI-compatible Completions API
fn generate_checkpoints(goal: &str) -> Vec<String> {
    // Load API key and model from environment
    let api_key = env::var("GROQ_API_KEY").expect("Please set the GROQ_API_KEY environment variable");
    let model = env::var("GROQ_MODEL").unwrap_or_else(|_| "llama-3.3-70b-versatile".into());

    // Improved prompt for consistent, actionable checkpoints
    let prompt = format!(
        "Generate exactly 5 concise development tasks to achieve the following goal: '{}'.\nEach task must be on its own line, in this format: Task description (Difficulty: N), where N is an integer from 1 (easiest) to 10 (hardest).\nDo not include any commentary, explanations, or extra numbers. Only output the 5 tasks in the specified format.",
        goal
    );

    // Prepare chat payload
    let payload = serde_json::json!({
        "model": model,
        "messages": [
            { "role": "user", "content": prompt }
        ],
        "max_tokens": 512,
        "temperature": 0.1
    });

    let client = reqwest::blocking::Client::new();
    let resp = client
        .post("https://api.groq.com/openai/v1/chat/completions")
        .bearer_auth(api_key)
        .json(&payload)
        .send();

    if let Ok(r) = resp {
        if r.status().is_success() {
            if let Ok(json) = r.json::<serde_json::Value>() {
                if let Some(choices) = json.get("choices").and_then(|v| v.as_array()) {
                    if let Some(choice) = choices.get(0) {
                        if let Some(content) = choice
                            .get("message")
                            .and_then(|m| m.get("content"))
                            .and_then(|v| v.as_str())
                        {
                            // Clean lines and strip any leading numbering
                            return content
                                .lines()
                                .map(str::trim)
                                .filter(|s| !s.is_empty())
                                .map(|s| {
                                    let s = s.trim_start();
                                    // Remove leading digits and punctuation
                                    let s = s.trim_start_matches(|c: char| c.is_digit(10) || c == '.' || c == ')' );
                                    s.trim_start().to_string()
                                })
                                .collect();
                        }
                    }
                }
            }
        } else {
            eprintln!("GroqCloud API error: {}", r.status());
        }
    } else {
        eprintln!("Failed to call GroqCloud API");
    }

    // Fallback stub if API fails
    vec![
        format!("Define project scope for: {}", goal),
        "Initialize repository and config".into(),
        "Implement core add/commit/status commands".into(),
        "Add file watcher for auto-commit".into(),
        "Integrate LLM for task planning".into(),
    ]
}

fn parse_tasks_with_difficulty(llm_output: &str) -> Vec<Task> {
    use regex::Regex;
    let re = Regex::new(r"^(?:\d+\.\s*)?(.*)\(Difficulty:\s*(\d+)\)").unwrap();
    let mut tasks = Vec::new();
    let mut id = 1;
    for line in llm_output.lines() {
        let line = line.trim();
        if line.is_empty() { continue; }
        if let Some(caps) = re.captures(line) {
            let desc = caps.get(1).map(|m| m.as_str().trim().trim_end_matches('-').trim()).unwrap_or("");
            let diff = caps.get(2).and_then(|m| m.as_str().parse::<u32>().ok()).unwrap_or(1);
            tasks.push(Task {
                id,
                description: desc.to_string(),
                completed: false,
                difficulty: diff,
                xp: 0,
                completion_reason: None,
            });
            id += 1;
        }
    }
    tasks
}

