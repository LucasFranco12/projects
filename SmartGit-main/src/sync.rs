use anyhow::{Result, anyhow};
use crate::config::FirebaseConfig;
use crate::ProgressData;
use base64::Engine;
use crate::is_ignored;

pub trait ProgressSync {
    fn push(&mut self, user: &str, proj: &str, data: &ProgressData) -> Result<()>;
    fn pull(&mut self, user: &str) -> Result<UserProgress>;
}

pub struct FirestoreSync {
    pub project_id: String,
    pub id_token: String,
}

impl FirestoreSync {
    pub fn new(project_id: String, id_token: String) -> Self {
        Self { project_id, id_token }
    }

    // Helper method to ensure valid token before Firestore operations
    fn ensure_valid_token(&mut self) -> Result<()> {
        let mut config = FirebaseConfig::load().ok_or_else(|| anyhow!("No Firebase config found"))?;
        config.ensure_valid_token()?;
        self.id_token = config.id_token;
        Ok(())
    }

    pub fn push_commit(&mut self, user: &str, project: &str, commit: &crate::Commit) -> Result<()> {
        self.ensure_valid_token()?;
        let endpoint = format!(
            "https://firestore.googleapis.com/v1/projects/{}/databases/(default)/documents/users/{}/projects/{}/commits",
            self.project_id, user, project
        );
        let files_json: Vec<_> = commit.files.iter().map(|f| serde_json::json!({
            "mapValue": {
                "fields": {
                    "filename": { "stringValue": &f.filename },
                    "hash": { "stringValue": &f.hash }
                }
            }
        })).collect();
        let body = serde_json::json!({
            "fields": {
                "message": { "stringValue": &commit.message },
                "timestamp": { "stringValue": &commit.timestamp },
                "files": { "arrayValue": { "values": files_json } },
                "hostname": { "stringValue": &commit.hostname },
                "os": { "stringValue": &commit.os },
                "username": { "stringValue": &commit.username }
            }
        });
        let client = reqwest::blocking::Client::new();
        let resp = client.post(&endpoint)
            .bearer_auth(&self.id_token)
            .json(&body)
            .send()?;
        resp.error_for_status()?;
        Ok(())
    }
    pub fn push_progress_doc(&mut self, user: &str, project: &str, data: &crate::ProgressData, timestamp: &str) -> Result<()> {
        self.ensure_valid_token()?;
        let endpoint = format!(
            "https://firestore.googleapis.com/v1/projects/{}/databases/(default)/documents/users/{}/projects/{}/progress",
            self.project_id, user, project
        );
        let body = serde_json::json!({
            "fields": {
                "xp": { "integerValue": data.xp },
                "completedTasks": { "integerValue": data.completed_tasks },
                "totalTasks": { "integerValue": data.total_tasks },
                "timestamp": { "stringValue": timestamp }
            }
        });
        let client = reqwest::blocking::Client::new();
        let resp = client
            .post(&endpoint)
            .bearer_auth(&self.id_token)
            .json(&body)
            .send()?;
        resp.error_for_status()?;
        Ok(())
    }
    pub fn pull_project(&mut self, user: &str, project: &str) -> Result<()> {
        self.ensure_valid_token()?;
        use std::fs;
        use std::path::Path;
        // Ensure the project document exists before pulling
        self.ensure_project_doc(user, project)?;
        // Before pulling, clean the working directory except ignored files, .smartgit, .smartgitignore, and src
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
        // 1. Pull commits
        let commits_url = format!(
            "https://firestore.googleapis.com/v1/projects/{}/databases/(default)/documents/users/{}/projects/{}/commits",
            self.project_id, user, project
        );
        let client = reqwest::blocking::Client::new();
        let commits_resp = client.get(&commits_url)
            .bearer_auth(&self.id_token)
            .send()?;
        if !commits_resp.status().is_success() {
            return Err(anyhow::anyhow!("Failed to fetch commits: {}", commits_resp.status()));
        }
        let commits_json: serde_json::Value = commits_resp.json()?;
        let mut commits = Vec::new();
        if let Some(docs) = commits_json.get("documents").and_then(|v| v.as_array()) {
            for doc in docs {
                if let Some(fields) = doc.get("fields") {
                    let message = fields.get("message").and_then(|v| v.get("stringValue")).and_then(|v| v.as_str()).unwrap_or("").to_string();
                    let timestamp = fields.get("timestamp").and_then(|v| v.get("stringValue")).and_then(|v| v.as_str()).unwrap_or("").to_string();
                    let files = fields.get("files").and_then(|v| v.get("arrayValue")).and_then(|v| v.get("values")).and_then(|v| v.as_array()).map(|arr| {
                        arr.iter().filter_map(|f| {
                            let f = f.get("mapValue")?.get("fields")?;
                            Some(crate::IndexEntry {
                                filename: f.get("filename").and_then(|v| v.get("stringValue")).and_then(|v| v.as_str()).unwrap_or("").to_string(),
                                hash: f.get("hash").and_then(|v| v.get("stringValue")).and_then(|v| v.as_str()).unwrap_or("").to_string(),
                            })
                        }).collect::<Vec<_>>()
                    }).unwrap_or_default();
                    let hostname = fields.get("hostname").and_then(|v| v.get("stringValue")).and_then(|v| v.as_str()).unwrap_or("").to_string();
                    let os = fields.get("os").and_then(|v| v.get("stringValue")).and_then(|v| v.as_str()).unwrap_or("").to_string();
                    let username = fields.get("username").and_then(|v| v.get("stringValue")).and_then(|v| v.as_str()).unwrap_or("").to_string();
                    commits.push(crate::Commit { message, timestamp, files, hostname, os, username });
                }
            }
        }
        fs::create_dir_all(".smartgit").ok();
        fs::write(".smartgit/commits.json", serde_json::to_string_pretty(&commits).unwrap()).ok();
        // 2. Pull objects
        let objects_url = format!(
            "https://firestore.googleapis.com/v1/projects/{}/databases/(default)/documents/users/{}/projects/{}/objects/objects",
            self.project_id, user, project
        );
        let client = reqwest::blocking::Client::new();
        let objects_resp = client.get(&objects_url)
            .bearer_auth(&self.id_token)
            .send()?;
        if objects_resp.status().is_success() {
            let objects_json: serde_json::Value = objects_resp.json()?;
            if let Some(fields) = objects_json.get("fields").and_then(|v| v.as_object()) {
                fs::create_dir_all(".smartgit/objects").ok();
                for (hash, val) in fields {
                    if let Some(b64) = val.get("stringValue").and_then(|v| v.as_str()) {
                        let data = base64::engine::general_purpose::STANDARD.decode(b64).unwrap_or_default();
                        let obj_path = Path::new(".smartgit/objects").join(hash);
                        fs::write(obj_path, data).ok();
                    }
                }
            }
        } else if objects_resp.status().as_u16() == 404 {
            // Objects document does not exist, skip
            eprintln!("[smartgit] No objects found in Firestore for this project.");
        } else {
            let status = objects_resp.status();
            let err_text = objects_resp.text().unwrap_or_default();
            eprintln!("[smartgit] Failed to fetch objects: {}\n{}", status, err_text);
        }
        // 3. Pull latest progress (just get the most recent doc in /progress)
        let progress_url = format!(
            "https://firestore.googleapis.com/v1/projects/{}/databases/(default)/documents/users/{}/projects/{}/progress",
            self.project_id, user, project
        );
        let progress_resp = client.get(&progress_url)
            .bearer_auth(&self.id_token)
            .send()?;
        if progress_resp.status().is_success() {
            let progress_json: serde_json::Value = progress_resp.json()?;
            if let Some(docs) = progress_json.get("documents").and_then(|v| v.as_array()) {
                // Find the latest by timestamp
                let mut latest: Option<&serde_json::Value> = None;
                for doc in docs {
                    if let Some(fields) = doc.get("fields") {
                        if let Some(ts) = fields.get("timestamp").and_then(|v| v.get("stringValue")).and_then(|v| v.as_str()) {
                            let ts_opt = Some(ts);
                            let latest_ts_opt = latest.and_then(|d| d.get("fields")).and_then(|f| f.get("timestamp")).and_then(|v| v.get("stringValue")).and_then(|v| v.as_str());
                            if latest.is_none() || ts_opt > latest_ts_opt {
                                latest = Some(doc);
                            }
                        }
                    }
                }
                if let Some(doc) = latest {
                    if let Some(fields) = doc.get("fields") {
                        let xp = fields.get("xp").and_then(|v| v.get("integerValue")).and_then(|v| v.as_str()).and_then(|v| v.parse().ok()).unwrap_or(0);
                        let completed_tasks = fields.get("completedTasks").and_then(|v| v.get("integerValue")).and_then(|v| v.as_str()).and_then(|v| v.parse().ok()).unwrap_or(0);
                        let total_tasks = fields.get("totalTasks").and_then(|v| v.get("integerValue")).and_then(|v| v.as_str()).and_then(|v| v.parse().ok()).unwrap_or(0);
                        let progress = crate::ProgressData { xp, completed_tasks, total_tasks };
                        fs::write(".smartgit/progress.json", serde_json::to_string_pretty(&progress).unwrap()).ok();
                    }
                }
            }
        }
        // 4. Pull latest tasks (just get the most recent doc in /progress with type=="tasks")
        let tasks_url = format!(
            "https://firestore.googleapis.com/v1/projects/{}/databases/(default)/documents/users/{}/projects/{}/progress",
            self.project_id, user, project
        );
        let tasks_resp = client.get(&tasks_url)
            .bearer_auth(&self.id_token)
            .send()?;
        if tasks_resp.status().is_success() {
            let tasks_json: serde_json::Value = tasks_resp.json()?;
            if let Some(docs) = tasks_json.get("documents").and_then(|v| v.as_array()) {
                // Find the latest doc with type=="tasks" by timestamp
                let mut latest: Option<&serde_json::Value> = None;
                for doc in docs {
                    if let Some(fields) = doc.get("fields") {
                        let is_tasks = fields.get("type").and_then(|v| v.get("stringValue")).map(|v| v == "tasks").unwrap_or(false);
                        let ts = fields.get("timestamp").and_then(|v| v.get("stringValue")).and_then(|v| v.as_str());
                        let ts_opt = ts;
                        let latest_ts_opt = latest.and_then(|d| d.get("fields")).and_then(|f| f.get("timestamp")).and_then(|v| v.get("stringValue")).and_then(|v| v.as_str());
                        if is_tasks && ts_opt.is_some() {
                            if latest.is_none() || ts_opt > latest_ts_opt {
                                latest = Some(doc);
                            }
                        }
                    }
                }
                if let Some(doc) = latest {
                    if let Some(fields) = doc.get("fields") {
                        if let Some(json_str) = fields.get("json").and_then(|v| v.get("stringValue")).and_then(|v| v.as_str()) {
                            fs::write(".smartgit/tasks.json", json_str).ok();
                        }
                    }
                }
            }
        }

        // Create a map to track the most recent version of each file
        let mut latest_files: std::collections::HashMap<String, (String, String)> = std::collections::HashMap::new();
        
        // Iterate through commits in chronological order
        for commit in commits.iter() {
            for entry in &commit.files {
                // Update the map with the most recent version of each file
                latest_files.insert(entry.filename.clone(), (entry.hash.clone(), commit.timestamp.clone()));
            }
        }

        // Restore the most recent version of each file
        for (filename, (hash, _)) in latest_files {
            let obj_path = Path::new(".smartgit/objects").join(&hash);
            if obj_path.exists() {
                if let Ok(data) = fs::read(&obj_path) {
                    let file_path = Path::new(&filename);
                    // Create parent directories if needed
                    if let Some(parent) = file_path.parent() {
                        let _ = fs::create_dir_all(parent);
                    }
                    let _ = fs::write(&file_path, &data);
                }
            }
        }

        Ok(())
    }
    pub fn list_projects(&mut self, user: &str) -> Result<Vec<String>> {
        self.ensure_valid_token()?;
        // List all project documents under /users/{user}/projects
        let url = format!(
            "https://firestore.googleapis.com/v1/projects/{}/databases/(default)/documents/users/{}/projects",
            self.project_id, user
        );
        let client = reqwest::blocking::Client::new();
        let resp = client.get(&url)
            .bearer_auth(&self.id_token)
            .send()?;
        if !resp.status().is_success() {
            let status = resp.status();
            let err_text = resp.text().unwrap_or_default();
            eprintln!("[smartgit] Failed to fetch projects: {}\nURL: {}\nResponse: {}", status, url, err_text);
            return Err(anyhow::anyhow!("Failed to fetch projects: {}", status));
        }
        let json: serde_json::Value = resp.json()?;
        let mut projects: Vec<String> = Vec::new();
        if let Some(docs) = json.get("documents").and_then(|v| v.as_array()) {
            for doc in docs {
                if let Some(name) = doc.get("name").and_then(|v| v.as_str()) {
                    // name is like .../users/{user}/projects/{project}
                    let parts: Vec<&str> = name.split('/').collect();
                    if let Some(proj) = parts.last() {
                        let proj_str = proj.to_string();
                        if !projects.contains(&proj_str) {
                            projects.push(proj_str);
                        }
                    }
                }
            }
        }
        Ok(projects)
    }
    /// Ensure the project document exists at /users/{user}/projects/{project}
    pub fn ensure_project_doc(&mut self, user: &str, project: &str) -> Result<()> {
        self.ensure_valid_token()?;
        let endpoint = format!(
            "https://firestore.googleapis.com/v1/projects/{}/databases/(default)/documents/users/{}/projects/{}",
            self.project_id, user, project
        );
        let body = serde_json::json!({ "fields": {} }); // empty document
        let client = reqwest::blocking::Client::new();
        let resp = client
            .patch(&endpoint)
            .bearer_auth(&self.id_token)
            .json(&body)
            .send()?;
        resp.error_for_status()?;
        Ok(())
    }
}

pub struct UserProgress {
    #[allow(dead_code)]
    pub projects: Vec<(String, ProgressData)>,
}

impl ProgressSync for FirestoreSync {
    fn push(&mut self, user: &str, proj: &str, data: &crate::ProgressData) -> Result<()> {
        self.ensure_valid_token()?;
        use chrono::Utc;
        let endpoint = format!(
            "https://firestore.googleapis.com/v1/projects/{}/databases/(default)/documents/users/{}/projects/{}",
            self.project_id, user, proj
        );
        let body = serde_json::json!({
            "fields": {
                "xp": { "integerValue": data.xp },
                "completedTasks": { "integerValue": data.completed_tasks },
                "totalTasks": { "integerValue": data.total_tasks },
                "updatedAt": { "timestampValue": Utc::now().to_rfc3339() },
            }
        });
        let client = reqwest::blocking::Client::new();
        let resp = client
            .patch(&endpoint)
            .bearer_auth(&self.id_token)
            .json(&body)
            .send()?;
        resp.error_for_status()?;
        Ok(())
    }
    fn pull(&mut self, _user: &str) -> Result<UserProgress> {
        self.ensure_valid_token()?;
        unimplemented!()
    }
}
